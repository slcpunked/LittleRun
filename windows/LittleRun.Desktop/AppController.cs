using System.Collections.Concurrent;
using System.Text.Json.Nodes;
using LittleRun.Desktop.Backend;
using LittleRun.Desktop.Setup;

namespace LittleRun.Desktop;

/// <summary>What the UI asks setup to do.</summary>
/// <param name="Device">"cpu" or "cuda".</param>
/// <param name="Model">Pinned model name.</param>
/// <param name="ImportDir">Use this existing folder instead of downloading.</param>
public sealed record SetupRequest(string Device, string Model, string? ImportDir, bool AcceptLicense);

/// <summary>One generation request from the UI.</summary>
/// <param name="Character">Story chat character for this chat (preset id or "custom"); null uses the setting.</param>
public sealed record GenRequest(string ReqId, string Mode, List<(string role, string content)> Messages, string? Prompt,
                                bool? Thinking, string? Character = null);

/// <summary>Everything the app does apart from drawing: setup, backend lifetime, generation, persistence.</summary>
public sealed class AppController : IDisposable
{
    private readonly ConcurrentDictionary<string, CancellationTokenSource> _gens = new();
    private readonly HttpClient _api = new() { Timeout = Timeout.InfiniteTimeSpan };
    private readonly HttpClient _downloads = ModelInstaller.CreateHttp();
    private readonly SemaphoreSlim _setupGate = new(1, 1);
    private readonly object _chatsGate = new();
    private Lazy<Payload> _payload;
    private CancellationTokenSource? _setupCts;

    public AppController(AppPaths paths)
    {
        Paths = paths;
        paths.EnsureBase();
        Settings = Settings.Load(paths);
        Backend = new BackendProcess(paths);
        Backend.StatusChanged += s =>
        {
            Emit("backend", s);
            if (s.State == "ready")
                _ = WarmStoryAsync();
        };
        _payload = new Lazy<Payload>(() => Payload.Unpack(paths));
    }

    public AppPaths Paths { get; }
    public Settings Settings { get; private set; }
    public BackendProcess Backend { get; }
    public Payload Payload => _payload.Value;

    /// <summary>Events for the UI: (name, data).</summary>
    public event Action<string, object>? Event;

    private void Emit(string name, object data) => Event?.Invoke(name, data);

    /// <summary>Use an unpacked payload folder instead of the embedded one (development and tests).</summary>
    public void UsePayloadDirectory(string dir) => _payload = new Lazy<Payload>(() => Payload.FromDirectory(dir));

    public ModelSpec? CurrentModel =>
        Settings.ModelName != null && Payload.Info.Models.TryGetValue(Settings.ModelName, out var m) ? m : null;

    /// <summary>True when the runtime matches this build and an installed model folder is recorded.</summary>
    public bool IsSetUp =>
        CurrentModel is { } m && Settings.ModelDir != null && Directory.Exists(Settings.ModelDir) &&
        ModelInstaller.MissingFiles(m, Settings.ModelDir).Count == 0 &&
        new RuntimeInstaller(Paths, Payload).IsCurrent(Settings.Device);

    public void StartBackendIfReady()
    {
        if (IsSetUp)
            Backend.Start(Settings, CurrentModel!);
    }

    public async Task RunSetupAsync(SetupRequest req, CancellationToken outer = default)
    {
        if (!await _setupGate.WaitAsync(0, outer))
            throw new InvalidOperationException("Setup is already running");
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(outer);
        _setupCts = cts;
        using var r = new SetupReporter(Paths.SetupLog, e => Emit("setup", e));
        try
        {
            if (!Payload.Info.Models.TryGetValue(req.Model, out var spec))
                throw new SetupFailedException("model", $"Unknown model '{req.Model}'");
            if (!req.AcceptLicense)
                throw new SetupFailedException("model", $"The {spec.License} license must be accepted first.");
            var device = req.Device == "cuda" ? "cuda" : "cpu";
            if (device == "cuda" &&
                SystemProbe.GpuProblem(SystemProbe.Probe(Paths.Root).NvidiaGpu, Payload.Info.CudaMinDriver) is { } gpuProblem)
                throw new SetupFailedException("torch", gpuProblem);
            r.Log("runtime", $"--- setup {DateTime.Now:u}: device={device} model={req.Model} import={req.ImportDir ?? "-"}");

            Backend.Stop();
            var runtime = new RuntimeInstaller(Paths, Payload);
            if (runtime.IsCurrent(device))
            {
                foreach (var step in new[] { "runtime", "torch", "backend", "kernel", "verify" })
                    r.Done(step, "Already installed");
            }
            else
            {
                await runtime.InstallAsync(device, r, cts.Token);
            }

            var models = new ModelInstaller(Paths, Payload, _downloads);
            string dir;
            if (!string.IsNullOrWhiteSpace(req.ImportDir))
            {
                dir = Path.GetFullPath(req.ImportDir);
                r.Done("model", $"Using {dir}");
            }
            else
            {
                dir = models.DefaultDir(req.Model);
                if (ModelInstaller.MissingFiles(spec, dir).Count == 0)
                {
                    models.InstallManifest(spec, dir);
                    r.Done("model", "Already downloaded");
                }
                else
                {
                    await models.DownloadAsync(req.Model, dir, r, cts.Token);
                }
            }
            await models.ValidateAsync(spec, dir, r, cts.Token);

            Settings.Device = device;
            Settings.ModelName = req.Model;
            Settings.ModelDir = dir;
            Settings.LicenseAccepted = true;
            Settings.Save(Paths);
            r.Report(new SetupEvent("finish", "done", "Setup complete"));
            Backend.Start(Settings, spec);
        }
        catch (OperationCanceledException)
        {
            r.Report(new SetupEvent("finish", "failed", "Setup was cancelled"));
            throw;
        }
        catch (SetupFailedException e)
        {
            r.Report(new SetupEvent(e.Step, "failed", e.Message));
            r.Report(new SetupEvent("finish", "failed", e.Message));
            throw;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or HttpRequestException
                                       or InvalidOperationException or System.ComponentModel.Win32Exception)
        {
            r.Report(new SetupEvent("finish", "failed", e.Message));
            throw;
        }
        finally
        {
            _setupCts = null;
            _setupGate.Release();
        }
    }

    public void CancelSetup() => _setupCts?.Cancel();

    /// <summary>Stream one generation; deltas and the result go out as gen.* events.</summary>
    public async Task<GenerationResult> GenerateAsync(GenRequest g)
    {
        var spec = CurrentModel ?? throw new ApiException("No model is installed");
        if (Backend.Status.State != "ready")
            throw new ApiException(Backend.Status.State == "error" ? Backend.Status.Message : "The model is still loading");
        using var cts = new CancellationTokenSource();
        _gens[g.ReqId] = cts;
        try
        {
            var s = Settings;
            var opts = new ApiClient.Options(s.MaxTokens, s.Temperature, s.TopP, s.TopK, s.RepetitionPenalty,
                                             g.Thinking ?? s.Thinking);
            var client = new ApiClient(_api, Backend.BaseUrl, s.ApiKey, spec.ModelId);
            void Delta(string t) => Emit("gen.delta", new { reqId = g.ReqId, text = t });
            if (g.Mode == "complete" || !s.UseStory(spec))
                _warmKey = null; // this request replaces the backend's cached story prefix
            if (g.Mode == "complete")
                return await client.CompleteAsync(g.Prompt ?? "", opts with { Temperature = Math.Min(opts.Temperature, 0.7) },
                                                  Delta, cts.Token);
            if (s.UseStory(spec))
            {
                // Models without chat training continue a story instead (see StoryChat).
                var character = StoryChat.Resolve(g.Character ?? s.Character, s.CustomCharacter);
                var maxChars = Math.Max(2000, (s.MaxContext - StoryChat.MaxReplyTokens - 64) * 3); // ~3 chars per token
                var prompt = StoryChat.BuildPrompt(character, s.UserName, g.Messages, maxChars);
                return await client.CompleteAsync(prompt,
                    opts with { MaxTokens = Math.Min(opts.MaxTokens, StoryChat.MaxReplyTokens), Stop = StoryChat.Stop },
                    Delta, cts.Token);
            }
            var msgs = new List<(string role, string content)>();
            if (!string.IsNullOrWhiteSpace(s.SystemPrompt))
                msgs.Add(("system", s.SystemPrompt));
            msgs.AddRange(g.Messages);
            return await client.ChatAsync(msgs, opts, Delta, cts.Token);
        }
        finally
        {
            _gens.TryRemove(g.ReqId, out _);
        }
    }

    private string? _warmKey;
    private int _warming;

    /// <summary>
    /// Story chat sends the character description and examples (several hundred tokens) with every message. The
    /// backend reuses the cached start of the previous prompt, so reading that part ahead of time makes the first
    /// reply as quick as later ones (on CPU it otherwise takes about a minute).
    /// </summary>
    public async Task WarmStoryAsync(bool force = false)
    {
        var spec = CurrentModel;
        var s = Settings;
        if (spec == null || !s.UseStory(spec) || Backend.Status.State != "ready")
            return;
        var character = StoryChat.Resolve(s.Character, s.CustomCharacter);
        var prompt = StoryChat.BuildPrompt(character, s.UserName, [], int.MaxValue);
        var key = prompt + "|" + Backend.Status.Port;
        if ((!force && key == _warmKey) || Interlocked.Exchange(ref _warming, 1) == 1)
            return;
        Emit("warmup", new { running = true, name = character.Name });
        try
        {
            var client = new ApiClient(_api, Backend.BaseUrl, s.ApiKey, spec.ModelId);
            await client.CompleteAsync(prompt, new ApiClient.Options(1, 0, 1, 0, 1, false), _ => { }, CancellationToken.None);
            _warmKey = key;
        }
        catch (Exception e) when (e is HttpRequestException or ApiException or TaskCanceledException)
        {
            // Only a speed-up; the real request still works without it.
        }
        finally
        {
            Interlocked.Exchange(ref _warming, 0);
            Emit("warmup", new { running = false, name = character.Name });
        }
    }

    public void StopGeneration(string reqId)
    {
        if (_gens.TryGetValue(reqId, out var cts))
            cts.Cancel();
    }

    /// <summary>Apply settings from the UI. Restarts the backend when a load-time option changed.</summary>
    public bool UpdateSettings(Settings next)
    {
        next.Normalize();
        next.ApiKey = string.IsNullOrWhiteSpace(next.ApiKey) ? Settings.ApiKey : next.ApiKey;
        // Setup owns these; the settings dialog cannot point them elsewhere.
        next.ModelDir = Settings.ModelDir;
        next.ModelName = Settings.ModelName;
        next.LicenseAccepted = Settings.LicenseAccepted;
        next.Device = Settings.Device;
        var restart = next.MaxContext != Settings.MaxContext || next.Port != Settings.Port || next.ApiKey != Settings.ApiKey;
        Settings = next;
        Settings.Save(Paths);
        if (restart && IsSetUp)
            Backend.Start(Settings, CurrentModel!); // warms up again once ready
        else
            _ = WarmStoryAsync(); // no-op unless the character or name changed
        return restart;
    }

    public JsonArray LoadChats()
    {
        lock (_chatsGate)
        {
            try
            {
                return File.Exists(Paths.Chats) ? JsonNode.Parse(File.ReadAllText(Paths.Chats)) as JsonArray ?? [] : [];
            }
            catch (System.Text.Json.JsonException)
            {
                File.Move(Paths.Chats, Paths.Chats + ".corrupt-" + DateTime.Now.ToString("yyyyMMdd-HHmmss"), true);
                return [];
            }
        }
    }

    public void SaveChats(JsonArray chats)
    {
        lock (_chatsGate)
        {
            var tmp = Paths.Chats + ".tmp";
            File.WriteAllText(tmp, chats.ToJsonString(JsonFile.Options));
            File.Move(tmp, Paths.Chats, true);
        }
    }

    public void Dispose()
    {
        foreach (var c in _gens.Values) c.Cancel();
        _setupCts?.Cancel();
        Backend.Dispose();
        _api.Dispose();
        _downloads.Dispose();
    }
}
