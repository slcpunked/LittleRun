using System.Diagnostics;
using System.Net;
using System.Net.Http.Json;
using System.Net.Sockets;
using System.Text.Json;
using LittleRun.Desktop.Setup;

namespace LittleRun.Desktop.Backend;

public sealed record BackendStatus(string State, string Message, int Port = 0, string? Device = null)
{
    public static readonly BackendStatus Stopped = new("stopped", "Not running");
}

/// <summary>Runs `littlerun serve` for the installed model and keeps the UI told about its state.</summary>
/// <remarks>States: stopped, starting, loading, ready, error.</remarks>
public sealed class BackendProcess : IDisposable
{
    private readonly AppPaths _paths;
    private readonly object _gate = new();
    private Process? _proc;
    private CancellationTokenSource? _watch;
    private StreamWriter? _log;
    private readonly Queue<string> _tail = new();
    private BackendStatus _status = BackendStatus.Stopped;
    private bool _stopping;

    public BackendProcess(AppPaths paths)
    {
        _paths = paths;
    }

    public event Action<BackendStatus>? StatusChanged;

    public BackendStatus Status { get { lock (_gate) return _status; } }

    public string BaseUrl => $"http://127.0.0.1:{Status.Port}";

    /// <summary>Arguments for `python -m littlerun serve`. Public for tests.</summary>
    internal static List<string> ServeArgs(string modelDir, string modelId, int port, string device, int maxContext) =>
    [
        "-m", "littlerun", "serve", modelDir,
        "--host", "127.0.0.1",
        "--port", port.ToString(System.Globalization.CultureInfo.InvariantCulture),
        "--model-id", modelId,
        "--device", device,
        "--max-context", maxContext.ToString(System.Globalization.CultureInfo.InvariantCulture),
        "--max-tokens-limit", "2048",
        "--api-key-env", "LITTLERUN_API_KEY",
    ];

    /// <summary>The preferred port if it is free, otherwise any free loopback port.</summary>
    internal static int PickPort(int preferred)
    {
        if (preferred > 0 && IsFree(preferred))
            return preferred;
        var l = new TcpListener(IPAddress.Loopback, 0);
        l.Start();
        var p = ((IPEndPoint)l.LocalEndpoint).Port;
        l.Stop();
        return p;
    }

    private static bool IsFree(int port)
    {
        try
        {
            var l = new TcpListener(IPAddress.Loopback, port);
            l.Start();
            l.Stop();
            return true;
        }
        catch (SocketException)
        {
            return false;
        }
    }

    public void Start(Settings s, ModelSpec spec)
    {
        Stop();
        if (s.ModelDir == null || !File.Exists(_paths.VenvPython))
        {
            Set(new BackendStatus("error", "Setup has not finished"));
            return;
        }
        var port = PickPort(s.Port);
        Directory.CreateDirectory(_paths.Logs);
        if (File.Exists(_paths.BackendLog))
            File.Copy(_paths.BackendLog, Path.Combine(_paths.Logs, "backend.prev.log"), true);
        var log = new StreamWriter(new FileStream(_paths.BackendLog, FileMode.Create, FileAccess.Write, FileShare.ReadWrite))
            { AutoFlush = true };
        var psi = ProcessRunner.StartInfo(_paths.VenvPython, ServeArgs(s.ModelDir, spec.ModelId, port, s.Device, s.MaxContext),
                                          RuntimeInstaller.BackendEnv(_paths, s.ApiKey), _paths.Runtime);
        var p = new Process { StartInfo = psi, EnableRaisingEvents = true };
        var cts = new CancellationTokenSource();
        lock (_gate)
        {
            _stopping = false;
            _tail.Clear();
            _log = log;
            _proc = p;
            _watch = cts;
        }
        p.OutputDataReceived += (_, e) => OnLine(e.Data);
        p.ErrorDataReceived += (_, e) => OnLine(e.Data);
        p.Exited += (_, _) => OnExit(p);
        Set(new BackendStatus("starting", "Starting the backend", port, s.Device));
        log.WriteLine($"# {DateTime.Now:u} {psi.FileName} {string.Join(' ', psi.ArgumentList)}");
        try
        {
            p.Start();
        }
        catch (System.ComponentModel.Win32Exception e)
        {
            Set(new BackendStatus("error", $"Could not start Python: {e.Message}", port));
            return;
        }
        ChildJob.Add(p);
        p.StandardInput.Close();
        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        _ = WatchHealthAsync(port, s.Device, cts.Token);
    }

    private void OnLine(string? line)
    {
        if (line == null)
            return;
        lock (_gate)
        {
            _log?.WriteLine(line);
            _tail.Enqueue(line);
            while (_tail.Count > 30) _tail.Dequeue();
        }
        // Loader progress lines look like "[load] decoding 252 factorized linears".
        var i = line.IndexOf("[load] ", StringComparison.Ordinal);
        var st = Status;
        if (i >= 0 && st.State is "starting" or "loading")
            Set(st with { State = "loading", Message = Capitalize(line[(i + 7)..]) });
    }

    private void OnExit(Process p)
    {
        bool stopping;
        string tail;
        lock (_gate)
        {
            if (!ReferenceEquals(p, _proc))
                return;
            stopping = _stopping;
            tail = string.Join("\n", _tail);
            _watch?.Cancel();
        }
        if (stopping)
        {
            Set(BackendStatus.Stopped);
            return;
        }
        int code;
        try { code = p.ExitCode; } catch (InvalidOperationException) { code = -1; }
        Set(new BackendStatus("error", $"The backend stopped (exit {code}). {LastError(tail)}", Status.Port));
    }

    /// <summary>The most useful line of a failed run: a raised error, else the last line.</summary>
    internal static string LastError(string tail)
    {
        var lines = tail.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var err = lines.LastOrDefault(l => l.Contains("Error", StringComparison.Ordinal) ||
                                           l.StartsWith("SystemExit", StringComparison.Ordinal));
        return err ?? lines.LastOrDefault() ?? "";
    }

    private async Task WatchHealthAsync(int port, string device, CancellationToken ct)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
        var url = $"http://127.0.0.1:{port}/health";
        while (!ct.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(Status.State == "ready" ? 5000 : 700, ct);
                var h = await http.GetFromJsonAsync<JsonElement>(url, ct);
                var ready = h.TryGetProperty("ready", out var r) && r.GetBoolean();
                var error = h.TryGetProperty("error", out var e) && e.ValueKind == JsonValueKind.String ? e.GetString() : null;
                if (error != null && Status.State != "error")
                    Set(new BackendStatus("error", "The model failed to load: " + error, port, device));
                else if (ready && Status.State != "ready")
                    Set(new BackendStatus("ready", "Ready", port, device));
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                return;
            }
            catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or JsonException)
            {
                // Not listening yet (still importing torch) or briefly busy; keep polling.
            }
        }
    }

    public void Stop()
    {
        Process? p;
        lock (_gate)
        {
            p = _proc;
            _stopping = true;
            _watch?.Cancel();
        }
        if (p != null)
        {
            try
            {
                if (!p.HasExited)
                {
                    p.Kill(entireProcessTree: true);
                    p.WaitForExit(5000);
                }
            }
            catch (InvalidOperationException) { }
            catch (System.ComponentModel.Win32Exception) { }
        }
        lock (_gate)
        {
            _proc?.Dispose();
            _proc = null;
            _log?.Dispose();
            _log = null;
        }
        Set(BackendStatus.Stopped);
    }

    private void Set(BackendStatus s)
    {
        lock (_gate)
        {
            if (_status == s)
                return;
            _status = s;
        }
        StatusChanged?.Invoke(s);
    }

    private static string Capitalize(string s) => s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s[1..];

    public void Dispose() => Stop();
}
