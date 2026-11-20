using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;

namespace LittleRun.Desktop.Setup;

/// <summary>Downloads a pinned checkpoint (resumable, sha256-checked) or adopts an existing folder.</summary>
public sealed class ModelInstaller(AppPaths paths, Payload payload, HttpClient http)
{
    public const string ManifestName = "littlerun.manifest.json";

    public static HttpClient CreateHttp()
    {
        var h = new HttpClient(new SocketsHttpHandler
        {
            AllowAutoRedirect = true,
            MaxAutomaticRedirections = 10,
            ConnectTimeout = TimeSpan.FromSeconds(30),
            AutomaticDecompression = DecompressionMethods.None,
        })
        { Timeout = Timeout.InfiniteTimeSpan };
        h.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("LittleRun", payload_version()));
        return h;

        static string payload_version() =>
            typeof(ModelInstaller).Assembly.GetName().Version?.ToString(3) ?? "0";
    }

    public string DefaultDir(string name) => Path.Combine(paths.Models, name);

    /// <summary>Files of <paramref name="spec"/> missing from <paramref name="dir"/>.</summary>
    public static List<string> MissingFiles(ModelSpec spec, string dir) =>
        spec.Files.Where(f => !File.Exists(Path.Combine(dir, f))).ToList();

    public async Task DownloadAsync(string name, string dir, SetupReporter r, CancellationToken ct)
    {
        var spec = payload.Info.Models[name];
        Directory.CreateDirectory(dir);
        var free = new DriveInfo(Path.GetPathRoot(Path.GetFullPath(dir))!).AvailableFreeSpace;
        long have = spec.Files.Select(f => Path.Combine(dir, f)).Where(File.Exists).Sum(f => new FileInfo(f).Length);
        var need = Math.Max(0, spec.SizeBytes - have) + 512L * 1024 * 1024;
        if (free < need)
            throw new SetupFailedException("model",
                $"Not enough disk space: {Fmt(need)} needed, {Fmt(free)} free on {Path.GetPathRoot(dir)}.");

        long done = have, total = Math.Max(spec.SizeBytes, have);
        var sw = Stopwatch.StartNew();
        var last = TimeSpan.Zero;
        long lastBytes = done;
        double rate = 0;
        r.Running("model", $"Downloading {spec.DisplayName}", total > 0 ? (double)done / total : null);

        foreach (var file in spec.Files)
        {
            var dest = Path.Combine(dir, file);
            if (File.Exists(dest))
                continue;
            r.Log("model", $"downloading {file}");
            await DownloadFileAsync(spec.FileUrl(file), dest, delta =>
            {
                done += delta;
                if (done > total) total = done;
                var now = sw.Elapsed;
                if (now - last < TimeSpan.FromMilliseconds(250))
                    return;
                var inst = (done - lastBytes) / Math.Max((now - last).TotalSeconds, 1e-3);
                rate = rate == 0 ? inst : rate * 0.7 + inst * 0.3;
                last = now;
                lastBytes = done;
                r.Report(new SetupEvent("model", "running", $"Downloading {spec.DisplayName}",
                                        total > 0 ? (double)done / total : null, done, total, rate));
            }, ct);
            if (spec.Sha256.TryGetValue(file, out var want))
            {
                r.Running("model", $"Checking {file}", total > 0 ? (double)done / total : null);
                var got = await Sha256Async(dest, ct);
                if (!string.Equals(got, want, StringComparison.OrdinalIgnoreCase))
                {
                    File.Delete(dest);
                    throw new SetupFailedException("model", $"{file} failed its sha256 check and was deleted. Try again.");
                }
            }
        }
        InstallManifest(spec, dir);
        r.Done("model", $"Downloaded {spec.DisplayName}");
    }

    /// <summary>Download one file to dest, resuming a previous dest.part. Reports bytes as they arrive.</summary>
    internal async Task DownloadFileAsync(string url, string dest, Action<long> onBytes, CancellationToken ct)
    {
        var part = dest + ".part";
        for (var attempt = 1; ; attempt++)
        {
            long offset = File.Exists(part) ? new FileInfo(part).Length : 0;
            try
            {
                using var req = new HttpRequestMessage(HttpMethod.Get, url);
                if (offset > 0)
                    req.Headers.Range = new RangeHeaderValue(offset, null);
                using var resp = await http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
                if (resp.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable && offset > 0)
                {
                    File.Delete(part); // stale partial file; start over
                    continue;
                }
                resp.EnsureSuccessStatusCode();
                var append = offset > 0 && resp.StatusCode == HttpStatusCode.PartialContent;
                if (offset > 0 && !append)
                    onBytes(-offset); // server ignored the range: count from zero again
                await using (var body = await resp.Content.ReadAsStreamAsync(ct))
                await using (var f = new FileStream(part, append ? FileMode.Append : FileMode.Create, FileAccess.Write,
                                                    FileShare.None, 1 << 20))
                {
                    var buf = new byte[1 << 20];
                    int n;
                    while ((n = await body.ReadAsync(buf, ct)) > 0)
                    {
                        await f.WriteAsync(buf.AsMemory(0, n), ct);
                        onBytes(n);
                    }
                }
                File.Move(part, dest, overwrite: true);
                return;
            }
            catch (Exception e) when (attempt < 5 && !ct.IsCancellationRequested &&
                                      e is HttpRequestException or IOException)
            {
                await Task.Delay(TimeSpan.FromSeconds(2 * attempt), ct); // resume from what arrived
            }
        }
    }

    /// <summary>Copy the reviewed manifest shipped with this build into the checkpoint folder.</summary>
    public void InstallManifest(ModelSpec spec, string dir)
    {
        var dest = Path.Combine(dir, ManifestName);
        if (!File.Exists(dest))
            File.Copy(payload.ManifestPath(spec), dest);
    }

    /// <summary>Run `littlerun validate` (sha256 of every file against the manifest, codec and shape checks).</summary>
    public async Task ValidateAsync(ModelSpec spec, string dir, SetupReporter r, CancellationToken ct)
    {
        var missing = MissingFiles(spec, dir);
        if (missing.Count > 0)
            throw new SetupFailedException("validate", $"The folder is missing {string.Join(", ", missing)}.");
        InstallManifest(spec, dir);
        r.Running("validate", "Verifying the model files (sha256)");
        string? ok = null;
        var tail = new Queue<string>();
        var code = await ProcessRunner.RunAsync(
            ProcessRunner.StartInfo(paths.VenvPython, ["-m", "littlerun", "validate", dir],
                                    RuntimeInstaller.BackendEnv(paths, null), paths.Runtime),
            line =>
            {
                r.Log("validate", line);
                if (line.StartsWith("OK:", StringComparison.Ordinal)) ok = line;
                tail.Enqueue(line);
                if (tail.Count > 4) tail.Dequeue();
            }, ct);
        if (code != 0 || ok == null)
            throw new SetupFailedException("validate", "The model files did not pass validation: " + string.Join(" ", tail));
        r.Done("validate", "Model verified");
    }

    internal static async Task<string> Sha256Async(string path, CancellationToken ct)
    {
        await using var f = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, true);
        var h = await SHA256.HashDataAsync(f, ct);
        return Convert.ToHexString(h).ToLowerInvariant();
    }

    public static string Fmt(long bytes) => bytes switch
    {
        >= 1L << 30 => $"{bytes / (double)(1L << 30):0.0} GB",
        >= 1L << 20 => $"{bytes / (double)(1L << 20):0} MB",
        _ => $"{bytes / 1024.0:0} KB",
    };
}
