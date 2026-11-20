using System.Text.Json;

namespace LittleRun.Desktop.Setup;

/// <summary>What installed.json records about the managed Python runtime.</summary>
public sealed class RuntimeStamp
{
    public string PayloadHash { get; set; } = "";
    public string Device { get; set; } = "";
    public string Torch { get; set; } = "";
    public string Python { get; set; } = "";
    public bool KernelOk { get; set; }
    public bool CudaAvailable { get; set; }
    public DateTime InstalledAt { get; set; }
}

/// <summary>Installs a private Python + PyTorch + LittleRun backend under runtime\ with the bundled uv.</summary>
public sealed class RuntimeInstaller(AppPaths paths, Payload payload)
{
    public static RuntimeStamp? ReadStamp(AppPaths paths) => JsonFile.Read<RuntimeStamp>(paths.RuntimeStamp);

    /// <summary>True when the runtime matches this build and the chosen device.</summary>
    public bool IsCurrent(string device)
    {
        var s = ReadStamp(paths);
        return s != null && s.PayloadHash == payload.Hash && s.Device == device && File.Exists(paths.VenvPython);
    }

    /// <summary>Environment for every uv call: keep Python, downloads and caches inside runtime\.</summary>
    internal Dictionary<string, string?> UvEnv() => new()
    {
        ["UV_PYTHON_INSTALL_DIR"] = paths.PythonInstalls,
        ["UV_CACHE_DIR"] = paths.UvCache,
        ["UV_PYTHON_PREFERENCE"] = "only-managed",
        ["UV_PYTHON_DOWNLOADS"] = "automatic",
        ["UV_LINK_MODE"] = "copy",
        ["UV_HTTP_TIMEOUT"] = "120",
    };

    /// <summary>The uv commands setup runs, in order. Public for tests.</summary>
    internal IReadOnlyList<(string step, string message, string[] args)> Plan(string device)
    {
        var index = payload.Info.TorchIndexes.TryGetValue(device, out var u) ? u
            : throw new SetupFailedException("torch", $"no PyTorch index for device '{device}'");
        var py = paths.VenvPython;
        var torchSize = device == "cuda" ? "about 3 GB" : "about 250 MB";
        return
        [
            ("runtime", $"Installing Python {payload.Info.Python}",
                ["venv", paths.Venv, "--python", payload.Info.Python, "--clear", "--no-project"]),
            ("torch", $"Installing PyTorch {payload.Info.Torch} ({(device == "cuda" ? "NVIDIA GPU" : "CPU")}, {torchSize})",
                ["pip", "install", "--python", py, "--index-url", index, $"torch=={payload.Info.Torch}"]),
            ("backend", "Installing the LittleRun backend",
                ["pip", "install", "--python", py, "-r", payload.RequirementsPath]),
            ("backend", "Installing the LittleRun backend",
                ["pip", "install", "--python", py, "--no-deps", "--reinstall-package", "littlerun", payload.WheelPath]),
        ];
    }

    public async Task<RuntimeStamp> InstallAsync(string device, SetupReporter r, CancellationToken ct)
    {
        Directory.CreateDirectory(paths.Runtime);
        var env = UvEnv();
        string? lastStep = null;
        foreach (var (step, message, args) in Plan(device))
        {
            if (step != lastStep)
            {
                if (lastStep != null)
                    r.Done(lastStep, "Done");
                r.Running(step, message);
                lastStep = step;
            }
            var tail = new Queue<string>();
            var code = await ProcessRunner.RunAsync(ProcessRunner.StartInfo(payload.UvExe, args, env, paths.Runtime), line =>
            {
                r.Log(step, line);
                tail.Enqueue(line);
                if (tail.Count > 8) tail.Dequeue();
            }, ct);
            if (code != 0)
                throw new SetupFailedException(step, $"{message} failed (uv exit {code}). {string.Join(" ", tail).Trim()}");
        }
        r.Done(lastStep!, "Done");

        r.Running("kernel", "Installing the fast CPU kernel");
        if (Directory.Exists(paths.Kernel))
            Directory.Delete(paths.Kernel, true);
        Directory.CreateDirectory(paths.Kernel);
        foreach (var rel in payload.Info.Kernel)
        {
            var src = Path.Combine(payload.Dir, rel.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(src))
                File.Copy(src, Path.Combine(paths.Kernel, Path.GetFileName(src)), true);
        }
        r.Done("kernel", payload.Info.Kernel.Count > 0 ? "Done" : "Not included in this build (slower CPU path)");

        r.Running("verify", "Checking the runtime");
        var facts = new Dictionary<string, string>();
        var verifyCode = await ProcessRunner.RunAsync(
            ProcessRunner.StartInfo(paths.VenvPython, ["-m", "littlerun", "doctor"], BackendEnv(paths, null), paths.Runtime),
            line =>
            {
                r.Log("verify", line);
                var sp = line.IndexOf(' ');
                if (sp > 0 && !line.Contains("INFO") && !line.Contains("WARNING"))
                    facts[line[..sp]] = line[sp..].Trim();
            }, ct);
        if (verifyCode != 0)
            throw new SetupFailedException("verify", "The runtime check failed. See setup.log for details.");
        var stamp = new RuntimeStamp
        {
            PayloadHash = payload.Hash,
            Device = device,
            Torch = facts.GetValueOrDefault("torch", ""),
            Python = facts.GetValueOrDefault("python", ""),
            KernelOk = facts.GetValueOrDefault("packed_kernel", "").StartsWith("ok", StringComparison.Ordinal),
            CudaAvailable = facts.GetValueOrDefault("cuda_available", "") == "True",
            InstalledAt = DateTime.UtcNow,
        };
        if (device == "cuda" && !stamp.CudaAvailable)
            throw new SetupFailedException("verify",
                "PyTorch cannot use the NVIDIA GPU (CUDA unavailable). Update the NVIDIA driver, or choose CPU in setup.");
        JsonFile.Write(paths.RuntimeStamp, stamp);
        r.Done("verify", $"PyTorch {stamp.Torch}, Python {stamp.Python}" +
                         (stamp.KernelOk ? ", fast kernel ready" : ", fast kernel unavailable") +
                         (device == "cuda" ? ", GPU ready" : ""));

        // The downloaded wheels are no longer needed; PyTorch alone can be several GB.
        try { Directory.Delete(paths.UvCache, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        return stamp;
    }

    /// <summary>Environment for the backend process (doctor, validate, serve).</summary>
    public static Dictionary<string, string?> BackendEnv(AppPaths paths, string? apiKey)
    {
        var env = new Dictionary<string, string?>
        {
            ["LITTLERUN_KERNEL_DIR"] = paths.Kernel,
            ["LITTLERUN_BUILD_DIR"] = Path.Combine(paths.Runtime, "kernel-build"),
            // Model files are local and pinned: never let transformers or the hub reach the network.
            ["HF_HUB_OFFLINE"] = "1",
            ["TRANSFORMERS_OFFLINE"] = "1",
            ["HF_HUB_DISABLE_TELEMETRY"] = "1",
            ["HF_HOME"] = Path.Combine(paths.Runtime, "hf-home"),
        };
        if (apiKey != null)
            env["LITTLERUN_API_KEY"] = apiKey;
        return env;
    }

    internal static string Describe(RuntimeStamp s) => JsonSerializer.Serialize(s, JsonFile.Options);
}
