using System.Diagnostics;
using System.Text;
using LittleRun.Desktop.Backend;
using LittleRun.Desktop.Setup;

namespace LittleRun.Desktop;

/// <summary>
/// Checks the build without a window.
///   --selftest                 payload, uv, wheel, kernel, UI files and WebView2 (no network, no install)
///   --e2e [--model-dir DIR]    full setup into LITTLERUN_HOME, start the model, run a completion and a chat
///   --report FILE              also write the output to FILE
/// </summary>
internal static class Headless
{
    public static int Run(string[] args)
    {
        var report = Arg(args, "--report");
        var log = new StringBuilder();
        void Say(string s)
        {
            Console.WriteLine(s);
            log.AppendLine(s);
            if (report != null)
                File.WriteAllText(report, log.ToString());
        }

        int code;
        try
        {
            code = args.Contains("--e2e") ? E2E(args, Say).GetAwaiter().GetResult() : SelfTest(Say);
        }
        catch (Exception e)
        {
            Say($"FAIL unexpected: {e}");
            code = 1;
        }
        Say(code == 0 ? "RESULT PASS" : "RESULT FAIL");
        return code;
    }

    private static string? Arg(string[] args, string name)
    {
        var i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }

    private static int SelfTest(Action<string> say)
    {
        var fails = 0;
        void Check(string name, bool ok, string detail = "")
        {
            say($"{(ok ? "ok  " : "FAIL")} {name}{(detail.Length > 0 ? ": " + detail : "")}");
            if (!ok) fails++;
        }

        var asm = typeof(Headless).Assembly;
        Check("version", true, Application.ProductVersion.Split('+')[0]);
        foreach (var f in new[] { "index.html", "app.js", "app.css", "lib.js" })
            Check($"ui/{f}", asm.GetManifestResourceInfo("ui/" + f) != null);

        var home = Path.Combine(Path.GetTempPath(), "littlerun-selftest-" + Environment.ProcessId);
        try
        {
            var paths = new AppPaths(home);
            paths.EnsureBase();
            Check("payload embedded", Payload.IsEmbedded());
            var payload = Payload.Unpack(paths);
            var info = payload.Info;
            Check("payload unpacked", File.Exists(Path.Combine(payload.Dir, ".complete")), payload.Hash);
            Check("pins", info.Torch.Length > 0 && info.Python.Length > 0 && info.TorchIndexes.ContainsKey("cpu") &&
                          info.TorchIndexes.ContainsKey("cuda"),
                  $"python {info.Python}, torch {info.Torch}, uv {info.Uv}");
            Check("backend wheel", File.Exists(payload.WheelPath), info.Wheel);
            Check("requirements", File.Exists(payload.RequirementsPath) &&
                                  !File.ReadAllText(payload.RequirementsPath).Contains("\ntorch=="));
            foreach (var k in info.Kernel)
                Check($"kernel {Path.GetFileName(k)}", File.Exists(Path.Combine(payload.Dir, k)));
            Check("kernel included", info.Kernel.Count > 0);
            var spec = info.Models[info.DefaultModel];
            Check($"model spec {info.DefaultModel}", spec.Files.Count > 0 && spec.Sha256.Count > 0 &&
                                                     File.Exists(payload.ManifestPath(spec)) && spec.SizeBytes > 0);
            var uv = Process.Start(new ProcessStartInfo(payload.UvExe, "--version")
            {
                RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true,
            })!;
            var uvOut = uv.StandardOutput.ReadToEnd().Trim();
            uv.WaitForExit();
            Check("uv runs", uv.ExitCode == 0 && uvOut.Contains(info.Uv), uvOut);
            var wv = SystemProbe.WebView2();
            Check("WebView2 runtime", wv != null, wv ?? "missing");
        }
        finally
        {
            try { Directory.Delete(home, true); } catch (IOException) { }
        }
        return fails == 0 ? 0 : 1;
    }

    private static async Task<int> E2E(string[] args, Action<string> say)
    {
        var paths = AppPaths.Default();
        say($"home {paths.Root}");
        using var app = new AppController(paths);
        var device = Arg(args, "--device") ?? "cpu";
        var model = app.Payload.Info.DefaultModel;
        var sw = Stopwatch.StartNew();
        app.Event += (name, data) =>
        {
            if (name == "setup" && data is SetupEvent e && e.State is "done" or "failed" or "running" && e.Bytes == null)
                say($"[{sw.Elapsed:mm\\:ss}] setup {e.Step} {e.State}: {e.Message}");
            else if (name == "backend" && data is BackendStatus b)
                say($"[{sw.Elapsed:mm\\:ss}] backend {b.State}: {b.Message}");
        };
        await app.RunSetupAsync(new SetupRequest(device, model, Arg(args, "--model-dir"), AcceptLicense: true));
        say($"setup finished in {sw.Elapsed.TotalSeconds:0}s; set up = {app.IsSetUp}");

        var deadline = DateTime.UtcNow.AddMinutes(10);
        while (app.Backend.Status.State is not ("ready" or "error") && DateTime.UtcNow < deadline)
            await Task.Delay(500);
        if (app.Backend.Status.State != "ready")
        {
            say($"FAIL backend did not become ready: {app.Backend.Status.Message}");
            return 1;
        }
        say($"backend ready on port {app.Backend.Status.Port} after {sw.Elapsed.TotalSeconds:0}s");

        var text = new StringBuilder();
        app.Event += (name, data) =>
        {
            if (name == "gen.delta")
                text.Append(data.GetType().GetProperty("text")!.GetValue(data));
        };
        var saved = app.Settings.Temperature;
        app.Settings.Temperature = 0; // greedy for a stable check
        app.Settings.MaxTokens = 12;
        var r = await app.GenerateAsync(new GenRequest("e2e-complete", "complete", [], "The capital of France is", null));
        say($"completion: \"{text}\" ({r.FinishReason}, {r.CompletionTokens} tokens, {r.Seconds:0.0}s)");
        var completionOk = text.ToString().Contains("Paris", StringComparison.Ordinal);

        text.Clear();
        app.Settings.MaxTokens = 16;
        var c = await app.GenerateAsync(new GenRequest("e2e-chat", "chat", [("user", "Hello!")], null, false));
        say($"chat: {c.CompletionTokens} tokens, finish={c.FinishReason}, {c.CompletionTokens / Math.Max(c.Seconds, 1e-3):0.0} tok/s overall");
        var chatOk = c.CompletionTokens > 0;
        app.Settings.Temperature = saved;

        // Cancelling mid-stream must free the model for the next request.
        var stopper = Task.Run(async () => { await Task.Delay(1500); app.StopGeneration("e2e-stop"); });
        app.Settings.MaxTokens = 200;
        var stoppedOk = false;
        try
        {
            await app.GenerateAsync(new GenRequest("e2e-stop", "complete", [], "Once upon a time", null));
        }
        catch (OperationCanceledException)
        {
            stoppedOk = true;
        }
        await stopper;
        say($"stop mid-stream: {(stoppedOk ? "cancelled" : "not cancelled")}");
        app.Settings.MaxTokens = 4;
        var after = await app.GenerateAsync(new GenRequest("e2e-after", "complete", [], "Hello", null));
        say($"request after stop: {after.CompletionTokens} tokens");

        app.Backend.Stop();
        var ok = completionOk && chatOk && stoppedOk && after.CompletionTokens > 0;
        say(ok ? "e2e checks passed" : $"FAIL completionOk={completionOk} chatOk={chatOk} stoppedOk={stoppedOk}");
        return ok ? 0 : 1;
    }
}
