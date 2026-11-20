using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using LittleRun.Desktop.Setup;
using Xunit;

namespace LittleRun.Desktop.Tests;

public class PayloadTests
{
    [Fact]
    public void Reads_a_staged_payload_folder()
    {
        using var home = new TempHome();
        var p = Payload.FromDirectory(TestPayload.Create(home.Dir));
        Assert.Equal("2.15.0", p.Info.Torch);
        Assert.Equal("test-model", p.Info.Models["m"].ModelId);
        Assert.Equal("https://huggingface.co/org/m/resolve/abc123/config.json", p.Info.Models["m"].FileUrl("config.json"));
        Assert.EndsWith("littlerun-0.2.0-py3-none-any.whl", p.WheelPath);
    }

    [Fact]
    public void Rejects_a_payload_without_its_default_model()
    {
        using var home = new TempHome();
        var dir = TestPayload.Create(home.Dir);
        var f = Path.Combine(dir, "payload.json");
        File.WriteAllText(f, File.ReadAllText(f).Replace("\"defaultModel\": \"m\"", "\"defaultModel\": \"nope\""));
        Assert.Throws<InvalidOperationException>(() => Payload.FromDirectory(dir));
    }

    [Fact]
    public void Test_assembly_has_no_payload()
    {
        Assert.False(Payload.IsEmbedded(typeof(PayloadTests).Assembly));
    }
}

public class RuntimeInstallerTests
{
    [Theory]
    [InlineData("cpu", "https://download.pytorch.org/whl/cpu")]
    [InlineData("cuda", "https://download.pytorch.org/whl/cu132")]
    public void Plan_installs_pinned_torch_from_the_device_index(string device, string index)
    {
        using var home = new TempHome();
        var payload = Payload.FromDirectory(TestPayload.Create(home.Dir));
        var plan = new RuntimeInstaller(home.Paths, payload).Plan(device);

        var venv = plan[0].args;
        Assert.Equal(["venv", home.Paths.Venv, "--python", "3.12.16"], venv.Take(4));
        var torch = plan.Single(p => p.step == "torch").args;
        Assert.Contains(index, torch);
        Assert.Contains("torch==2.15.0", torch);
        var wheel = plan.Last().args;
        Assert.Contains("--no-deps", wheel);
        Assert.Equal(payload.WheelPath, wheel.Last());
    }

    [Fact]
    public void Uv_keeps_python_and_caches_inside_the_runtime_folder()
    {
        using var home = new TempHome();
        var env = new RuntimeInstaller(home.Paths, Payload.FromDirectory(TestPayload.Create(home.Dir))).UvEnv();
        Assert.StartsWith(home.Paths.Runtime, env["UV_PYTHON_INSTALL_DIR"]);
        Assert.StartsWith(home.Paths.Runtime, env["UV_CACHE_DIR"]);
        Assert.Equal("only-managed", env["UV_PYTHON_PREFERENCE"]);
    }

    [Fact]
    public void Runtime_is_current_only_for_the_same_payload_and_device()
    {
        using var home = new TempHome();
        var payload = Payload.FromDirectory(TestPayload.Create(home.Dir), "hash1");
        var ri = new RuntimeInstaller(home.Paths, payload);
        Assert.False(ri.IsCurrent("cpu"));
        Directory.CreateDirectory(Path.GetDirectoryName(home.Paths.VenvPython)!);
        File.WriteAllText(home.Paths.VenvPython, "");
        JsonFile.Write(home.Paths.RuntimeStamp, new RuntimeStamp { PayloadHash = "hash1", Device = "cpu" });
        Assert.True(ri.IsCurrent("cpu"));
        Assert.False(ri.IsCurrent("cuda"));
        Assert.False(new RuntimeInstaller(home.Paths, Payload.FromDirectory(payload.Dir, "hash2")).IsCurrent("cpu"));
    }

    [Fact]
    public void Backend_never_reaches_the_hub()
    {
        using var home = new TempHome();
        var env = RuntimeInstaller.BackendEnv(home.Paths, "k");
        Assert.Equal("1", env["HF_HUB_OFFLINE"]);
        Assert.Equal("1", env["TRANSFORMERS_OFFLINE"]);
        Assert.Equal(home.Paths.Kernel, env["LITTLERUN_KERNEL_DIR"]);
        Assert.Equal("k", env["LITTLERUN_API_KEY"]);
    }
}

public class ModelInstallerTests
{
    private static readonly byte[] Data = Enumerable.Range(0, 300_000).Select(i => (byte)(i * 7)).ToArray();

    /// <summary>Serves Data, honouring Range requests unless told not to.</summary>
    private static FakeHandler Server(bool honourRange = true) => new(req =>
    {
        long from = 0;
        if (honourRange && req.Headers.Range?.Ranges.FirstOrDefault()?.From is long f)
            from = f;
        var resp = new HttpResponseMessage(from > 0 ? HttpStatusCode.PartialContent : HttpStatusCode.OK)
        {
            Content = new ByteArrayContent(Data[(int)from..]),
        };
        if (from > 0)
            resp.Content.Headers.ContentRange = new ContentRangeHeaderValue(from, Data.Length - 1, Data.Length);
        return Task.FromResult(resp);
    });

    private static ModelInstaller Installer(TempHome home, FakeHandler h) =>
        new(home.Paths, Payload.FromDirectory(TestPayload.Create(home.Dir)), new HttpClient(h));

    [Fact]
    public async Task Resumes_a_partial_download_with_a_range_request()
    {
        using var home = new TempHome();
        var h = Server();
        var dest = Path.Combine(home.Dir, "f.bin");
        File.WriteAllBytes(dest + ".part", Data[..100_000]);
        long reported = 0;
        await Installer(home, h).DownloadFileAsync("http://x/f", dest, n => reported += n, CancellationToken.None);
        Assert.Equal(Data, File.ReadAllBytes(dest));
        Assert.False(File.Exists(dest + ".part"));
        Assert.Equal(100_000, h.Requests.Single().Headers.Range!.Ranges.Single().From);
        Assert.Equal(200_000, reported);
    }

    [Fact]
    public async Task Starts_over_when_the_server_ignores_the_range()
    {
        using var home = new TempHome();
        var dest = Path.Combine(home.Dir, "f.bin");
        File.WriteAllBytes(dest + ".part", new byte[50_000]); // wrong bytes; must not be kept
        long reported = 0;
        await Installer(home, Server(honourRange: false)).DownloadFileAsync("http://x/f", dest, n => reported += n,
                                                                              CancellationToken.None);
        Assert.Equal(Data, File.ReadAllBytes(dest));
        Assert.Equal(Data.Length - 50_000, reported); // -50_000 correction, then the full file
    }

    [Fact]
    public async Task A_file_with_the_wrong_hash_is_deleted_and_setup_fails()
    {
        using var home = new TempHome();
        var installer = Installer(home, Server());
        var dir = Path.Combine(home.Dir, "model");
        using var r = new SetupReporter(home.Paths.SetupLog, _ => { });
        var e = await Assert.ThrowsAsync<SetupFailedException>(() => installer.DownloadAsync("m", dir, r, CancellationToken.None));
        Assert.Equal("model", e.Step);
        Assert.Contains("sha256", e.Message);
        Assert.False(File.Exists(Path.Combine(dir, "model.safetensors")));
        Assert.True(File.Exists(Path.Combine(dir, "config.json"))); // unhashed files are kept for the resume
    }

    [Fact]
    public async Task Sha256_matches_the_framework()
    {
        using var home = new TempHome();
        var f = Path.Combine(home.Dir, "x");
        File.WriteAllBytes(f, Data);
        Assert.Equal(Convert.ToHexString(SHA256.HashData(Data)).ToLowerInvariant(),
                     await ModelInstaller.Sha256Async(f, CancellationToken.None));
    }

    [Fact]
    public void Missing_files_lists_what_is_absent()
    {
        using var home = new TempHome();
        var spec = Payload.FromDirectory(TestPayload.Create(home.Dir)).Info.Models["m"];
        File.WriteAllText(Path.Combine(home.Dir, "config.json"), "{}");
        Assert.Equal(["model.safetensors"], ModelInstaller.MissingFiles(spec, home.Dir));
    }
}
