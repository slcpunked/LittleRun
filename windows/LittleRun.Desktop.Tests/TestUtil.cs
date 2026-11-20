using System.Net;
using System.Text;
using LittleRun.Desktop;

namespace LittleRun.Desktop.Tests;

/// <summary>A temporary LittleRun home folder, deleted afterwards.</summary>
public sealed class TempHome : IDisposable
{
    public TempHome()
    {
        Dir = Path.Combine(Path.GetTempPath(), "lr-test-" + Guid.NewGuid().ToString("N")[..10]);
        Directory.CreateDirectory(Dir);
        Paths = new AppPaths(Dir);
        Paths.EnsureBase();
    }

    public string Dir { get; }
    public AppPaths Paths { get; }

    public void Dispose()
    {
        try { Directory.Delete(Dir, true); } catch (IOException) { }
    }
}

/// <summary>HttpMessageHandler that answers from a function.</summary>
public sealed class FakeHandler(Func<HttpRequestMessage, Task<HttpResponseMessage>> respond) : HttpMessageHandler
{
    public List<HttpRequestMessage> Requests { get; } = new();
    public List<string> Bodies { get; } = new();

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
    {
        Requests.Add(request);
        Bodies.Add(request.Content == null ? "" : await request.Content.ReadAsStringAsync(ct));
        return await respond(request);
    }

    public static HttpResponseMessage Sse(params string[] events)
    {
        var body = string.Concat(events.Select(e => $"data: {e}\n\n"));
        return new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(body, Encoding.UTF8, "text/event-stream"),
        };
    }
}

public static class TestPayload
{
    /// <summary>Write a minimal payload folder like build.ps1 stages.</summary>
    public static string Create(string root)
    {
        var dir = Path.Combine(root, "payload-src");
        Directory.CreateDirectory(Path.Combine(dir, "manifests"));
        File.WriteAllText(Path.Combine(dir, "manifests", "m.manifest.json"), "{}");
        File.WriteAllText(Path.Combine(dir, "requirements.txt"), "safetensors==0.5.0\n");
        File.WriteAllText(Path.Combine(dir, "payload.json"), """
        {
          "version": "0.2.0", "uv": "0.13.9", "python": "3.12.16", "torch": "2.15.0",
          "torchIndexes": { "cpu": "https://download.pytorch.org/whl/cpu", "cuda": "https://download.pytorch.org/whl/cu132" },
          "cudaMinDriver": 580, "wheel": "littlerun-0.2.0-py3-none-any.whl", "requirements": "requirements.txt",
          "kernel": ["kernel/littlerun_signed_gemm.pyd"], "defaultModel": "m",
          "models": { "m": { "displayName": "Test model", "modelId": "test-model", "repo": "org/m", "revision": "abc123",
            "files": ["config.json", "model.safetensors"], "sha256": { "model.safetensors": "00" },
            "license": "cc-by-nc-4.0", "manifest": "manifests/m.manifest.json", "sizeBytes": 1000, "chatTuned": false } }
        }
        """);
        return dir;
    }
}
