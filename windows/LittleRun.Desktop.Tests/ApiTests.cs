using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using LittleRun.Desktop.Backend;
using Xunit;

namespace LittleRun.Desktop.Tests;

public class SseTests
{
    private static async Task<List<string>> Read(string raw)
    {
        var list = new List<string>();
        await foreach (var d in Sse.ReadAsync(new MemoryStream(Encoding.UTF8.GetBytes(raw)), CancellationToken.None))
            list.Add(d);
        return list;
    }

    [Fact]
    public async Task Splits_events_and_skips_comments()
    {
        var got = await Read(": keep-alive\n\ndata: {\"a\":1}\n\ndata: [DONE]\n\n");
        Assert.Equal(["{\"a\":1}", "[DONE]"], got);
    }

    [Fact]
    public async Task Joins_multi_line_data_and_handles_missing_final_blank_line()
    {
        var got = await Read("data: one\ndata: two\n\ndata:last");
        Assert.Equal(["one\ntwo", "last"], got);
    }

    [Fact]
    public async Task Handles_crlf_line_endings()
    {
        Assert.Equal(["x"], await Read("data: x\r\n\r\n"));
    }
}

public class ApiClientTests
{
    private static readonly ApiClient.Options Opts = new(64, 0.0, 1.0, 0, 1.0, Thinking: false);

    [Fact]
    public async Task Chat_streams_deltas_and_reads_usage()
    {
        var h = new FakeHandler(_ => Task.FromResult(FakeHandler.Sse(
            """{"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}""",
            """{"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}""",
            """{"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}""",
            """{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""",
            """{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":3}}}""",
            "[DONE]")));
        var api = new ApiClient(new HttpClient(h), "http://127.0.0.1:1", "key-123", "test-model");
        var text = new StringBuilder();
        var r = await api.ChatAsync([("user", "hi")], Opts, t => text.Append(t), CancellationToken.None);

        Assert.Equal("Hello", text.ToString());
        Assert.Equal("stop", r.FinishReason);
        Assert.Equal((5, 2, 3), (r.PromptTokens, r.CompletionTokens, r.CachedTokens));
        var req = h.Requests.Single();
        Assert.Equal("http://127.0.0.1:1/v1/chat/completions", req.RequestUri!.ToString());
        Assert.Equal("Bearer key-123", req.Headers.Authorization!.ToString());
        var body = JsonNode.Parse(h.Bodies.Single())!;
        Assert.Equal("test-model", body["model"]!.GetValue<string>());
        Assert.True(body["stream"]!.GetValue<bool>());
        Assert.False(body["chat_template_kwargs"]!["enable_thinking"]!.GetValue<bool>());
        Assert.Equal("hi", body["messages"]![0]!["content"]!.GetValue<string>());
    }

    [Fact]
    public async Task Completion_uses_text_field()
    {
        var h = new FakeHandler(_ => Task.FromResult(FakeHandler.Sse(
            """{"choices":[{"index":0,"text":" Paris","finish_reason":null}]}""",
            """{"choices":[{"index":0,"text":"","finish_reason":"length"}]}""",
            "[DONE]")));
        var api = new ApiClient(new HttpClient(h), "http://x", "k", "m");
        var text = new StringBuilder();
        var r = await api.CompleteAsync("The capital of France is", Opts, t => text.Append(t), CancellationToken.None);
        Assert.Equal(" Paris", text.ToString());
        Assert.Equal("length", r.FinishReason);
        Assert.EndsWith("/v1/completions", h.Requests.Single().RequestUri!.AbsolutePath);
        Assert.Equal("The capital of France is", JsonNode.Parse(h.Bodies.Single())!["prompt"]!.GetValue<string>());
    }

    [Fact]
    public async Task Error_event_in_stream_throws()
    {
        var h = new FakeHandler(_ => Task.FromResult(FakeHandler.Sse(
            """{"error":{"message":"out of memory","type":"server_error"}}""", "[DONE]")));
        var api = new ApiClient(new HttpClient(h), "http://x", "k", "m");
        var e = await Assert.ThrowsAsync<ApiException>(() => api.CompleteAsync("x", Opts, _ => { }, CancellationToken.None));
        Assert.Equal("out of memory", e.Message);
    }

    [Fact]
    public async Task Http_error_uses_server_message()
    {
        var h = new FakeHandler(_ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.ServiceUnavailable)
        {
            Content = new StringContent("""{"error":{"message":"model not ready (state=loading)"}}"""),
        }));
        var api = new ApiClient(new HttpClient(h), "http://x", "k", "m");
        var e = await Assert.ThrowsAsync<ApiException>(() => api.ChatAsync([("user", "x")], Opts, _ => { }, CancellationToken.None));
        Assert.Equal("model not ready (state=loading)", e.Message);
        Assert.Equal("HTTP 502", ApiClient.ErrorMessage("<html>bad gateway</html>", 502));
    }
}
