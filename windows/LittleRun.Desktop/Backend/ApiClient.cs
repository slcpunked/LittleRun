using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace LittleRun.Desktop.Backend;

/// <summary>Result of one streamed generation.</summary>
public sealed record GenerationResult(string FinishReason, int PromptTokens, int CompletionTokens, int CachedTokens,
                                      double Seconds, double? FirstTokenSeconds);

/// <summary>Minimal Server-Sent Events reader for the OpenAI-style stream: yields each `data:` payload.</summary>
public static class Sse
{
    public static async IAsyncEnumerable<string> ReadAsync(Stream body,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        using var reader = new StreamReader(body, Encoding.UTF8);
        var data = new StringBuilder();
        while (true)
        {
            var line = await reader.ReadLineAsync(ct);
            if (line == null)
                break;
            if (line.Length == 0)
            {
                if (data.Length > 0)
                {
                    yield return data.ToString();
                    data.Clear();
                }
                continue;
            }
            if (line.StartsWith(':'))
                continue; // comment / keep-alive
            if (line.StartsWith("data:", StringComparison.Ordinal))
            {
                if (data.Length > 0) data.Append('\n');
                data.Append(line.AsSpan(line.Length > 5 && line[5] == ' ' ? 6 : 5));
            }
        }
        if (data.Length > 0)
            yield return data.ToString();
    }
}

public sealed class ApiException(string message) : Exception(message);

/// <summary>Talks to the local LittleRun server.</summary>
public sealed class ApiClient(HttpClient http, string baseUrl, string apiKey, string modelId)
{
    public sealed record Options(int MaxTokens, double Temperature, double TopP, int TopK, double RepetitionPenalty,
                                 bool Thinking, IReadOnlyList<string>? Stop = null);

    public Task<GenerationResult> ChatAsync(IEnumerable<(string role, string content)> messages, Options o,
                                            Action<string> onDelta, CancellationToken ct)
    {
        var body = Common(o);
        body["messages"] = new JsonArray(messages.Select(m => (JsonNode)new JsonObject
        {
            ["role"] = m.role,
            ["content"] = m.content,
        }).ToArray());
        body["chat_template_kwargs"] = new JsonObject { ["enable_thinking"] = o.Thinking };
        return StreamAsync("/v1/chat/completions", body, chat: true, onDelta, ct);
    }

    public Task<GenerationResult> CompleteAsync(string prompt, Options o, Action<string> onDelta, CancellationToken ct)
    {
        var body = Common(o);
        body["prompt"] = prompt;
        return StreamAsync("/v1/completions", body, chat: false, onDelta, ct);
    }

    private JsonObject Common(Options o)
    {
        var body = new JsonObject
        {
            ["model"] = modelId,
            ["stream"] = true,
            ["stream_options"] = new JsonObject { ["include_usage"] = true },
            ["max_tokens"] = o.MaxTokens,
            ["temperature"] = o.Temperature,
            ["top_p"] = o.TopP,
            ["top_k"] = o.TopK,
            ["repetition_penalty"] = o.RepetitionPenalty,
        };
        if (o.Stop is { Count: > 0 })
            body["stop"] = new JsonArray(o.Stop.Select(s => (JsonNode)JsonValue.Create(s)!).ToArray());
        return body;
    }

    private async Task<GenerationResult> StreamAsync(string path, JsonObject body, bool chat, Action<string> onDelta,
                                                     CancellationToken ct)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var req = new HttpRequestMessage(HttpMethod.Post, baseUrl + path)
        {
            Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"),
        };
        req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
        req.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));
        using var resp = await http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
        if (!resp.IsSuccessStatusCode)
            throw new ApiException(ErrorMessage(await resp.Content.ReadAsStringAsync(ct), (int)resp.StatusCode));

        string finish = "stop";
        int prompt = 0, completion = 0, cached = 0;
        double? first = null;
        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        await foreach (var data in Sse.ReadAsync(stream, ct))
        {
            if (data == "[DONE]")
                break;
            var node = JsonNode.Parse(data);
            if (node?["error"] is JsonNode err)
                throw new ApiException(err["message"]?.GetValue<string>() ?? "generation failed");
            if (node?["usage"] is JsonObject u)
            {
                prompt = u["prompt_tokens"]?.GetValue<int>() ?? 0;
                completion = u["completion_tokens"]?.GetValue<int>() ?? 0;
                cached = u["prompt_tokens_details"]?["cached_tokens"]?.GetValue<int>() ?? 0;
            }
            if (node?["choices"] is not JsonArray choices || choices.Count == 0)
                continue;
            var c = choices[0]!;
            var text = chat ? c["delta"]?["content"]?.GetValue<string>() : c["text"]?.GetValue<string>();
            if (!string.IsNullOrEmpty(text))
            {
                first ??= sw.Elapsed.TotalSeconds;
                onDelta(text);
            }
            if (c["finish_reason"] is JsonValue f && f.GetValueKind() == JsonValueKind.String)
                finish = f.GetValue<string>();
        }
        return new GenerationResult(finish, prompt, completion, cached, sw.Elapsed.TotalSeconds, first);
    }

    internal static string ErrorMessage(string body, int status)
    {
        try
        {
            var msg = JsonNode.Parse(body)?["error"]?["message"]?.GetValue<string>();
            if (!string.IsNullOrEmpty(msg))
                return msg;
        }
        catch (JsonException) { }
        return $"HTTP {status}";
    }
}
