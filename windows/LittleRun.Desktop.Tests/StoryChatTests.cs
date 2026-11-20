using System.Text;
using System.Text.Json.Nodes;
using LittleRun.Desktop.Backend;
using Xunit;

namespace LittleRun.Desktop.Tests;

public class StoryChatTests
{
    private static readonly StoryCharacter Ada = StoryChat.Preset("ada")!;

    [Fact]
    public void Presets_are_complete()
    {
        Assert.Equal(["ada", "maya", "leo"], StoryChat.Presets.Select(p => p.Id));
        foreach (var p in StoryChat.Presets)
        {
            Assert.False(string.IsNullOrWhiteSpace(p.Self));
            Assert.Contains("{user}", p.Scene);
            Assert.Equal(5, p.Examples.Count);
            Assert.All(p.Examples, e => Assert.DoesNotContain("\"", e.Q + e.A));
        }
    }

    [Fact]
    public void Prompt_is_a_first_person_story_ending_in_an_open_quote()
    {
        var p = StoryChat.BuildPrompt(Ada, "Dana", [("user", "How are you?")], 20000);
        Assert.StartsWith("I'm Ada, a friendly AI assistant", p);
        Assert.Contains("This evening Dana opened the laptop", p);
        Assert.Contains("Dana asked, \"Tell me something funny.\"\nI said, \"Why did the scarecrow", p);
        Assert.EndsWith("\nDana asked, \"How are you?\"\nI said, \"", p);
    }

    [Fact]
    public void Conversation_turns_follow_the_examples_in_order()
    {
        var p = StoryChat.BuildPrompt(Ada, "Sam",
            [("user", "Hi!"), ("assistant", "Hello there."), ("user", "What's for dinner?")], 20000);
        var examples = p.IndexOf("Bye!", StringComparison.Ordinal);
        var hi = p.IndexOf("Sam asked, \"Hi!\"\nI said, \"Hello there.\"", StringComparison.Ordinal);
        Assert.True(examples > 0 && hi > examples);
        Assert.EndsWith("Sam asked, \"What's for dinner?\"\nI said, \"", p);
    }

    [Fact]
    public void User_text_cannot_break_the_quotes_or_lines()
    {
        var p = StoryChat.BuildPrompt(Ada, "Sam", [("user", "She said \"hi\"\nthen left")], 20000);
        Assert.EndsWith("Sam asked, \"She said 'hi' then left\"\nI said, \"", p);
    }

    [Fact]
    public void Oldest_turns_are_dropped_to_fit_but_the_character_stays()
    {
        var msgs = new List<(string, string)>();
        for (var i = 0; i < 40; i++)
        {
            msgs.Add(("user", $"Question number {i} with some padding text to take room"));
            msgs.Add(("assistant", $"Answer number {i} with some padding text to take room"));
        }
        msgs.Add(("user", "Last question"));
        var full = StoryChat.BuildPrompt(Ada, "Sam", msgs, 1_000_000);
        var cut = StoryChat.BuildPrompt(Ada, "Sam", msgs, 3500);
        Assert.True(cut.Length <= 3500, $"prompt is {cut.Length} chars");
        Assert.StartsWith("I'm Ada", cut);
        Assert.Contains("Bye!", cut);                      // examples kept
        Assert.DoesNotContain("Question number 0 ", cut);  // oldest dropped
        Assert.Contains("Question number 39 ", cut);       // newest kept
        Assert.EndsWith("Sam asked, \"Last question\"\nI said, \"", cut);
        Assert.Contains("Question number 0 ", full);
    }

    [Fact]
    public void Custom_character_is_used_only_when_it_has_a_description()
    {
        var custom = new StoryCharacter { Id = "custom", Name = "Robin", Self = "My name is Robin.", Scene = "{user} came in." };
        Assert.Same(custom, StoryChat.Resolve("custom", custom));
        Assert.Equal("ada", StoryChat.Resolve("custom", new StoryCharacter { Self = " " }).Id);
        Assert.Equal("leo", StoryChat.Resolve("leo", custom).Id);
        Assert.Equal("ada", StoryChat.Resolve("nobody", custom).Id);
    }

    [Fact]
    public void Tidy_turns_a_speech_comma_into_a_full_stop()
    {
        Assert.Equal("It's winter.", StoryChat.Tidy(" It's winter, "));
        Assert.Equal("Spring!", StoryChat.Tidy("Spring!"));
        Assert.Equal("One. Two! Three?", StoryChat.Tidy("One. Two! Three? Four. Five."));
        Assert.Equal("It's 3.5 miles. Easy walk.", StoryChat.Tidy("It's 3.5 miles. Easy walk."));
    }

    [Fact]
    public void Story_is_used_for_models_without_chat_training_unless_the_template_is_chosen()
    {
        var notChat = new ModelSpec { ChatTuned = false };
        var chat = new ModelSpec { ChatTuned = true };
        Assert.True(new Settings().UseStory(notChat));
        Assert.False(new Settings().UseStory(chat));
        Assert.False(new Settings { ChatStyle = "template" }.UseStory(notChat));
        Assert.True(new Settings { ChatStyle = "story" }.UseStory(chat));
    }

    [Fact]
    public void Settings_normalize_story_fields()
    {
        var s = new Settings
        {
            ChatStyle = "weird", Character = "custom", UserName = "  Dana \"D\"\n ",
            CustomCharacter = new StoryCharacter { Self = "", Examples = [new("q", ""), new("Hi?", "Hello.")] },
        };
        s.Normalize();
        Assert.Equal("auto", s.ChatStyle);
        Assert.Equal("ada", s.Character); // custom without a description falls back
        Assert.Equal("Dana 'D'", s.UserName);
        Assert.Single(s.CustomCharacter.Examples);
    }

    [Fact]
    public async Task Completion_requests_send_stop_sequences()
    {
        var h = new FakeHandler(_ => Task.FromResult(FakeHandler.Sse(
            """{"choices":[{"index":0,"text":"Fine, thanks.","finish_reason":"stop"}]}""", "[DONE]")));
        var api = new ApiClient(new HttpClient(h), "http://x", "k", "m");
        var opts = new ApiClient.Options(150, 0.7, 0.8, 20, 1.15, false, StoryChat.Stop);
        var text = new StringBuilder();
        await api.CompleteAsync("prompt", opts, t => text.Append(t), CancellationToken.None);
        var stop = JsonNode.Parse(h.Bodies.Single())!["stop"]!.AsArray().Select(n => n!.GetValue<string>());
        Assert.Equal(["\"", "\n"], stop);
        Assert.Equal("Fine, thanks.", text.ToString());
    }
}
