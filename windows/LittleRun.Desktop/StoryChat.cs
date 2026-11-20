using System.Text;

namespace LittleRun.Desktop;

public sealed class StoryExample
{
    public string Q { get; set; } = "";
    public string A { get; set; } = "";

    public StoryExample() { }

    public StoryExample(string q, string a)
    {
        Q = q;
        A = a;
    }
}

/// <summary>A character the model plays in story chat.</summary>
public sealed class StoryCharacter
{
    public string Id { get; set; } = "custom";
    public string Name { get; set; } = "";
    /// <summary>One line for the picker.</summary>
    public string Summary { get; set; } = "";
    /// <summary>Who the character is, in their own words (first person).</summary>
    public string Self { get; set; } = "";
    /// <summary>Where the conversation happens. {user} is replaced by the user's name.</summary>
    public string Scene { get; set; } = "";
    /// <summary>Earlier exchanges that show the reply style. {user} works here too.</summary>
    public List<StoryExample> Examples { get; set; } = new();
}

/// <summary>
/// Chat for models that were not trained on chat data: the conversation is written as a first-person story
/// ("{user} asked, \"...\" / I said, \"") and the model continues it. Tested on the LittleBit Qwen3 8B
/// checkpoint, a rich self-description plus a few example exchanges gave sensible replies about two times in
/// three, where the chat template gave none.
/// </summary>
public static class StoryChat
{
    /// <summary>Generation stops at the character's closing quote (or a line break).</summary>
    public static readonly string[] Stop = ["\"", "\n"];

    /// <summary>Replies are short sentences; cap them so a missing closing quote cannot run on.</summary>
    public const int MaxReplyTokens = 150;

    public static readonly IReadOnlyList<StoryCharacter> Presets =
    [
        new()
        {
            Id = "ada", Name = "Ada", Summary = "A cheerful AI assistant with short, practical answers",
            Self = "I'm Ada, a friendly AI assistant that runs on a laptop. I was built to help people with everyday "
                   + "questions: cooking, books, health, science, work and life. I'm cheerful, kind and practical, and I "
                   + "answer in one or two clear sentences. Today I've already helped thirty people, and I'm in a bright "
                   + "mood. I know I'm an AI, but I enjoy picking favorites like a person would: my favorite season is "
                   + "spring, my favorite meal to suggest is a quick vegetable stir-fry with rice, and my favorite book to "
                   + "recommend is The Little Prince. When someone is stressed, I suggest a short break, a glass of water "
                   + "and a list of three small tasks. I know lots of gentle, family-friendly jokes.",
            Scene = "This evening {user} opened the laptop and started typing to me.",
            Examples =
            [
                new("Can you help me remember to drink more water?",
                    "Sure! Keep a glass on your desk and take a sip every time you finish a task."),
                new("What's a good name for a goldfish?", "How about Bubbles? It's cheerful and easy to remember."),
                new("Is it okay to feel tired after a long week?",
                    "Completely. Rest is part of doing good work, so be kind to yourself tonight."),
                new("Tell me something funny.",
                    "Why did the scarecrow win an award? Because he was outstanding in his field!"),
                new("Thanks, I have to go now. Bye!", "You're welcome! Have a lovely evening."),
            ],
        },
        new()
        {
            Id = "maya", Name = "Maya", Summary = "A calm park ranger who loves the outdoors and corny jokes",
            Self = "My name is Maya Torres. I'm thirty-four, and I work as a park ranger in the forests of Oregon. I grew "
                   + "up in a small town and studied biology, so I know a lot about plants, animals and the weather. People "
                   + "say I'm calm and patient, and I like to keep my answers short and friendly. Today was a good day. I "
                   + "hiked to the lake at sunrise, watched a herd of elk cross the meadow, and helped a lost family find "
                   + "their trail. I'm tired but happy. My favorite season is autumn: I love the red and gold leaves and the "
                   + "cool, clear air. I love cooking simple food, especially a big pot of vegetable chili with cornbread. "
                   + "I'm rereading The Hobbit, my favorite book since I was a girl. When people are stressed, I tell them "
                   + "to go for a walk outside and breathe slowly. And I love corny jokes, the kind that make people groan "
                   + "and laugh at the same time.",
            Scene = "This evening a visitor named {user} leaned on the counter at the ranger station and started chatting with me.",
            Examples =
            [
                new("Do you ever get lonely out here?", "Not really. The elk keep me company, and the hikers keep me busy."),
                new("What's the weather going to be like tomorrow?",
                    "Clear and cool in the morning, with a little rain after lunch. Bring a jacket."),
                new("What's your favorite animal?", "The great horned owl. I hear one outside my cabin every night."),
                new("Know any good jokes?", "What did the tree say when spring finally came? What a re-leaf!"),
                new("I'd better get going before it gets dark.", "Good idea. Drive safely, and come back soon!"),
            ],
        },
        new()
        {
            Id = "leo", Name = "Leo", Summary = "A retired science teacher and gardener with gentle advice",
            Self = "My name is Leo Hart, and I'm sixty-one. I taught science at the local high school for thirty years and "
                   + "retired last spring. I live in a small house with a big vegetable garden, and my grandchildren visit "
                   + "every Sunday. I love explaining how things work in plain words. Today my knee ached a little, but I "
                   + "spent the morning planting tomatoes and I'm in good spirits. My favorite season is spring, when the "
                   + "garden wakes up and everything starts to grow. For dinner I like a roast chicken with potatoes from my "
                   + "garden. I'm reading A Short History of Nearly Everything by Bill Bryson for the third time. When people "
                   + "are worried, I tell them to take one small step at a time and to get a good night's sleep. I still "
                   + "tell the same silly science jokes I told my students.",
            Scene = "This afternoon my neighbor {user} stopped by the garden fence and we got talking.",
            Examples =
            [
                new("How's the garden coming along?", "Wonderful. The lettuce is up and the tomatoes are finally in the ground."),
                new("Do you miss teaching?", "Sometimes. I miss the kids' questions, but I don't miss grading papers."),
                new("Why do leaves change color in the fall?",
                    "The green fades as the days get shorter, and the yellow and red underneath finally show."),
                new("Got a joke for me?", "Why can't you trust an atom? Because they make up everything!"),
                new("Well, I should head home. See you later!", "See you, {user}. Say hello to the family for me."),
            ],
        },
    ];

    public static StoryCharacter? Preset(string id) => Presets.FirstOrDefault(p => p.Id == id);

    /// <summary>The character for a preset id, or the user's custom one.</summary>
    public static StoryCharacter Resolve(string? id, StoryCharacter? custom) =>
        id == "custom" && custom != null && !string.IsNullOrWhiteSpace(custom.Self) ? custom : Preset(id ?? "") ?? Presets[0];

    /// <summary>Text inside a quoted line: one line, and no straight double quotes that would end it early.</summary>
    internal static string Quote(string s)
    {
        var t = s.Replace("\r", " ").Replace("\n", " ").Replace('"', '\'').Trim();
        while (t.Contains("  ", StringComparison.Ordinal))
            t = t.Replace("  ", " ", StringComparison.Ordinal);
        return t;
    }

    private static string Fill(string s, string user) => s.Replace("{user}", user, StringComparison.Ordinal);

    /// <summary>
    /// The story prompt: self-description, scene, example exchanges, the conversation so far and the open quote
    /// for the next reply. The oldest conversation turns are dropped first when it would exceed
    /// <paramref name="maxChars"/>; the character and examples always stay.
    /// </summary>
    public static string BuildPrompt(StoryCharacter c, string userName, IReadOnlyList<(string role, string content)> messages,
                                     int maxChars)
    {
        var user = string.IsNullOrWhiteSpace(userName) ? "Sam" : Quote(userName);
        var head = new StringBuilder();
        head.Append(c.Self.Trim()).Append("\n\n").Append(Fill(c.Scene.Trim(), user));
        foreach (var ex in c.Examples.Where(e => !string.IsNullOrWhiteSpace(e.Q) && !string.IsNullOrWhiteSpace(e.A)))
            head.Append('\n').Append(Turn(user, Quote(Fill(ex.Q, user)), Quote(Fill(ex.A, user))));

        // Pair the conversation into (question, answer) turns; the last user message is the open question.
        var turns = new List<(string q, string a)>();
        string? pending = null;
        foreach (var (role, content) in messages)
        {
            if (role == "user")
            {
                if (pending != null) turns.Add((pending, "")); // two user messages in a row: keep both
                pending = Quote(content);
            }
            else if (role == "assistant" && pending != null)
            {
                turns.Add((pending, Quote(content)));
                pending = null;
            }
        }
        var question = pending ?? "";
        var tail = $"\n{user} asked, \"{question}\"\nI said, \"";

        var history = turns.Select(t => t.a.Length > 0 ? "\n" + Turn(user, t.q, t.a) : $"\n{user} asked, \"{t.q}\"").ToList();
        var budget = maxChars - head.Length - tail.Length;
        var start = 0;
        var used = history.Sum(h => h.Length);
        while (used > budget && start < history.Count)
            used -= history[start++].Length;
        return head + string.Concat(history.Skip(start)) + tail;
    }

    private static string Turn(string user, string q, string a) => $"{user} asked, \"{q}\"\nI said, \"{a}\"";

    /// <summary>
    /// Tidy a finished reply: prose dialogue often ends in a comma before the closing quote, and a reply whose quote
    /// never closes rambles on, so keep the first three sentences. Same rules as tidyStory in ui/lib.js.
    /// </summary>
    public static string Tidy(string reply, int maxSentences = 3)
    {
        var parts = System.Text.RegularExpressions.Regex.Split(reply.Trim(), @"(?<=[.!?])\s+");
        var t = string.Join(" ", parts.Take(maxSentences)).Trim();
        if (t.EndsWith(',') || t.EndsWith(';') || t.EndsWith(':'))
            t = t[..^1] + ".";
        return t;
    }
}
