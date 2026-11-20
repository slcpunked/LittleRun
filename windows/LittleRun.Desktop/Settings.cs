using System.Security.Cryptography;

namespace LittleRun.Desktop;

/// <summary>User settings, stored in settings.json.</summary>
public sealed class Settings
{
    /// <summary>"cpu" or "cuda". Chosen during setup; decides which PyTorch build is installed.</summary>
    public string Device { get; set; } = "cpu";

    /// <summary>Folder of the installed model (inside models\ or an imported folder).</summary>
    public string? ModelDir { get; set; }

    /// <summary>The pinned model name the folder holds (key in payload.json "models").</summary>
    public string? ModelName { get; set; }

    public bool LicenseAccepted { get; set; }

    /// <summary>Port for the local API. 0 picks a free port each start.</summary>
    public int Port { get; set; } = 11435;

    /// <summary>Key the local API requires. Generated once; shown in Settings for other OpenAI-style clients.</summary>
    public string ApiKey { get; set; } = NewKey();

    public int MaxContext { get; set; } = 4096;
    public int MaxTokens { get; set; } = 512;
    public double Temperature { get; set; } = 0.7;
    public double TopP { get; set; } = 0.8;
    public int TopK { get; set; } = 20;
    public double RepetitionPenalty { get; set; } = 1.15;
    public bool Thinking { get; set; }
    public string SystemPrompt { get; set; } = "";
    public bool CloseToTray { get; set; } = true;
    public bool TrayHintShown { get; set; }

    /// <summary>"auto" (story chat for models not trained on chat), "story" or "template".</summary>
    public string ChatStyle { get; set; } = "auto";

    /// <summary>Story chat character: a preset id (ada, maya, leo) or "custom".</summary>
    public string Character { get; set; } = "ada";

    /// <summary>How story chat names the user ("Sam asked, ...").</summary>
    public string UserName { get; set; } = "Sam";

    public StoryCharacter CustomCharacter { get; set; } = NewCustom();

    public static StoryCharacter NewCustom() => new()
    {
        Id = "custom",
        Name = "Custom",
        Summary = "Your own character",
        Self = "",
        Scene = "Today {user} came over and we started talking.",
        Examples = [new("How are you?", "I'm doing well, thanks for asking.")],
    };

    /// <summary>Whether chat for <paramref name="m"/> uses the story format.</summary>
    public bool UseStory(ModelSpec m) => ChatStyle == "story" || (ChatStyle == "auto" && !m.ChatTuned);

    public static string NewKey() => "lr-" + Convert.ToHexString(RandomNumberGenerator.GetBytes(24)).ToLowerInvariant();

    public static Settings Load(AppPaths paths)
    {
        var s = JsonFile.Read<Settings>(paths.Settings) ?? new Settings();
        s.Normalize();
        return s;
    }

    public void Save(AppPaths paths) => JsonFile.Write(paths.Settings, this);

    /// <summary>Clamp values a hand-edited file could break.</summary>
    public void Normalize()
    {
        Device = Device is "cuda" ? "cuda" : "cpu";
        if (Port is < 0 or > 65535) Port = 11435;
        if (string.IsNullOrWhiteSpace(ApiKey) || ApiKey.Length < 16) ApiKey = NewKey();
        MaxContext = Math.Clamp(MaxContext, 512, 32768);
        MaxTokens = Math.Clamp(MaxTokens, 1, 2048);
        Temperature = Math.Clamp(Temperature, 0.0, 2.0);
        TopP = TopP is > 0 and <= 1 ? TopP : 0.8;
        TopK = Math.Max(0, TopK);
        RepetitionPenalty = RepetitionPenalty is > 0 and <= 2 ? RepetitionPenalty : 1.15;
        SystemPrompt ??= "";
        if (ChatStyle is not ("auto" or "story" or "template")) ChatStyle = "auto";
        if (Character != "custom" && StoryChat.Preset(Character) == null) Character = "ada";
        UserName = StoryChat.Quote(UserName ?? "");
        if (UserName.Length == 0) UserName = "Sam";
        if (UserName.Length > 40) UserName = UserName[..40];
        CustomCharacter ??= NewCustom();
        CustomCharacter.Id = "custom";
        CustomCharacter.Name = string.IsNullOrWhiteSpace(CustomCharacter.Name) ? "Custom" : CustomCharacter.Name.Trim();
        CustomCharacter.Self ??= "";
        CustomCharacter.Scene ??= "";
        CustomCharacter.Examples = (CustomCharacter.Examples ?? new())
            .Where(e => e != null && !string.IsNullOrWhiteSpace(e.Q) && !string.IsNullOrWhiteSpace(e.A)).Take(12).ToList();
        // A custom character without a description cannot carry a story; fall back to the default preset.
        if (Character == "custom" && string.IsNullOrWhiteSpace(CustomCharacter.Self)) Character = "ada";
    }
}
