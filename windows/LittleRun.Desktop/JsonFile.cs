using System.Text.Json;
using System.Text.Json.Serialization;

namespace LittleRun.Desktop;

/// <summary>Atomic JSON read/write: a crash mid-write never leaves a half-written file.</summary>
public static class JsonFile
{
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public static T? Read<T>(string path) where T : class
    {
        if (!File.Exists(path))
            return null;
        try
        {
            using var s = File.OpenRead(path);
            return JsonSerializer.Deserialize<T>(s, Options);
        }
        catch (JsonException)
        {
            // Keep the unreadable file for inspection rather than silently losing it.
            var bad = path + ".corrupt-" + DateTime.Now.ToString("yyyyMMdd-HHmmss");
            File.Move(path, bad, overwrite: true);
            return null;
        }
    }

    public static void Write<T>(string path, T value)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var tmp = path + ".tmp";
        using (var s = File.Create(tmp))
        {
            JsonSerializer.Serialize(s, value, Options);
        }
        File.Move(tmp, path, overwrite: true);
    }
}
