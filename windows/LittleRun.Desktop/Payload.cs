using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;

namespace LittleRun.Desktop;

/// <summary>payload.json, written by build.ps1.</summary>
public sealed class PayloadInfo
{
    public string Version { get; set; } = "";
    public string Uv { get; set; } = "";
    public string Python { get; set; } = "";
    public string Torch { get; set; } = "";
    public Dictionary<string, string> TorchIndexes { get; set; } = new();
    public double CudaMinDriver { get; set; } = 580;
    public string Wheel { get; set; } = "";
    public string Requirements { get; set; } = "requirements.txt";
    public List<string> Kernel { get; set; } = new();
    public string DefaultModel { get; set; } = "";
    public Dictionary<string, ModelSpec> Models { get; set; } = new();
}

/// <summary>A pinned checkpoint the app knows how to download and run.</summary>
public sealed class ModelSpec
{
    public string DisplayName { get; set; } = "";
    public string ModelId { get; set; } = "";
    public string Repo { get; set; } = "";
    public string Revision { get; set; } = "";
    public List<string> Files { get; set; } = new();
    public Dictionary<string, string> Sha256 { get; set; } = new();
    public string License { get; set; } = "";
    public string LicenseUrl { get; set; } = "";
    public string Manifest { get; set; } = "";
    public long SizeBytes { get; set; }
    public bool ChatTuned { get; set; }
    public string Note { get; set; } = "";

    public string FileUrl(string file) => $"https://huggingface.co/{Repo}/resolve/{Revision}/{file}";
}

/// <summary>The backend payload embedded in the exe, unpacked to runtime\payload\&lt;hash&gt;.</summary>
public sealed class Payload
{
    private const string Prefix = "payload/";

    private Payload(string dir, string hash, PayloadInfo info)
    {
        Dir = dir;
        Hash = hash;
        Info = info;
    }

    public string Dir { get; }
    public string Hash { get; }
    public PayloadInfo Info { get; }

    public string UvExe => Path.Combine(Dir, "uv.exe");
    public string WheelPath => Path.Combine(Dir, Info.Wheel);
    public string RequirementsPath => Path.Combine(Dir, Info.Requirements);
    public string ManifestPath(ModelSpec m) => Path.Combine(Dir, m.Manifest);

    public static bool IsEmbedded(Assembly? asm = null) =>
        Names(asm ?? typeof(Payload).Assembly).Any(n => Normalize(n) == Prefix + "payload.json");

    /// <summary>Unpack the embedded payload (once per content hash) and return it.</summary>
    public static Payload Unpack(AppPaths paths, Assembly? asm = null)
    {
        asm ??= typeof(Payload).Assembly;
        var names = Names(asm).Where(n => Normalize(n).StartsWith(Prefix, StringComparison.Ordinal))
                              .OrderBy(n => Normalize(n), StringComparer.Ordinal).ToList();
        if (!names.Any(n => Normalize(n) == Prefix + "payload.json"))
            throw new InvalidOperationException("This build has no backend payload. Build it with build.bat.");

        var hash = ContentHash(asm, names);
        var dir = Path.Combine(paths.PayloadCache, hash);
        if (!File.Exists(Path.Combine(dir, ".complete")))
        {
            var tmp = dir + ".tmp-" + Environment.ProcessId;
            if (Directory.Exists(tmp))
                Directory.Delete(tmp, true);
            foreach (var n in names)
            {
                var rel = Normalize(n)[Prefix.Length..].Replace('/', Path.DirectorySeparatorChar);
                var dest = Path.Combine(tmp, rel);
                Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
                using var src = asm.GetManifestResourceStream(n)!;
                using var f = File.Create(dest);
                src.CopyTo(f);
            }
            File.WriteAllText(Path.Combine(tmp, ".complete"), hash);
            if (Directory.Exists(dir))
                Directory.Delete(dir, true);
            Directory.Move(tmp, dir);
        }
        CleanOld(paths.PayloadCache, hash);
        return FromDirectory(dir, hash);
    }

    /// <summary>Use an already unpacked payload folder (tests, development).</summary>
    public static Payload FromDirectory(string dir, string? hash = null)
    {
        var info = JsonSerializer.Deserialize<PayloadInfo>(File.ReadAllText(Path.Combine(dir, "payload.json")),
                                                           JsonFile.Options)
                   ?? throw new InvalidOperationException("payload.json is empty");
        if (info.Models.Count == 0 || !info.Models.ContainsKey(info.DefaultModel))
            throw new InvalidOperationException("payload.json names no usable default model");
        return new Payload(dir, hash ?? "dev", info);
    }

    private static IEnumerable<string> Names(Assembly asm) => asm.GetManifestResourceNames();

    internal static string Normalize(string name) => name.Replace('\\', '/');

    private static string ContentHash(Assembly asm, List<string> names)
    {
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var buf = new byte[1 << 16];
        foreach (var n in names)
        {
            sha.AppendData(System.Text.Encoding.UTF8.GetBytes(Normalize(n) + "\0"));
            using var s = asm.GetManifestResourceStream(n)!;
            int r;
            while ((r = s.Read(buf, 0, buf.Length)) > 0)
                sha.AppendData(buf, 0, r);
        }
        return Convert.ToHexString(sha.GetHashAndReset())[..16].ToLowerInvariant();
    }

    private static void CleanOld(string cache, string keep)
    {
        try
        {
            foreach (var d in Directory.EnumerateDirectories(cache))
                if (!string.Equals(Path.GetFileName(d), keep, StringComparison.OrdinalIgnoreCase))
                    Directory.Delete(d, true);
        }
        catch (IOException) { /* in use by another instance; try next time */ }
        catch (UnauthorizedAccessException) { }
    }
}
