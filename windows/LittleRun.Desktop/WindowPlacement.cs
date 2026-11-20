using Microsoft.Win32;

namespace LittleRun.Desktop;

/// <summary>Saved window position and size.</summary>
public sealed class WindowPlacement
{
    public int X { get; set; }
    public int Y { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public bool Maximized { get; set; }

    /// <summary>
    /// The saved bounds if at least a usable part of them is on one of <paramref name="screens"/>, else null
    /// (a monitor was unplugged or the resolution changed). Size is clamped to the minimum.
    /// </summary>
    public static Rectangle? Restore(WindowPlacement? p, IReadOnlyList<Rectangle> screens, Size min)
    {
        if (p == null || p.Width <= 0 || p.Height <= 0)
            return null;
        var r = new Rectangle(p.X, p.Y, Math.Max(p.Width, min.Width), Math.Max(p.Height, min.Height));
        // The top strip (where the title bar is) must be reachable, or the window cannot be moved back.
        var grip = new Rectangle(r.X + 40, r.Y, Math.Max(r.Width - 80, 1), 32);
        return screens.Any(s => s.IntersectsWith(grip)) ? r : null;
    }
}

/// <summary>Start LittleRun in the tray when the user signs in (per-user Run key, no admin rights).</summary>
public static class StartupRegistration
{
    private const string RunKey = @"Software\Microsoft\Windows\CurrentVersion\Run";
    private const string Name = "LittleRun";

    public static string Command(string exe) => $"\"{exe}\" --minimized";

    public static bool IsEnabled()
    {
        using var k = Registry.CurrentUser.OpenSubKey(RunKey);
        return k?.GetValue(Name) is string;
    }

    public static void Set(bool enabled, string exe)
    {
        using var k = Registry.CurrentUser.CreateSubKey(RunKey);
        if (enabled)
            k.SetValue(Name, Command(exe));
        else if (k.GetValue(Name) != null)
            k.DeleteValue(Name);
    }
}
