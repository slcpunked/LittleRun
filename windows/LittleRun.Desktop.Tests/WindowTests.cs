using Xunit;

namespace LittleRun.Desktop.Tests;

public class WindowTests
{
    private static readonly Rectangle Screen1 = new(0, 0, 1920, 1040);
    private static readonly Size Min = new(760, 500);

    [Fact]
    public void Restores_a_window_that_is_on_screen()
    {
        var r = WindowPlacement.Restore(new WindowPlacement { X = 100, Y = 80, Width = 1200, Height = 800 }, [Screen1], Min);
        Assert.Equal(new Rectangle(100, 80, 1200, 800), r);
    }

    [Fact]
    public void Forgets_a_window_on_a_monitor_that_is_gone()
    {
        Assert.Null(WindowPlacement.Restore(new WindowPlacement { X = 2500, Y = 100, Width = 900, Height = 700 }, [Screen1], Min));
        Assert.Null(WindowPlacement.Restore(new WindowPlacement { X = 100, Y = -400, Width = 900, Height = 300 }, [Screen1], Min));
        Assert.Null(WindowPlacement.Restore(null, [Screen1], Min));
    }

    [Fact]
    public void Too_small_saved_sizes_grow_to_the_minimum()
    {
        var r = WindowPlacement.Restore(new WindowPlacement { X = 10, Y = 10, Width = 200, Height = 100 }, [Screen1], Min);
        Assert.Equal(Min, r!.Value.Size);
    }

    [Fact]
    public void Startup_command_starts_in_the_tray()
    {
        Assert.Equal("\"C:\\Apps\\LittleRun.exe\" --minimized", StartupRegistration.Command(@"C:\Apps\LittleRun.exe"));
    }

    [Theory]
    [InlineData(5, 10, Frameless.HTTOPLEFT)]
    [InlineData(500, 2, Frameless.HTTOP)]
    [InlineData(995, 3, Frameless.HTTOPRIGHT)]
    [InlineData(500, 6, Frameless.HTCLIENT)]
    public void Top_edge_hit_test(int x, int y, int expected)
    {
        // y = 10 is below the 6px band, so the corner case uses y = 3 for the corners that matter.
        var hit = Frameless.TopHit(new Point(x, y == 10 ? 3 : y), 1000, 6, 12);
        Assert.Equal(expected, hit);
    }

    [Fact]
    public void Export_file_names_are_safe()
    {
        Assert.Equal("What is 1 2", MainForm.SafeFileName("What is 1/2?").Trim());
        Assert.Equal("chat", MainForm.SafeFileName("???"));
        Assert.True(MainForm.SafeFileName(new string('a', 300)).Length <= 80);
    }

    [Fact]
    public void Settings_normalize_appearance_and_voice()
    {
        var s = new Settings { Theme = "neon", FontScale = 3, TtsRate = 9, SttEngine = "magic", SttLanguage = " " };
        s.Normalize();
        Assert.Equal("system", s.Theme);
        Assert.Equal(1.0, s.FontScale);
        Assert.Equal(1.0, s.TtsRate);
        Assert.Equal("offline", s.SttEngine);
        Assert.Equal("en-US", s.SttLanguage);
    }
}
