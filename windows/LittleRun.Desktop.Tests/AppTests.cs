using System.Net;
using System.Net.Sockets;
using System.Text.Json.Nodes;
using LittleRun.Desktop.Backend;
using Xunit;

namespace LittleRun.Desktop.Tests;

public class SettingsTests
{
    [Fact]
    public void Normalize_clamps_hand_edited_values()
    {
        var s = new Settings { Device = "tpu", Port = 99999, ApiKey = "short", MaxContext = 10, MaxTokens = 0, Temperature = 9,
                               TopP = 0, RepetitionPenalty = -1 };
        s.Normalize();
        Assert.Equal("cpu", s.Device);
        Assert.Equal(11435, s.Port);
        Assert.StartsWith("lr-", s.ApiKey);
        Assert.Equal(512, s.MaxContext);
        Assert.Equal(1, s.MaxTokens);
        Assert.Equal(2.0, s.Temperature);
        Assert.Equal(0.8, s.TopP);
        Assert.Equal(1.15, s.RepetitionPenalty);
    }

    [Fact]
    public void Round_trips_and_keeps_its_key()
    {
        using var home = new TempHome();
        var s = Settings.Load(home.Paths);
        s.SystemPrompt = "Be brief.";
        s.Save(home.Paths);
        var again = Settings.Load(home.Paths);
        Assert.Equal("Be brief.", again.SystemPrompt);
        Assert.Equal(s.ApiKey, again.ApiKey);
    }

    [Fact]
    public void A_corrupt_file_is_set_aside_and_defaults_are_used()
    {
        using var home = new TempHome();
        File.WriteAllText(home.Paths.Settings, "{ not json");
        var s = Settings.Load(home.Paths);
        Assert.Equal("cpu", s.Device);
        Assert.Single(Directory.GetFiles(home.Dir, "settings.json.corrupt-*"));
    }
}

public class AppControllerTests
{
    [Fact]
    public void Chats_round_trip_and_a_corrupt_file_is_set_aside()
    {
        using var home = new TempHome();
        using var app = new AppController(home.Paths);
        Assert.Empty(app.LoadChats());
        app.SaveChats([new JsonObject { ["id"] = "a", ["title"] = "Hi" }]);
        Assert.Equal("Hi", app.LoadChats()[0]!["title"]!.GetValue<string>());

        File.WriteAllText(home.Paths.Chats, "[{");
        Assert.Empty(app.LoadChats());
        Assert.Single(Directory.GetFiles(home.Dir, "chats.json.corrupt-*"));
    }

    [Fact]
    public void Settings_from_the_ui_cannot_change_setup_owned_fields()
    {
        using var home = new TempHome();
        using var app = new AppController(home.Paths);
        app.UsePayloadDirectory(TestPayload.Create(home.Dir));
        app.Settings.ModelDir = @"C:\models\m";
        app.Settings.Device = "cuda";
        var next = new Settings { ModelDir = @"C:\evil", Device = "cpu", MaxContext = 8192, ApiKey = app.Settings.ApiKey };
        var restart = app.UpdateSettings(next);
        Assert.True(restart); // context length changed
        Assert.Equal(@"C:\models\m", app.Settings.ModelDir);
        Assert.Equal("cuda", app.Settings.Device);
        Assert.Equal(8192, Settings.Load(home.Paths).MaxContext);
    }

    [Fact]
    public void Not_set_up_without_runtime_or_model()
    {
        using var home = new TempHome();
        using var app = new AppController(home.Paths);
        app.UsePayloadDirectory(TestPayload.Create(home.Dir));
        Assert.False(app.IsSetUp);
        Assert.Null(app.CurrentModel);
    }

    [Fact]
    public async Task Generate_refuses_while_the_model_is_not_ready()
    {
        using var home = new TempHome();
        using var app = new AppController(home.Paths);
        app.UsePayloadDirectory(TestPayload.Create(home.Dir));
        app.Settings.ModelName = "m";
        var e = await Assert.ThrowsAsync<ApiException>(() =>
            app.GenerateAsync(new GenRequest("r", "chat", [("user", "hi")], null, null)));
        Assert.Contains("loading", e.Message);
    }
}

public class BackendProcessTests
{
    [Fact]
    public void Serve_args_bind_localhost_and_require_the_key()
    {
        var a = BackendProcess.ServeArgs(@"C:\m", "test-model", 1234, "cpu", 4096);
        Assert.Equal(["-m", "littlerun", "serve", @"C:\m"], a.Take(4));
        Assert.Equal("127.0.0.1", a[a.IndexOf("--host") + 1]);
        Assert.Equal("1234", a[a.IndexOf("--port") + 1]);
        Assert.Equal("LITTLERUN_API_KEY", a[a.IndexOf("--api-key-env") + 1]);
        Assert.Equal("cpu", a[a.IndexOf("--device") + 1]);
    }

    [Fact]
    public void Picks_the_preferred_port_unless_it_is_taken()
    {
        var busy = new TcpListener(IPAddress.Loopback, 0);
        busy.Start();
        try
        {
            var taken = ((IPEndPoint)busy.LocalEndpoint).Port;
            var other = BackendProcess.PickPort(taken);
            Assert.NotEqual(taken, other);
            Assert.InRange(other, 1, 65535);
        }
        finally
        {
            busy.Stop();
        }
        var free = BackendProcess.PickPort(0);
        Assert.Equal(free, BackendProcess.PickPort(free));
    }

    [Fact]
    public void Last_error_prefers_the_raised_exception()
    {
        var tail = "INFO starting\nTraceback (most recent call last):\n  File x\nlittlerun.errors.CompatibilityError: estimated 9 GiB exceeds\nINFO bye";
        Assert.Equal("littlerun.errors.CompatibilityError: estimated 9 GiB exceeds", BackendProcess.LastError(tail));
        Assert.Equal("last line", BackendProcess.LastError("first\nlast line"));
    }

    [Fact]
    public void Start_without_setup_reports_an_error_instead_of_throwing()
    {
        using var home = new TempHome();
        using var b = new BackendProcess(home.Paths);
        var seen = new List<string>();
        b.StatusChanged += s => seen.Add(s.State);
        b.Start(new Settings(), new ModelSpec { ModelId = "m" });
        Assert.Equal("error", b.Status.State);
        Assert.Contains("error", seen);
    }
}

public class SystemProbeTests
{
    [Fact]
    public void Parses_nvidia_smi_output()
    {
        var g = SystemProbe.ParseNvidiaSmi("NVIDIA GeForce RTX 5060 Ti, 610.88, 16311\n");
        Assert.NotNull(g);
        Assert.Equal("NVIDIA GeForce RTX 5060 Ti", g!.Name);
        Assert.Equal(610.88, g.DriverVersion, 3);
        Assert.Equal(16311, g.MemoryMiB);
        Assert.Null(SystemProbe.ParseNvidiaSmi("No devices were found"));
    }

    [Theory]
    [InlineData(16311, 610.88, null)]       // RTX 5060 Ti 16 GB
    [InlineData(15360, 610.0, null)]        // 16 GB cards that report a little less
    [InlineData(12282, 610.0, "16 GB")]     // 12 GB: too tight with Windows on the card
    [InlineData(8188, 610.0, "16 GB")]
    [InlineData(16311, 560.0, "driver")]
    public void Gpu_needs_16_gb_and_a_current_driver(long mib, double driver, string? problem)
    {
        var p = SystemProbe.GpuProblem(new GpuInfo("Test GPU", driver, mib), 580);
        if (problem == null) Assert.Null(p);
        else Assert.Contains(problem, p);
        Assert.NotNull(SystemProbe.GpuProblem(null, 580));
    }

    [Fact]
    public void Probe_reports_memory_and_disk()
    {
        using var home = new TempHome();
        var f = SystemProbe.Probe(home.Dir);
        Assert.True(f.TotalRamBytes > 0 && f.AvailableRamBytes > 0 && f.FreeDiskBytes > 0);
        Assert.True(f.LogicalCores > 0);
    }
}
