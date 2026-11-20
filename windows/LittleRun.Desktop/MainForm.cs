using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using LittleRun.Desktop.Backend;
using LittleRun.Desktop.Setup;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace LittleRun.Desktop;

/// <summary>The app window: the chat UI in WebView2, plus a tray icon that keeps the model running.</summary>
public sealed class MainForm : Form
{
    internal const string Origin = "https://app.littlerun.local";

    private readonly AppController _app;
    private readonly WebView2 _web = new() { Dock = DockStyle.Fill, DefaultBackgroundColor = Color.Transparent };
    private readonly NotifyIcon _tray = new();
    private readonly ToolStripMenuItem _trayStatus = new("Starting…") { Enabled = false };
    private readonly SpeechInput _speech = new();
    private bool _quitting;
    private bool _webReady;
    private bool _restoreMaximized;

    /// <summary>Strip above the page (title-bar colour) for resizing from the top edge.</summary>
    private int TopBand => LogicalToDeviceUnits(5);

    public MainForm(AppController app, bool startHidden)
    {
        _app = app;
        Text = "LittleRun";
        Icon = AppIcon();
        // Sizable, but the caption is removed in WndProc (WM_NCCALCSIZE): the page draws the title bar.
        FormBorderStyle = FormBorderStyle.Sizable;
        DoubleBuffered = true;
        MinimumSize = new Size(760, 500);
        Size = new Size(1120, 780);
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = Color.FromArgb(0x15, 0x15, 0x18);
        Padding = new Padding(0, TopBand, 0, 0);
        var saved = WindowPlacement.Restore(app.Settings.Window, Screen.AllScreens.Select(s => s.WorkingArea).ToList(),
                                            MinimumSize);
        if (saved is { } r)
        {
            StartPosition = FormStartPosition.Manual;
            Bounds = r;
            _restoreMaximized = app.Settings.Window!.Maximized;
        }
        Controls.Add(_web);
        if (startHidden)
        {
            WindowState = FormWindowState.Minimized;
            ShowInTaskbar = false;
        }
        else if (_restoreMaximized)
        {
            Shown += (_, _) => WindowState = FormWindowState.Maximized;
        }
        _speech.Event += (kind, text) => Post(new { type = "event", name = "stt", data = new { kind, text } });
        // The installer's "start when I sign in" option writes the same Run key; show its real state.
        app.Settings.StartWithWindows = StartupRegistration.IsEnabled();

        _tray.Icon = Icon;
        _tray.Text = "LittleRun";
        _tray.Visible = true;
        _tray.DoubleClick += (_, _) => ShowFromTray();
        var menu = new ContextMenuStrip();
        menu.Items.Add(_trayStatus);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add("Open LittleRun", null, (_, _) => ShowFromTray());
        menu.Items.Add("Restart model", null, (_, _) => _app.StartBackendIfReady());
        menu.Items.Add("Open logs folder", null, (_, _) => OpenPath(_app.Paths.Logs));
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add("Quit LittleRun", null, (_, _) => Quit());
        _tray.ContextMenuStrip = menu;

        _app.Event += OnAppEvent;
        Load += async (_, _) => await InitWebAsync();
        if (startHidden)
            Shown += (_, _) => Hide();
    }

    internal static Icon AppIcon()
    {
        try
        {
            return Icon.ExtractAssociatedIcon(Environment.ProcessPath!) ?? SystemIcons.Application;
        }
        catch (ArgumentException)
        {
            return SystemIcons.Application;
        }
    }

    public void ShowFromTray()
    {
        ShowInTaskbar = true;
        Show();
        if (WindowState == FormWindowState.Minimized)
            WindowState = _restoreMaximized ? FormWindowState.Maximized : FormWindowState.Normal;
        Activate();
    }

    // ------------------------------------------------------------------ frameless window

    protected override void OnHandleCreated(EventArgs e)
    {
        base.OnHandleCreated(e);
        Frameless.Style(Handle);
        Frameless.FrameChanged(Handle);
    }

    protected override void WndProc(ref Message m)
    {
        if (m.Msg == Frameless.WM_NCCALCSIZE && m.WParam != IntPtr.Zero)
        {
            Frameless.CalcSize(Handle, m.LParam, WindowState == FormWindowState.Maximized);
            m.Result = IntPtr.Zero;
            return;
        }
        base.WndProc(ref m);
        if (m.Msg == Frameless.WM_NCHITTEST && m.Result == Frameless.HTCLIENT && WindowState == FormWindowState.Normal)
        {
            var p = PointToClient(new Point(unchecked((short)(long)m.LParam), unchecked((short)((long)m.LParam >> 16))));
            m.Result = Frameless.TopHit(p, ClientSize.Width, TopBand, LogicalToDeviceUnits(12));
        }
    }

    protected override void OnResize(EventArgs e)
    {
        base.OnResize(e);
        Padding = WindowState == FormWindowState.Maximized ? Padding.Empty : new Padding(0, TopBand, 0, 0);
        if (WindowState != FormWindowState.Minimized)
            _restoreMaximized = WindowState == FormWindowState.Maximized;
        if (_webReady)
            Post(new { type = "event", name = "window", data = new { maximized = WindowState == FormWindowState.Maximized } });
    }

    private void SavePlacement()
    {
        var b = WindowState == FormWindowState.Normal ? Bounds : RestoreBounds;
        if (b.Width <= 0 || b.Height <= 0)
            return;
        _app.Settings.Window = new WindowPlacement
        {
            X = b.X, Y = b.Y, Width = b.Width, Height = b.Height,
            Maximized = WindowState == FormWindowState.Maximized || (WindowState == FormWindowState.Minimized && _restoreMaximized),
        };
        _app.Settings.Save(_app.Paths);
    }

    private void ToggleMaximize() =>
        WindowState = WindowState == FormWindowState.Maximized ? FormWindowState.Normal : FormWindowState.Maximized;

    /// <summary>The page tells us its theme so the resize band and the window border match it.</summary>
    private void ApplyTheme(bool dark, string bg, string border)
    {
        try
        {
            BackColor = ColorTranslator.FromHtml(bg);
            Frameless.Theme(Handle, dark, ColorTranslator.FromHtml(border));
        }
        catch (Exception e) when (e is ArgumentException or FormatException) { }
    }

    private void Quit()
    {
        _quitting = true;
        Close();
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        SavePlacement();
        _speech.Stop();
        if (!_quitting && e.CloseReason == CloseReason.UserClosing && _app.Settings.CloseToTray)
        {
            e.Cancel = true;
            Hide();
            if (!_app.Settings.TrayHintShown)
            {
                _tray.ShowBalloonTip(4000, "LittleRun is still running",
                                     "The model stays loaded in the background. Right-click the tray icon to quit.",
                                     ToolTipIcon.Info);
                _app.Settings.TrayHintShown = true;
                _app.Settings.Save(_app.Paths);
            }
            return;
        }
        _tray.Visible = false;
        base.OnFormClosing(e);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _app.Event -= OnAppEvent;
            _speech.Dispose();
            _tray.Dispose();
            _web.Dispose();
        }
        base.Dispose(disposing);
    }

    // ------------------------------------------------------------------ WebView2

    private async Task InitWebAsync()
    {
        try
        {
            var env = await CoreWebView2Environment.CreateAsync(null, _app.Paths.WebViewData);
            await _web.EnsureCoreWebView2Async(env);
        }
        catch (Exception ex) when (ex is WebView2RuntimeNotFoundException or System.Runtime.InteropServices.COMException)
        {
            MessageBox.Show(this, "LittleRun needs the Microsoft Edge WebView2 Runtime.\n\n" + ex.Message,
                            "LittleRun", MessageBoxButtons.OK, MessageBoxIcon.Error);
            OpenUrl("https://go.microsoft.com/fwlink/p/?LinkId=2124703");
            Quit();
            return;
        }
        var core = _web.CoreWebView2;
        var s = core.Settings;
        s.AreDevToolsEnabled = Debugger.IsAttached || Environment.GetEnvironmentVariable("LITTLERUN_DEVTOOLS") == "1";
        s.IsStatusBarEnabled = false;
        s.IsZoomControlEnabled = true;
        s.AreBrowserAcceleratorKeysEnabled = s.AreDevToolsEnabled;
        s.IsPasswordAutosaveEnabled = false;
        s.IsGeneralAutofillEnabled = false;
        // `app-region: drag` in the page becomes the title bar: drag, double-click to maximize, snap.
        s.IsNonClientRegionSupportEnabled = true;
        s.IsZoomControlEnabled = false; // text size is a setting instead
        core.AddWebResourceRequestedFilter(Origin + "/*", CoreWebView2WebResourceContext.All);
        core.WebResourceRequested += ServeUi;
        core.NavigationStarting += (_, e) =>
        {
            if (!e.Uri.StartsWith(Origin + "/", StringComparison.OrdinalIgnoreCase))
            {
                e.Cancel = true;
                OpenUrl(e.Uri);
            }
        };
        core.NewWindowRequested += (_, e) =>
        {
            e.Handled = true;
            OpenUrl(e.Uri);
        };
        core.WebMessageReceived += OnWebMessage;
        core.Navigate(Origin + "/index.html");
    }

    private void ServeUi(object? sender, CoreWebView2WebResourceRequestedEventArgs e)
    {
        var path = new Uri(e.Request.Uri).AbsolutePath.TrimStart('/');
        if (path.Length == 0) path = "index.html";
        var stream = typeof(MainForm).Assembly.GetManifestResourceStream("ui/" + path);
        if (stream == null)
        {
            e.Response = _web.CoreWebView2.Environment.CreateWebResourceResponse(null, 404, "Not Found", "");
            return;
        }
        var type = Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".html" => "text/html; charset=utf-8",
            ".js" => "text/javascript; charset=utf-8",
            ".css" => "text/css; charset=utf-8",
            ".svg" => "image/svg+xml",
            ".png" => "image/png",
            ".woff2" => "font/woff2",
            _ => "application/octet-stream",
        };
        // Everything is local: block every network origin except this app.
        const string csp = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; " +
                           "script-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'";
        e.Response = _web.CoreWebView2.Environment.CreateWebResourceResponse(
            stream, 200, "OK", $"Content-Type: {type}\r\nCache-Control: no-store\r\nContent-Security-Policy: {csp}");
    }

    // ------------------------------------------------------------------ bridge

    private void OnAppEvent(string name, object data)
    {
        if (name == "backend" && data is BackendStatus st)
            BeginInvokeSafe(() =>
            {
                _trayStatus.Text = st.State == "ready" ? "Model ready" : st.Message;
                var tip = "LittleRun: " + (st.State == "ready" ? "ready" : st.State);
                _tray.Text = tip.Length > 63 ? tip[..63] : tip;
            });
        // Events before the page has asked for its initial state would arrive before it can handle them.
        if (_webReady)
            Post(new { type = "event", name, data });
    }

    private void Post(object message)
    {
        var json = JsonSerializer.Serialize(message, JsonFile.Options);
        BeginInvokeSafe(() => _web.CoreWebView2?.PostWebMessageAsJson(json));
    }

    private void BeginInvokeSafe(Action a)
    {
        if (IsDisposed || Disposing)
            return;
        try
        {
            if (InvokeRequired) BeginInvoke(a);
            else a();
        }
        catch (InvalidOperationException) { } // window handle gone during shutdown
    }

    private async void OnWebMessage(object? sender, CoreWebView2WebMessageReceivedEventArgs e)
    {
        if (!e.Source.StartsWith(Origin + "/", StringComparison.OrdinalIgnoreCase))
            return;
        JsonNode? msg;
        try { msg = JsonNode.Parse(e.WebMessageAsJson); }
        catch (JsonException) { return; }
        var id = msg?["id"]?.GetValue<string>();
        var type = msg?["type"]?.GetValue<string>() ?? "";
        var data = msg?["data"];
        try
        {
            var result = await HandleAsync(type, data);
            if (id != null)
                Post(new { type = "reply", id, ok = true, data = result });
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            if (id != null)
                Post(new { type = "reply", id, ok = false, error = ex.Message });
        }
    }

    private async Task<object?> HandleAsync(string type, JsonNode? d)
    {
        switch (type)
        {
            case "app.init":
                _webReady = true;
                return await Task.Run(InitData);
            case "setup.start":
            {
                var req = new SetupRequest(Str(d, "device") ?? "cpu", Str(d, "model") ?? _app.Payload.Info.DefaultModel,
                                           Str(d, "importDir"), d?["acceptLicense"]?.GetValue<bool>() ?? false);
                await Task.Run(() => _app.RunSetupAsync(req));
                return InitData();
            }
            case "setup.cancel":
                _app.CancelSetup();
                return null;
            case "setup.pickFolder":
                return PickFolder();
            case "setup.checkFolder":
            {
                var dir = Str(d, "dir") ?? "";
                var spec = _app.Payload.Info.Models[Str(d, "model") ?? _app.Payload.Info.DefaultModel];
                return new { missing = Directory.Exists(dir) ? ModelInstaller.MissingFiles(spec, dir) : spec.Files };
            }
            case "chats.load":
                return await Task.Run(_app.LoadChats);
            case "chats.save":
                if (d?["chats"] is JsonArray chats)
                {
                    var copy = (JsonArray)chats.DeepClone();
                    await Task.Run(() => _app.SaveChats(copy));
                }
                return null;
            case "gen.start":
                return await GenerateAsync(d);
            case "gen.stop":
                _app.StopGeneration(Str(d, "reqId") ?? "");
                return null;
            case "settings.save":
            {
                var next = d.Deserialize<Settings>(JsonFile.Options) ?? throw new ArgumentException("no settings");
                var restarted = _app.UpdateSettings(next);
                return new { settings = _app.Settings, restarted };
            }
            case "settings.newKey":
            {
                var next = d.Deserialize<Settings>(JsonFile.Options) ?? _app.Settings;
                next.ApiKey = Settings.NewKey();
                _app.UpdateSettings(next);
                return new { settings = _app.Settings };
            }
            case "story.warm":
                _ = Task.Run(() => _app.WarmStoryAsync());
                return null;
            case "backend.restart":
                _app.StartBackendIfReady();
                return null;
            case "open.logs":
                OpenPath(_app.Paths.Logs);
                return null;
            case "open.data":
                OpenPath(_app.Paths.Root);
                return null;
            case "open.url":
            {
                var url = Str(d, "url") ?? "";
                if (url.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
                    OpenUrl(url);
                return null;
            }
            case "clipboard.copy":
                Clipboard.SetText(Str(d, "text") ?? " ");
                return null;
            case "window.hide":
                Hide();
                return null;
            case "window.minimize":
                WindowState = FormWindowState.Minimized;
                return null;
            case "window.maximize":
                ToggleMaximize();
                return new { maximized = WindowState == FormWindowState.Maximized };
            case "window.close":
                BeginInvoke(Close); // honours "keep running in the tray"
                return null;
            case "window.theme":
                ApplyTheme(d?["dark"]?.GetValue<bool>() ?? true, Str(d, "bg") ?? "#1b1b1d", Str(d, "border") ?? "#2e2e33");
                return null;
            case "speech.start":
                if (_app.Settings.SttEngine == "voicetyping")
                {
                    SpeechInput.WindowsVoiceTyping();
                    return new { engine = "voicetyping" };
                }
                await Task.Run(() => _speech.Start(_app.Settings.SttLanguage));
                return new { engine = "offline", listening = _speech.Listening };
            case "speech.stop":
                _speech.Stop();
                return null;
            case "chat.export":
            {
                string? saved = null;
                using var dlg = new SaveFileDialog
                {
                    Title = "Export chat",
                    Filter = "Markdown (*.md)|*.md|Text (*.txt)|*.txt",
                    FileName = SafeFileName(Str(d, "name") ?? "chat") + ".md",
                    InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                };
                if (dlg.ShowDialog(this) == DialogResult.OK)
                {
                    await File.WriteAllTextAsync(dlg.FileName, Str(d, "markdown") ?? "");
                    saved = dlg.FileName;
                }
                return new { saved };
            }
            case "open.file":
            {
                var f = Str(d, "path");
                if (f != null && File.Exists(f))
                    Process.Start(new ProcessStartInfo("explorer.exe", $"/select,\"{f}\"") { UseShellExecute = true });
                return null;
            }
            default:
                throw new ArgumentException($"unknown request '{type}'");
        }
    }

    private async Task<object> GenerateAsync(JsonNode? d)
    {
        var reqId = Str(d, "reqId") ?? Guid.NewGuid().ToString("N");
        var msgs = new List<(string, string)>();
        if (d?["messages"] is JsonArray arr)
            foreach (var m in arr)
                msgs.Add((m?["role"]?.GetValue<string>() ?? "user", m?["content"]?.GetValue<string>() ?? ""));
        var g = new GenRequest(reqId, Str(d, "mode") ?? "chat", msgs, Str(d, "prompt"), d?["thinking"]?.GetValue<bool>(),
                               Str(d, "character"));
        try
        {
            var r = await Task.Run(() => _app.GenerateAsync(g));
            return new { reqId, stopped = false, r.FinishReason, r.PromptTokens, r.CompletionTokens, r.CachedTokens,
                         r.Seconds, r.FirstTokenSeconds };
        }
        catch (OperationCanceledException)
        {
            return new { reqId, stopped = true };
        }
        catch (HttpRequestException ex)
        {
            throw new ApiException("Lost the connection to the model: " + ex.Message);
        }
    }

    private object InitData()
    {
        Payload? payload = null;
        string? payloadError = null;
        try { payload = _app.Payload; }
        catch (InvalidOperationException ex) { payloadError = ex.Message; }
        return new
        {
            version = Application.ProductVersion.Split('+')[0],
            payloadError,
            setUp = payload != null && _app.IsSetUp,
            needsUpdate = payload != null && _app.NeedsRuntimeUpdate,
            settings = _app.Settings,
            system = SystemProbe.Probe(_app.Paths.Root),
            models = payload?.Info.Models,
            defaultModel = payload?.Info.DefaultModel,
            cudaMinDriver = payload?.Info.CudaMinDriver,
            minGpuMemoryMiB = SystemProbe.MinGpuMemoryMiB,
            presets = StoryChat.Presets,
            speechLanguages = SpeechInput.Languages(),
            maximized = WindowState == FormWindowState.Maximized,
            torch = payload?.Info.Torch,
            runtime = RuntimeInstaller.ReadStamp(_app.Paths),
            backend = _app.Backend.Status,
            paths = new { root = _app.Paths.Root, models = _app.Paths.Models },
        };
    }

    private string? PickFolder()
    {
        string? result = null;
        Invoke(() =>
        {
            using var dlg = new FolderBrowserDialog
            {
                Description = "Choose the folder that holds the model files (model.safetensors, config.json, …)",
                UseDescriptionForTitle = true,
            };
            if (dlg.ShowDialog(this) == DialogResult.OK)
                result = dlg.SelectedPath;
        });
        return result;
    }

    internal static string SafeFileName(string name)
    {
        var bad = Path.GetInvalidFileNameChars();
        var s = new string(name.Select(c => bad.Contains(c) ? ' ' : c).ToArray()).Trim();
        if (s.Length > 80) s = s[..80].Trim();
        return s.Length == 0 ? "chat" : s;
    }

    private static string? Str(JsonNode? d, string key) => d?[key] is JsonValue v && v.TryGetValue<string>(out var s) ? s : null;

    internal static void OpenUrl(string url)
    {
        try { Process.Start(new ProcessStartInfo(url) { UseShellExecute = true }); }
        catch (System.ComponentModel.Win32Exception) { }
    }

    private static void OpenPath(string path)
    {
        Directory.CreateDirectory(path);
        Process.Start(new ProcessStartInfo("explorer.exe", $"\"{path}\"") { UseShellExecute = true });
    }
}
