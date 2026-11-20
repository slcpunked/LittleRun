namespace LittleRun.Desktop;

/// <summary>Where LittleRun keeps its runtime, models, chats and logs.</summary>
/// <remarks>Default root is %LOCALAPPDATA%\LittleRun. LITTLERUN_HOME overrides it (tests, E2E runs).</remarks>
public sealed class AppPaths
{
    public AppPaths(string root)
    {
        Root = Path.GetFullPath(root);
    }

    public static AppPaths Default()
    {
        var env = Environment.GetEnvironmentVariable("LITTLERUN_HOME");
        if (!string.IsNullOrWhiteSpace(env))
            return new AppPaths(env);
        var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return new AppPaths(Path.Combine(local, "LittleRun"));
    }

    public string Root { get; }
    public string Runtime => Path.Combine(Root, "runtime");
    public string PayloadCache => Path.Combine(Runtime, "payload");
    public string PythonInstalls => Path.Combine(Runtime, "python");
    public string UvCache => Path.Combine(Runtime, "uv-cache");
    public string Venv => Path.Combine(Runtime, "venv");
    public string VenvPython => Path.Combine(Venv, "Scripts", "python.exe");
    public string Kernel => Path.Combine(Runtime, "kernel");
    public string RuntimeStamp => Path.Combine(Runtime, "installed.json");
    public string Models => Path.Combine(Root, "models");
    public string Logs => Path.Combine(Root, "logs");
    public string BackendLog => Path.Combine(Logs, "backend.log");
    public string SetupLog => Path.Combine(Logs, "setup.log");
    public string AppLog => Path.Combine(Logs, "app.log");
    public string Settings => Path.Combine(Root, "settings.json");
    public string Chats => Path.Combine(Root, "chats.json");
    public string WebViewData => Path.Combine(Root, "webview");

    public void EnsureBase()
    {
        Directory.CreateDirectory(Root);
        Directory.CreateDirectory(Logs);
    }
}
