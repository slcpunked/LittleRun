using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace LittleRun.Desktop;

internal static class Program
{
    [STAThread]
    private static int Main(string[] args)
    {
        if (args.Contains("--selftest") || args.Contains("--e2e"))
        {
            AttachConsole(-1); // show output when started from a terminal
            return Headless.Run(args);
        }

        var paths = AppPaths.Default();
        // One instance per data folder: a second launch brings the running window forward instead.
        var key = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(paths.Root.ToLowerInvariant())))[..16];
        using var mutex = new Mutex(true, $@"Local\LittleRun.{key}", out var first);
        using var activate = new EventWaitHandle(false, EventResetMode.AutoReset, $@"Local\LittleRun.{key}.activate");
        if (!first)
        {
            activate.Set();
            return 0;
        }

        ApplicationConfiguration.Initialize();
        Application.SetUnhandledExceptionMode(UnhandledExceptionMode.CatchException);
        Application.ThreadException += (_, e) => LogCrash(paths, e.Exception);
        AppDomain.CurrentDomain.UnhandledException += (_, e) => LogCrash(paths, e.ExceptionObject as Exception);

        using var app = new AppController(paths);
        using var form = new MainForm(app, startHidden: args.Contains("--minimized"));
        var waiter = ThreadPool.RegisterWaitForSingleObject(activate, (_, _) =>
        {
            try { form.BeginInvoke(form.ShowFromTray); }
            catch (InvalidOperationException) { }
        }, null, Timeout.Infinite, executeOnlyOnce: false);
        // Load the model as soon as possible; the window shows its progress.
        Task.Run(() =>
        {
            try { app.StartBackendIfReady(); }
            catch (Exception e) when (e is InvalidOperationException or IOException) { LogCrash(paths, e); }
        });
        Application.Run(form);
        waiter.Unregister(null);
        return 0;
    }

    private static void LogCrash(AppPaths paths, Exception? e)
    {
        try
        {
            Directory.CreateDirectory(paths.Logs);
            File.AppendAllText(paths.AppLog, $"{DateTime.Now:u} {e}\n");
        }
        catch (IOException) { }
    }

    [DllImport("kernel32.dll")]
    private static extern bool AttachConsole(int processId);
}
