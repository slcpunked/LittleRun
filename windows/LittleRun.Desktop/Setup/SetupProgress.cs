namespace LittleRun.Desktop.Setup;

/// <summary>One update from a setup step, sent to the UI.</summary>
/// <param name="Step">Stable step id: runtime, torch, backend, kernel, verify, model, validate.</param>
/// <param name="State">running, done, failed or log.</param>
/// <param name="Message">Short text for the user, or one log line when State is "log".</param>
/// <param name="Fraction">0..1 when the step can measure progress.</param>
public sealed record SetupEvent(string Step, string State, string Message, double? Fraction = null,
                                long? Bytes = null, long? Total = null, double? BytesPerSecond = null);

public sealed class SetupFailedException(string step, string message) : Exception(message)
{
    public string Step { get; } = step;
}

/// <summary>Writes setup output to setup.log and forwards events.</summary>
public sealed class SetupReporter(string logPath, Action<SetupEvent> sink) : IDisposable
{
    private readonly StreamWriter _log = OpenLog(logPath);
    private readonly object _gate = new();

    private static StreamWriter OpenLog(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        return new StreamWriter(new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite)) { AutoFlush = true };
    }

    public void Report(SetupEvent e)
    {
        lock (_gate)
        {
            if (e.State != "log" || e.Message.Length > 0)
                _log.WriteLine($"{DateTime.Now:HH:mm:ss} [{e.Step}] {e.State}: {e.Message}");
        }
        sink(e);
    }

    public void Running(string step, string message, double? fraction = null) => Report(new SetupEvent(step, "running", message, fraction));
    public void Done(string step, string message) => Report(new SetupEvent(step, "done", message, 1));
    public void Log(string step, string line) => Report(new SetupEvent(step, "log", line));

    public void Dispose() => _log.Dispose();
}
