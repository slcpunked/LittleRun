using System.Globalization;
using System.Runtime.InteropServices;
using System.Speech.Recognition;

namespace LittleRun.Desktop;

/// <summary>
/// Speech to text with Windows' built-in offline recognizer (SAPI dictation). Results arrive as events:
/// partial guesses while speaking, then final phrases. Needs a microphone and a Windows speech language
/// (Settings > Time &amp; language > Speech).
/// </summary>
public sealed class SpeechInput : IDisposable
{
    private SpeechRecognitionEngine? _engine;
    private readonly object _gate = new();

    /// <summary>(kind, text): kind is "partial", "final", "state" ("listening"/"stopped") or "error".</summary>
    public event Action<string, string>? Event;

    public bool Listening { get; private set; }

    /// <summary>Recognizer cultures installed on this PC, e.g. "en-US".</summary>
    public static List<string> Languages()
    {
        try
        {
            return SpeechRecognitionEngine.InstalledRecognizers().Select(r => r.Culture.Name).Distinct().ToList();
        }
        catch (Exception e) when (e is COMException or InvalidOperationException or PlatformNotSupportedException)
        {
            return [];
        }
    }

    public void Start(string? language)
    {
        lock (_gate)
        {
            if (Listening)
                return;
            try
            {
                var installed = SpeechRecognitionEngine.InstalledRecognizers();
                if (installed.Count == 0)
                    throw new InvalidOperationException(
                        "No Windows speech recognizer is installed. Add a speech language in Windows Settings > Time & language > Speech.");
                var info = installed.FirstOrDefault(r => r.Culture.Name.Equals(language, StringComparison.OrdinalIgnoreCase))
                           ?? installed.FirstOrDefault(r => r.Culture.Name == CultureInfo.CurrentUICulture.Name)
                           ?? installed.FirstOrDefault(r => r.Culture.Name == "en-US")
                           ?? installed[0];
                var engine = new SpeechRecognitionEngine(info);
                engine.LoadGrammar(new DictationGrammar());
                engine.SetInputToDefaultAudioDevice();
                engine.SpeechHypothesized += (_, e) => Event?.Invoke("partial", e.Result.Text);
                engine.SpeechRecognized += (_, e) =>
                {
                    if (e.Result.Confidence > 0.2f)
                        Event?.Invoke("final", e.Result.Text);
                };
                engine.RecognizeCompleted += (_, e) =>
                {
                    if (e.Error != null)
                        Event?.Invoke("error", e.Error.Message);
                };
                engine.RecognizeAsync(RecognizeMode.Multiple);
                _engine = engine;
                Listening = true;
            }
            catch (Exception e) when (e is InvalidOperationException or COMException or ArgumentException)
            {
                _engine?.Dispose();
                _engine = null;
                var msg = e is InvalidOperationException && e.Message.Contains("audio", StringComparison.OrdinalIgnoreCase)
                    ? "No microphone was found." : e.Message;
                Event?.Invoke("error", msg);
                return;
            }
        }
        Event?.Invoke("state", "listening");
    }

    public void Stop()
    {
        SpeechRecognitionEngine? engine;
        lock (_gate)
        {
            engine = _engine;
            _engine = null;
            if (!Listening)
                return;
            Listening = false;
        }
        try
        {
            engine?.RecognizeAsyncStop();
        }
        catch (InvalidOperationException) { }
        // Dispose after pending results have been delivered.
        _ = Task.Delay(1500).ContinueWith(_ => engine?.Dispose(), TaskScheduler.Default);
        Event?.Invoke("state", "stopped");
    }

    public void Dispose() => Stop();

    /// <summary>Open Windows Voice Typing (Win+H) for the focused text box: online, usually more accurate.</summary>
    public static void WindowsVoiceTyping()
    {
        const ushort VK_LWIN = 0x5B, VK_H = 0x48;
        const uint KEYUP = 0x0002;
        var inputs = new[]
        {
            Key(VK_LWIN, 0), Key(VK_H, 0), Key(VK_H, KEYUP), Key(VK_LWIN, KEYUP),
        };
        SendInput((uint)inputs.Length, inputs, Marshal.SizeOf<INPUT>());

        static INPUT Key(ushort vk, uint flags) => new()
        {
            type = 1, // INPUT_KEYBOARD
            U = new InputUnion { ki = new KEYBDINPUT { wVk = vk, dwFlags = flags } },
        };
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct INPUT
    {
        public uint type;
        public InputUnion U;
    }

    [StructLayout(LayoutKind.Explicit)]
    private struct InputUnion
    {
        [FieldOffset(0)] public KEYBDINPUT ki;
        [FieldOffset(0)] public MOUSEINPUT mi;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct KEYBDINPUT
    {
        public ushort wVk, wScan;
        public uint dwFlags, time;
        public IntPtr dwExtraInfo;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MOUSEINPUT
    {
        public int dx, dy;
        public uint mouseData, dwFlags, time;
        public IntPtr dwExtraInfo;
    }

    [DllImport("user32.dll", SetLastError = true)]
    private static extern uint SendInput(uint count, INPUT[] inputs, int size);
}
