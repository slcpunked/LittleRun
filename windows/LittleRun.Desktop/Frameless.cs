using System.Runtime.InteropServices;

namespace LittleRun.Desktop;

/// <summary>
/// A normal resizable window whose caption is removed (WM_NCCALCSIZE), so the page draws its own title bar.
/// Windows keeps the invisible resize borders, rounded corners, shadow, snapping and maximize behaviour.
/// Only the top edge needs help: it has no invisible border, so a thin strip above the page resizes it.
/// </summary>
internal static class Frameless
{
    public const int WM_NCCALCSIZE = 0x0083;
    public const int WM_NCHITTEST = 0x0084;
    public const int HTCLIENT = 1, HTTOP = 12, HTTOPLEFT = 13, HTTOPRIGHT = 14;
    private const int SM_CXSIZEFRAME = 32, SM_CYSIZEFRAME = 33, SM_CXPADDEDBORDER = 92;

    /// <summary>Width (x) and height (y) of the invisible resize border at the window's DPI.</summary>
    public static (int x, int y) Frame(IntPtr hwnd)
    {
        var dpi = GetDpiForWindow(hwnd);
        if (dpi == 0) dpi = 96;
        var pad = GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
        return (GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi) + pad, GetSystemMetricsForDpi(SM_CYSIZEFRAME, dpi) + pad);
    }

    /// <summary>Client area = window minus the side and bottom borders (and the top too when maximized).</summary>
    public static void CalcSize(IntPtr hwnd, IntPtr lParam, bool maximized)
    {
        var p = Marshal.PtrToStructure<NCCALCSIZE_PARAMS>(lParam);
        var (fx, fy) = Frame(hwnd);
        p.rgrc0.Left += fx;
        p.rgrc0.Right -= fx;
        p.rgrc0.Bottom -= fy;
        if (maximized)
            p.rgrc0.Top += fy; // a maximized window hangs over the screen edge by the border size
        Marshal.StructureToPtr(p, lParam, false);
    }

    /// <summary>Top-edge resize inside the client area (y within <paramref name="band"/> pixels of the top).</summary>
    public static int TopHit(Point client, int width, int band, int corner)
    {
        if (client.Y >= band)
            return HTCLIENT;
        if (client.X < corner) return HTTOPLEFT;
        if (client.X >= width - corner) return HTTOPRIGHT;
        return HTTOP;
    }

    /// <summary>Re-run WM_NCCALCSIZE after the window is created.</summary>
    public static void FrameChanged(IntPtr hwnd) =>
        SetWindowPos(hwnd, IntPtr.Zero, 0, 0, 0, 0, 0x0020 /* FRAMECHANGED */ | 0x0002 | 0x0001 | 0x0004 | 0x0010);

    public static void Style(IntPtr hwnd)
    {
        var round = 2; // DWMWCP_ROUND (Windows 11)
        DwmSetWindowAttribute(hwnd, 33 /* DWMWA_WINDOW_CORNER_PREFERENCE */, ref round, sizeof(int));
    }

    public static void Theme(IntPtr hwnd, bool dark, Color border)
    {
        var d = dark ? 1 : 0;
        DwmSetWindowAttribute(hwnd, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, ref d, sizeof(int));
        var colorref = border.R | (border.G << 8) | (border.B << 16);
        DwmSetWindowAttribute(hwnd, 34 /* DWMWA_BORDER_COLOR */, ref colorref, sizeof(int));
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct RECT
    {
        public int Left, Top, Right, Bottom;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NCCALCSIZE_PARAMS
    {
        public RECT rgrc0, rgrc1, rgrc2;
        public IntPtr lppos;
    }

    [DllImport("user32.dll")]
    private static extern uint GetDpiForWindow(IntPtr hwnd);

    [DllImport("user32.dll")]
    private static extern int GetSystemMetricsForDpi(int index, uint dpi);

    [DllImport("user32.dll")]
    private static extern bool SetWindowPos(IntPtr hwnd, IntPtr after, int x, int y, int cx, int cy, uint flags);

    [DllImport("dwmapi.dll")]
    private static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int value, int size);
}
