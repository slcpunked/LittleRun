using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using Microsoft.Win32;

namespace LittleRun.Desktop;

public sealed record GpuInfo(string Name, double DriverVersion, long MemoryMiB);

public sealed record SystemFacts(
    string Cpu,
    int LogicalCores,
    long TotalRamBytes,
    long AvailableRamBytes,
    long FreeDiskBytes,
    GpuInfo? NvidiaGpu,
    string? WebView2Version);

/// <summary>Hardware facts used by setup to pick CPU or GPU and to warn about memory or disk.</summary>
public static class SystemProbe
{
    /// <summary>
    /// GPU memory the 8B checkpoint needs. Measured peak on an RTX 5060 Ti: 11.0 GB reserved at a 4096-token
    /// context, before the CUDA context and what Windows and other apps keep on the card. 16 GB cards report
    /// 15.9-16.4 GB, so the line sits at 15,000 MiB.
    /// </summary>
    public const int MinGpuMemoryMiB = 15000;

    /// <summary>Why this GPU cannot be used, or null when it can.</summary>
    public static string? GpuProblem(GpuInfo? gpu, double minDriver)
    {
        if (gpu == null)
            return "No NVIDIA graphics card was found.";
        if (gpu.DriverVersion < minDriver)
            return $"The GPU needs NVIDIA driver {minDriver} or newer (installed: {gpu.DriverVersion}).";
        if (gpu.MemoryMiB < MinGpuMemoryMiB)
            return $"The GPU needs 16 GB of memory; the {gpu.Name} has {gpu.MemoryMiB / 1024.0:0} GB. Use the CPU instead.";
        return null;
    }

    public static SystemFacts Probe(string dataRoot)
    {
        var (total, avail) = Memory();
        long free = 0;
        try
        {
            free = new DriveInfo(Path.GetPathRoot(Path.GetFullPath(dataRoot))!).AvailableFreeSpace;
        }
        catch (IOException) { }
        catch (ArgumentException) { }
        return new SystemFacts(CpuName(), Environment.ProcessorCount, total, avail, free, NvidiaGpu(), WebView2());
    }

    public static string? WebView2()
    {
        try
        {
            return Microsoft.Web.WebView2.Core.CoreWebView2Environment.GetAvailableBrowserVersionString();
        }
        catch (Exception) // WebView2RuntimeNotFoundException and loader failures
        {
            return null;
        }
    }

    private static string CpuName()
    {
        using var k = Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
        return (k?.GetValue("ProcessorNameString") as string)?.Trim() ?? "Unknown CPU";
    }

    /// <summary>Parse one line of `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits`.</summary>
    internal static GpuInfo? ParseNvidiaSmi(string output)
    {
        foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var parts = line.Split(',', StringSplitOptions.TrimEntries);
            if (parts.Length < 3)
                continue;
            var drv = parts[1].Split('.');
            if (!double.TryParse(drv.Length >= 2 ? $"{drv[0]}.{drv[1]}" : drv[0], NumberStyles.Float,
                                 CultureInfo.InvariantCulture, out var driver))
                continue;
            long.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out var mem);
            return new GpuInfo(parts[0], driver, mem);
        }
        return null;
    }

    private static GpuInfo? NvidiaGpu()
    {
        var exe = Path.Combine(Environment.SystemDirectory, "nvidia-smi.exe");
        if (!File.Exists(exe))
            return null;
        try
        {
            var psi = new ProcessStartInfo(exe, "--query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits")
            {
                UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true,
            };
            using var p = Process.Start(psi)!;
            var output = p.StandardOutput.ReadToEnd();
            return p.WaitForExit(10_000) && p.ExitCode == 0 ? ParseNvidiaSmi(output) : null;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return null;
        }
    }

    private static (long total, long avail) Memory()
    {
        var st = new MEMORYSTATUSEX { dwLength = (uint)Marshal.SizeOf<MEMORYSTATUSEX>() };
        return GlobalMemoryStatusEx(ref st) ? ((long)st.ullTotalPhys, (long)st.ullAvailPhys) : (0, 0);
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MEMORYSTATUSEX
    {
        public uint dwLength, dwMemoryLoad;
        public ulong ullTotalPhys, ullAvailPhys, ullTotalPageFile, ullAvailPageFile, ullTotalVirtual, ullAvailVirtual,
                     ullAvailExtendedVirtual;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GlobalMemoryStatusEx(ref MEMORYSTATUSEX st);
}
