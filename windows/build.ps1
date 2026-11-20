<#
.SYNOPSIS
  Builds the LittleRun Windows app: dist\LittleRun.exe (+ zip, checksums and, with Inno Setup, an installer).

.DESCRIPTION
  1. Checks the tools (.NET SDK 10+, Visual Studio C++ Build Tools, Node.js for UI tests).
  2. Downloads the pinned uv and makes a build venv with the pinned Python and PyTorch (CPU).
  3. Compiles the packed 1-bit kernel with MSVC for that PyTorch.
  4. Runs the Python tests (using the prebuilt kernel), the UI tests and the app tests.
  5. Builds the backend wheel and pinned requirements, and stages them with uv as the app payload.
  6. Publishes LittleRun.exe (self-contained, single file) with the payload inside, and self-tests it.
  7. Optional -E2E: installs the runtime into a scratch folder with the real exe, loads the model and runs a
     completion and a chat. Uses -ModelDir (or models\<name> when present) so it does not download 2.8 GB.

.EXAMPLE
  build.bat                 everything except the end-to-end run
  build.bat -E2E            also install, load the model and generate with the built exe
  build.bat -SkipTests      faster; the exe is still self-tested
#>
[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$E2E,
    [string]$ModelDir,
    [switch]$NoKernel,
    [switch]$NoInstaller,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# ------------------------------------------------------------------ pins (change together)
$UvVersion     = "0.13.9"
$UvSha256      = "edd61f7f5728a56cc62455386661860bdfc7bab65872f39b0c25bdcc1d8e1beb"
$PythonVersion = "3.12.16"
$TorchVersion  = "2.15.0"
$CpuIndex      = "https://download.pytorch.org/whl/cpu"
$CudaIndex     = "https://download.pytorch.org/whl/cu132"
$CudaMinDriver = 580   # CUDA 13.x needs an R580+ NVIDIA driver

$Root    = Split-Path -Parent $PSScriptRoot
$Build   = Join-Path $Root "build"
$Dist    = Join-Path $Root "dist"
$Tools   = Join-Path $Build "tools"
$Payload = Join-Path $Build "payload"
$Venv    = Join-Path $Build "venv"
$Py      = Join-Path $Venv "Scripts\python.exe"
$Desktop = Join-Path $PSScriptRoot "LittleRun.Desktop"
$Tests   = Join-Path $PSScriptRoot "LittleRun.Desktop.Tests"
$Started = Get-Date
$script:StepNo = 0

function Step([string]$text) {
    $script:StepNo++
    Write-Host ""
    Write-Host ("[{0}] {1}" -f $script:StepNo, $text) -ForegroundColor Cyan
}
function Fail([string]$text) {
    Write-Host ""
    Write-Host "BUILD FAILED: $text" -ForegroundColor Red
    exit 1
}
function Invoke-Checked([string]$what, [scriptblock]$block) {
    & $block
    if ($LASTEXITCODE -ne 0) { Fail "$what (exit $LASTEXITCODE)" }
}
# SHA-256 with .NET directly: Get-FileHash is missing when Windows PowerShell runs from a pwsh terminal.
function Get-Sha256([string]$path) {
    $s = [System.IO.File]::OpenRead($path)
    try { return ([System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash($s)) -replace "-", "").ToLowerInvariant() }
    finally { $s.Dispose() }
}

Write-Host "LittleRun Windows build" -ForegroundColor White
Write-Host "repo $Root"

if ($Clean) {
    foreach ($d in @($Payload, $Venv, (Join-Path $Build "kernel-build"), $Dist)) {
        if (Test-Path $d) { Remove-Item -Recurse -Force $d }
    }
}

# ------------------------------------------------------------------ 1. tools
Step "Checking tools"
$dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
if (-not $dotnet) { Fail ".NET SDK not found. Install the .NET 10 SDK: winget install Microsoft.DotNet.SDK.10" }
$sdks = & dotnet --list-sdks | ForEach-Object { [version](($_ -split " ")[0] -replace "-.*$", "") }
if (-not ($sdks | Where-Object { $_.Major -ge 10 })) { Fail ".NET SDK 10 or newer is required (found: $($sdks -join ', '))" }
Write-Host ("  .NET SDK  {0}" -f (($sdks | Sort-Object -Descending | Select-Object -First 1)))

$vcvars = $null
if (-not $NoKernel) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vs) { $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat" }
    }
    if (-not $vcvars -or -not (Test-Path $vcvars)) {
        Fail ("Visual Studio C++ Build Tools not found (needed for the fast CPU kernel). Install them with:`n" +
              "  winget install Microsoft.VisualStudio.2022.BuildTools --override `"--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive`"`n" +
              "or build without the kernel: build.bat -NoKernel")
    }
    Write-Host "  MSVC      $vs"
}
$node = Get-Command node -ErrorAction SilentlyContinue
if ($node) { Write-Host "  Node.js   $(& node --version)" } else { Write-Host "  Node.js   not found; UI tests will be skipped" -ForegroundColor Yellow }

$version = (Select-String -Path (Join-Path $Root "pyproject.toml") -Pattern '^version\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
Write-Host "  version   $version"

# ------------------------------------------------------------------ 2. uv and the build venv
Step "Getting uv $UvVersion"
$uv = Join-Path $Tools "uv\uv.exe"
if (-not (Test-Path $uv) -or -not ((& $uv --version) -match [regex]::Escape($UvVersion))) {
    New-Item -ItemType Directory -Force $Tools | Out-Null
    $zip = Join-Path $Tools "uv.zip"
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
    $got = Get-Sha256 $zip
    if ($got -ne $UvSha256) { Fail "uv download failed its SHA-256 check ($got)" }
    if (Test-Path (Join-Path $Tools "uv")) { Remove-Item -Recurse -Force (Join-Path $Tools "uv") }
    Expand-Archive $zip (Join-Path $Tools "uv")
    Remove-Item $zip
}
Write-Host "  $(& $uv --version)"

$env:UV_PYTHON_INSTALL_DIR = Join-Path $Build "python"
$env:UV_CACHE_DIR = Join-Path $Build "uv-cache"
$env:UV_PYTHON_PREFERENCE = "only-managed"
$env:UV_LINK_MODE = "copy"
$env:UV_NO_PROGRESS = "1"

Step "Build venv: Python $PythonVersion, PyTorch $TorchVersion (CPU)"
$haveVenv = (Test-Path $Py) -and ((& $Py -c "import sys,torch;print(sys.version.split()[0], torch.__version__)" 2>$null) -eq "$PythonVersion $TorchVersion+cpu")
if (-not $haveVenv) {
    Invoke-Checked "creating the build venv" { & $uv venv $Venv --python $PythonVersion --clear --no-project -q }
    Invoke-Checked "installing PyTorch" { & $uv pip install --python $Py --index-url $CpuIndex "torch==$TorchVersion" -q }
}
Invoke-Checked "installing LittleRun for tests" { & $uv pip install --python $Py -e "$Root[test]" ninja --no-config -q }
Write-Host "  $(& $Py -c "import sys,torch;print('Python', sys.version.split()[0], '| torch', torch.__version__)")"

# ------------------------------------------------------------------ 3. kernel
if (Test-Path $Payload) { Remove-Item -Recurse -Force $Payload }
New-Item -ItemType Directory -Force $Payload | Out-Null
$kernelDir = Join-Path $Payload "kernel"
if ($NoKernel) {
    Step "Skipping the CPU kernel (-NoKernel): the app will use the slower int8 path"
} else {
    Step "Compiling the packed 1-bit CPU kernel with MSVC"
    $work = Join-Path $Build "kernel-build"
    $script = Join-Path $PSScriptRoot "tools\build_kernel.py"
    # vcvars64.bat looks for vswhere.exe on PATH and complains (harmlessly) when it is not there.
    $vsInstaller = Split-Path $vswhere
    Invoke-Checked "compiling the kernel" { cmd /c "set `"PATH=%PATH%;$vsInstaller`" && call `"$vcvars`" >nul && `"$Py`" `"$script`" --out `"$kernelDir`" --work `"$work`"" }
    # The kernel uses OpenMP (vcomp140.dll), which is not on every PC: ship it next to the kernel.
    $redist = Join-Path (Split-Path (Split-Path (Split-Path $vcvars))) "Redist\MSVC"
    $vcomp = Get-ChildItem $redist -Recurse -Filter vcomp140.dll -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match '\\x64\\Microsoft\.VC\d+\.OpenMP\\' -and $_.FullName -notmatch '\\onecore\\' -and $_.FullName -notmatch '\\debug_' } |
        Sort-Object FullName -Descending | Select-Object -First 1
    if (-not $vcomp) { Fail "vcomp140.dll (OpenMP runtime) not found under $redist" }
    Copy-Item $vcomp.FullName $kernelDir
    Write-Host "  OpenMP runtime: $($vcomp.FullName)"
}

# ------------------------------------------------------------------ 4. tests
if ($SkipTests) {
    Step "Skipping tests (-SkipTests)"
} else {
    Step "Python tests"
    $env:LITTLERUN_KERNEL_DIR = if ($NoKernel) { "" } else { $kernelDir }
    $tmp = Join-Path $Build "pytest-tmp"
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
    Push-Location $Root
    try { Invoke-Checked "Python tests" { & $Py -m pytest (Join-Path $Root "tests") -q "--rootdir=$Root" "--basetemp=$tmp" -p no:cacheprovider } }
    finally { Pop-Location; Remove-Item Env:LITTLERUN_KERNEL_DIR -ErrorAction SilentlyContinue }
    if (-not $NoKernel) {
        $env:LITTLERUN_KERNEL_DIR = $kernelDir
        $doctor = & $Py -m littlerun doctor 2>&1 | Out-String
        Remove-Item Env:LITTLERUN_KERNEL_DIR
        if ($doctor -notmatch "packed_kernel\s+ok \(prebuilt\)") { Fail "the prebuilt kernel did not load:`n$doctor" }
        Write-Host "  prebuilt kernel loads without a compiler"
    }

    Step "UI tests"
    if ($node) { Invoke-Checked "UI tests" { & node --test (Join-Path $Desktop "ui\tests") } }
    else { Write-Host "  skipped (no Node.js)" -ForegroundColor Yellow }

    Step "App tests"
    Invoke-Checked "app tests" { & dotnet test $Tests -c Release --nologo -v q }
}

# ------------------------------------------------------------------ 5. payload
Step "Staging the backend payload"
Copy-Item $uv (Join-Path $Payload "uv.exe")
Push-Location $Root
try {
    $wheelOut = Join-Path $Build "wheel"
    if (Test-Path $wheelOut) { Remove-Item -Recurse -Force $wheelOut }
    Invoke-Checked "building the wheel" { & $uv build $Root --wheel --out-dir $wheelOut -q }
} finally { Pop-Location }
$wheel = Get-ChildItem $wheelOut -Filter "littlerun-$version-*.whl" | Select-Object -First 1
if (-not $wheel) { Fail "the wheel for version $version was not built" }
Copy-Item $wheel.FullName $Payload
$constraints = Join-Path $Build "constraints.txt"
"torch==$TorchVersion" | Set-Content $constraints -Encoding ascii
Invoke-Checked "pinning the runtime requirements" {
    & $uv pip compile (Join-Path $Root "pyproject.toml") --python-version 3.12 --python-platform x86_64-pc-windows-msvc `
        --constraint $constraints --no-emit-package torch --no-header -q -o (Join-Path $Payload "requirements.txt")
}
Invoke-Checked "writing payload.json" {
    & $Py (Join-Path $PSScriptRoot "tools\stage_payload.py") --payload $Payload --repo-root $Root --version $version `
        --uv $UvVersion --python $PythonVersion --torch $TorchVersion --cpu-index $CpuIndex --cuda-index $CudaIndex `
        --cuda-min-driver $CudaMinDriver --wheel $wheel.Name
}
Get-ChildItem $Payload -Recurse -File | ForEach-Object {
    Write-Host ("  {0,-48} {1,10:N0} KB" -f $_.FullName.Substring($Payload.Length + 1), ($_.Length / 1KB))
}

# ------------------------------------------------------------------ 6. app
Step "Publishing LittleRun.exe"
$running = Get-Process LittleRun -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($Dist, [StringComparison]::OrdinalIgnoreCase) }
if ($running) { Fail "LittleRun is running from $Dist. Quit it (tray icon > Quit LittleRun) and build again." }
if (Test-Path $Dist) { Remove-Item -Recurse -Force $Dist }
$publish = Join-Path $Build "publish"
if (Test-Path $publish) { Remove-Item -Recurse -Force $publish }
Invoke-Checked "publishing the app" {
    & dotnet publish $Desktop -c Release -r win-x64 --self-contained true --nologo -v q `
        -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -p:EnableCompressionInSingleFile=true `
        -p:DebugType=none -p:Version=$version -p:PayloadDir="$Payload\" -o $publish
}
New-Item -ItemType Directory -Force $Dist | Out-Null
$exe = Join-Path $Dist "LittleRun.exe"
Copy-Item (Join-Path $publish "LittleRun.exe") $exe
Write-Host ("  {0} ({1:N1} MB)" -f $exe, ((Get-Item $exe).Length / 1MB))

Step "Self-test of the built exe"
$report = Join-Path $Build "selftest.txt"
$p = Start-Process $exe -ArgumentList "--selftest", "--report", "`"$report`"" -Wait -PassThru -NoNewWindow
if ($p.ExitCode -ne 0) { Get-Content $report | ForEach-Object { Write-Host "  $_" }; Fail "self-test failed" }

# ------------------------------------------------------------------ 7. end to end
if ($E2E) {
    Step "End-to-end: install the runtime, load the model, generate (real exe)"
    if (-not $ModelDir) {
        $candidate = Join-Path $Root "models\littlebit-qwen3-8b-0.55bpw"
        if (Test-Path (Join-Path $candidate "model.safetensors")) { $ModelDir = $candidate }
    }
    $e2eHome = Join-Path $Build "e2e-home"
    if (Test-Path $e2eHome) { Remove-Item -Recurse -Force $e2eHome }
    $env:LITTLERUN_HOME = $e2eHome
    $e2eArgs = @("--e2e", "--report", "`"$(Join-Path $Build 'e2e.txt')`"")
    if ($ModelDir) { $e2eArgs += @("--model-dir", "`"$ModelDir`""); Write-Host "  model from $ModelDir" }
    else { Write-Host "  no local model: the e2e run downloads it (2.6 GB)" -ForegroundColor Yellow }
    try { $p = Start-Process $exe -ArgumentList $e2eArgs -Wait -PassThru -NoNewWindow }
    finally { Remove-Item Env:LITTLERUN_HOME }
    if ($p.ExitCode -ne 0) { Fail "end-to-end run failed (logs: $e2eHome\logs)" }
}

# ------------------------------------------------------------------ 8. package
Step "Packaging"
$zipName = "LittleRun-$version-win-x64.zip"
$zipPath = Join-Path $Dist $zipName
$stage = Join-Path $Build "zip"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item $exe $stage
Copy-Item (Join-Path $Root "LICENSE"), (Join-Path $Root "NOTICE") $stage
Copy-Item (Join-Path $PSScriptRoot "README-windows.txt") (Join-Path $stage "README.txt")
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -Force

$iscc = @(
    (Get-Command iscc -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
$setup = $null
if ($NoInstaller) {
    Write-Host "  installer skipped (-NoInstaller)"
} elseif ($iscc) {
    Invoke-Checked "building the installer" {
        & $iscc /Q "/DAppVersion=$version" "/DSourceExe=$exe" "/DRepoRoot=$Root" "/O$Dist" (Join-Path $PSScriptRoot "installer.iss")
    }
    $setup = Join-Path $Dist "LittleRunSetup-$version.exe"
} else {
    Write-Host "  Inno Setup not found: no installer built (winget install JRSoftware.InnoSetup)" -ForegroundColor Yellow
}

$sums = Get-ChildItem $Dist -File | Where-Object { $_.Extension -in ".exe", ".zip" } |
    ForEach-Object { "{0}  {1}" -f (Get-Sha256 $_.FullName), $_.Name }
$sums | Set-Content (Join-Path $Dist "SHA256SUMS.txt") -Encoding ascii

Write-Host ""
Write-Host ("BUILD SUCCEEDED in {0:mm\:ss}" -f ((Get-Date) - $Started)) -ForegroundColor Green
Get-ChildItem $Dist -File | ForEach-Object { Write-Host ("  {0,-36} {1,8:N1} MB" -f $_.Name, ($_.Length / 1MB)) }
