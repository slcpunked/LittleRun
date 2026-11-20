@echo off
rem Builds the LittleRun Windows app into dist\.
rem
rem   build              tests, kernel, payload, LittleRun.exe, zip (and installer with Inno Setup)
rem   build -E2E         also install the runtime and generate with the built exe
rem   build -SkipTests   faster; the exe is still self-tested
rem   build -NoKernel    without Visual Studio C++ Build Tools (slower CPU path in the app)
rem   build -Clean       start from scratch
rem
rem The work is done by windows\build.ps1.
setlocal
cd /d "%~dp0"

set "PS=powershell"
where pwsh >nul 2>nul && set "PS=pwsh"

%PS% -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows\build.ps1" %*
set "RC=%ERRORLEVEL%"

rem Keep the window open when started by double-click from Explorer.
echo %CMDCMDLINE% | find /i "%~nx0" >nul && pause
endlocal & exit /b %RC%
