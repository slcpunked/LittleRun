; LittleRun installer (Inno Setup 6). build.ps1 passes AppVersion, SourceExe and RepoRoot.
; Per-user install like Ollama: no admin rights, into %LOCALAPPDATA%\Programs\LittleRun.
; The runtime, models and chats live in %LOCALAPPDATA%\LittleRun and are kept on uninstall.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6B7C2F3E-1D4A-4E8B-9C21-5A3F0E7D9B64}
AppName=LittleRun
AppVersion={#AppVersion}
AppPublisher=slcpunked
AppPublisherURL=https://github.com/slcpunked/LittleRun
DefaultDirName={localappdata}\Programs\LittleRun
DefaultGroupName=LittleRun
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputBaseFilename=LittleRunSetup-{#AppVersion}
SetupIconFile={#RepoRoot}\windows\LittleRun.Desktop\app.ico
UninstallDisplayIcon={app}\LittleRun.exe
UninstallDisplayName=LittleRun
LicenseFile={#RepoRoot}\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked
Name: "autostart"; Description: "Start LittleRun in the tray when I sign in"; Flags: unchecked

[Files]
Source: "{#SourceExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\windows\README-windows.txt"; DestDir: "{app}"; DestName: "README.txt"; Flags: ignoreversion

[Icons]
Name: "{group}\LittleRun"; Filename: "{app}\LittleRun.exe"
Name: "{userdesktop}\LittleRun"; Filename: "{app}\LittleRun.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "LittleRun"; \
  ValueData: """{app}\LittleRun.exe"" --minimized"; Tasks: autostart; Flags: uninsdeletevalue

[Run]
Filename: "{app}\LittleRun.exe"; Description: "Start LittleRun"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\taskkill.exe"; Parameters: "/IM LittleRun.exe /F"; Flags: runhidden; RunOnceId: "StopLittleRun"
