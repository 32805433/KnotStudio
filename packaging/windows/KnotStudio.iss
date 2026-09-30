; Build with tools/build_desktop.py --installer on Windows.
#ifndef AppVersion
  #error AppVersion must come from pyproject.toml through build_desktop.py
#endif
#ifndef AppSource
  #error AppSource must point to the verified portable application
#endif
[Setup]
AppId={{FD93D574-5077-433D-884E-8C0FC9F0C589}
AppName=Knot Studio
AppVersion={#AppVersion}
AppPublisher=Knot Studio contributors
AppPublisherURL=https://github.com/32805433/KnotStudio
DefaultDirName={localappdata}\Programs\Knot Studio
DefaultGroupName=Knot Studio
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0
OutputDir={#OutputDirectory}
OutputBaseFilename=KnotStudio-{#AppVersion}-Windows-x86_64-Setup
SetupIconFile=..\..\.build-assets\KnotStudio.ico
UninstallDisplayIcon={app}\KnotStudio.exe
LicenseFile=..\..\LICENSE
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked
[Files]
Source: "{#AppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\Knot Studio"; Filename: "{app}\KnotStudio.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Knot Studio"; Filename: "{app}\KnotStudio.exe"; WorkingDir: "{app}"; Tasks: desktopicon
[Run]
Filename: "{app}\KnotStudio.exe"; Description: "Open Knot Studio"; Flags: nowait postinstall skipifsilent
