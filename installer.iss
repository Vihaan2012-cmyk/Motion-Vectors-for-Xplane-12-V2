; Motion Vectors for X-Plane 12 - installer
;
; Two things get installed and they go to different places:
;
;   the plugin  ->  <X-Plane>\Resources\plugins\MotionVectors\64\win.xpl
;   the layer   ->  <X-Plane>\MotionVectors\  plus a registry key
;
; The Vulkan loader finds an implicit layer through a registry value naming its
; manifest, so a plain file copy is not enough - the key is what makes it load.
; It is removed again on uninstall.

#define AppName    "Motion Vectors for X-Plane 12"
; Passed in by build.ps1 as /DAppVersion=..., which reads the VERSION file.
; The literal here is only a fallback for building by hand, and it is
; deliberately not a plausible version - a stale number that LOOKS right is
; worse than one that cannot be mistaken for a real build.
#ifndef AppVersion
  #define AppVersion "0.0.0-handbuilt"
#endif
#define AppPub     "Vihaan2012-cmyk"

[Setup]
AppId={{7A2C1E44-9E1B-4C6E-9E2E-4B7A0C2D51A1}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPub}
DefaultDirName={code:DefaultXPlaneDir}
DirExistsWarning=no
AppendDefaultDirName=no
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=MotionVectors-{#AppVersion}-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
UninstallDisplayName={#AppName}
WizardStyle=modern
DisableWelcomePage=no

[Messages]
WelcomeLabel2=This installs motion vectors for X-Plane 12.%n%nStart the sim from the shortcut this creates - it enables the layer for X-Plane only, so no other Vulkan application is affected.%n%nChoose your X-Plane 12 folder on the next page - the one containing X-Plane.exe.

[Files]
Source: "build\MotionVectors\MotionVectors.xpl";        DestDir: "{app}\Resources\plugins\MotionVectors\64"; DestName: "win.xpl"; Flags: ignoreversion
Source: "build\MotionVectors\vklayer\VkLayer_mv.dll";   DestDir: "{app}\MotionVectors";                      Flags: ignoreversion
Source: "build\MotionVectors\vklayer\VkLayer_mv.json";  DestDir: "{app}\MotionVectors";                      Flags: ignoreversion
; The Qt launcher and its runtime. recursesubdirs picks up the platform,
; imageformat and tls plugin folders windeployqt produced - Qt will not start
; without platforms\qwindows.dll, and a missing plugin folder fails at run time
; rather than at install time, which is the worse place to find out.
; This directory carries BOTH applications - the public launcher and the debug
; console - because they share one Qt runtime. Shipping them separately would
; mean two ~40 MB copies of Qt for two executables that differ by a few hundred
; kilobytes.
Source: "build\MotionVectors\qtlauncher\*"; DestDir: "{app}\MotionVectors\launcher"; Flags: ignoreversion recursesubdirs createallsubdirs
; Kept as a fallback: no dependencies, works even if the Qt runtime is broken.
Source: "build\MotionVectors\MotionVectorsLauncher.exe"; DestDir: "{app}\MotionVectors";                    Flags: ignoreversion
Source: "lua\MotionVectors.lua";           DestDir: "{app}\Resources\plugins\FlyWithLua\Scripts"; Flags: ignoreversion; Check: FlyWithLuaPresent
Source: "README.md";                      DestDir: "{app}\MotionVectors";                      Flags: ignoreversion

[Icons]
; The sim must be started through this. It sets VK_LAYER_PATH and
; VK_LOADER_LAYERS_ENABLE for that one process, which is what makes the layer
; EXPLICIT - loaded only here, never in any other Vulkan application.
Name: "{autoprograms}\X-Plane 12 with Motion Vectors"; Filename: "{app}\MotionVectors\launcher\MotionVectors.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\X-Plane 12 with Motion Vectors";  Filename: "{app}\MotionVectors\launcher\MotionVectors.exe"; WorkingDir: "{app}"; Tasks: desktopicon
; The debug console. Separate shortcut because it is used at a different TIME -
; the launcher's job ends when the sim starts, and the console's begins there.
; It edits the live control file, so every change it makes reaches a sim that is
; already flying and nothing it does requires a restart.
Name: "{autoprograms}\Motion Vectors Debug Console"; Filename: "{app}\MotionVectors\launcher\MotionVectorsDebug.exe"; WorkingDir: "{app}"; Tasks: debugconsole
Name: "{autodesktop}\Motion Vectors Debug Console";  Filename: "{app}\MotionVectors\launcher\MotionVectorsDebug.exe"; WorkingDir: "{app}"; Tasks: debugconsole and desktopicon

[Tasks]
Name: "desktopicon";   Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "debugconsole";  Description: "Add the Debug Console (tune and diagnose while the sim runs)"; GroupDescription: "Shortcuts:"

[Run]
Filename: "{app}\MotionVectors\launcher\MotionVectors.exe"; Description: "Start X-Plane 12 with motion vectors now"; Flags: nowait postinstall skipifsilent

[Code]
// Guess X-Plane's location so the common case needs no typing, and verify
// whatever is chosen actually is an X-Plane folder rather than failing later
// with files copied somewhere harmless.
// The Lua panel is optional: it needs FlyWithLua, which many installs do not
// have. Copying a script into a folder that does not exist would fail the
// install for a component nothing else depends on.
function FlyWithLuaPresent: Boolean;
begin
  Result := DirExists(WizardDirValue + '\Resources\plugins\FlyWithLua\Scripts');
end;

function DefaultXPlaneDir(Param: String): String;
var
  Candidates: array[0..3] of String;
  I: Integer;
begin
  Candidates[0] := 'D:\Steam Games\steamapps\common\X-Plane 12';
  Candidates[1] := ExpandConstant('{sd}\Steam\steamapps\common\X-Plane 12');
  Candidates[2] := ExpandConstant('{pf}\X-Plane 12');
  Candidates[3] := ExpandConstant('{sd}\X-Plane 12');
  for I := 0 to 3 do
    if FileExists(Candidates[I] + '\X-Plane.exe') then
    begin
      Result := Candidates[I];
      Exit;
    end;
  Result := ExpandConstant('{sd}\X-Plane 12');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    if not FileExists(WizardDirValue + '\X-Plane.exe') then
    begin
      // No line may BEGIN with '#': Inno's preprocessor reads a leading hash as
      // a directive, so a wrapped string continuing with #13#10 fails to
      // compile with "Unknown preprocessor directive".
      MsgBox('X-Plane.exe was not found in:' + Chr(13) + Chr(10) + Chr(13) + Chr(10)
             + WizardDirValue + Chr(13) + Chr(10) + Chr(13) + Chr(10)
             + 'Pick the folder that contains X-Plane.exe.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;
