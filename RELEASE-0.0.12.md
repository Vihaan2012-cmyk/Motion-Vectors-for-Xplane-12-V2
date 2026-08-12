# Motion Vectors 0.0.12 — the build produces what the installer ships

No change to the vectors. This release fixes two ways the build could ship
something nobody had built, and adds an installer to every version.

## The installer was shipping a launcher nobody built

`src\qtlauncher\main.cpp` existed. `installer.iss` shipped `build\qtlauncher\*`
unconditionally. **Nothing in between built it.** So every installer since that
folder was first populated by hand carried a stale Qt launcher, compiled from a
source that had not been touched since.

`build.ps1` now compiles it and runs `windeployqt`, clearing the output
directory first — `windeployqt` only *adds* files, so a runtime from an older Qt
would otherwise be shipped beside the current one.

## The version disagreed with itself

It lived in `installer.iss` alone, which said **0.0.08** while the tag said
0.0.11 and the plugin said nothing at all. A version that disagrees with itself
across three files is how a user ends up reporting a bug against a build nobody
can identify.

`VERSION` is now the one file. The compiler takes it as `-DMV_VERSION`, the
installer takes it as `/DAppVersion`, and the `.iss` fallback is
`0.0.0-handbuilt` rather than a plausible number — a stale version that *looks*
right is worse than one that cannot be mistaken for a real build.

## Also

- `build.ps1 -Installer` builds the setup; every release gets one from here on.
- `installer.iss` had its Qt comment block and its fallback comment pasted twice.
- The Qt launcher needs QtNetwork, which was never on the link line.

Both faults are the same principle: **if the build does not produce it, the
installer must not ship it.**
