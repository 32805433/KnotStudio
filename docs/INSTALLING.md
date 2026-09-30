# Install and open Knot Studio

Choose a package from [Releases](https://github.com/32805433/KnotStudio/releases)
for your operating system. If a platform is absent from a release, no packaged
app for that platform is available in that version. Source ZIPs are for developers.
Recognition runs on your computer and works offline.

## Requirements

| Package | Target system |
| --- | --- |
| macOS ARM64 | Apple silicon (M1 or later), macOS 15.7.5 or later |
| Windows x86-64 | Windows 11; Windows 10 22H2 is the compatibility target for the same download |
| Linux x86-64 | Ubuntu 24.04 LTS or later, with X11 or XWayland |

Intel Macs and Windows/Linux ARM devices do not have native packages. Ubuntu
24.04 is the Linux baseline; other distributions are not automatically supported.
The release notes identify platforms that have been tested on a desktop.

## macOS

1. Extract the **macOS-arm64.zip** download.
2. Drag **Knot Studio.app** into **Applications**.
3. Open the app and choose **Open…** to load a picture.

### If macOS blocks opening the app

The default build is **not notarized by Apple**. If macOS says it cannot check
Knot Studio for malicious software or cannot verify its developer, follow these
steps only if you trust the source of your copy:

1. Try opening the app once.
2. Open **System Settings → Privacy & Security**.
3. Find the Knot Studio message and click **Open Anyway**, then confirm.

This creates an exception for this app. Do not disable system-wide protections.
See [Apple's instructions](https://support.apple.com/102445).
A managed Mac may restrict this option. If the warning instead says the app
**contains malware**, **will damage your computer**, or **is damaged**, report the
exact message through [Issues](https://github.com/32805433/KnotStudio/issues).

## Windows

1. Download the **Windows-x86_64-Setup.exe** installer.
2. Run it and follow the instructions. Installation is for your user account;
   administrator access is not normally required.
3. Open **Knot Studio** from the Start menu.

Alternatively, extract the **Windows-x86_64.zip** portable download and open
**KnotStudio.exe** inside its folder. Keep the whole folder together: the
executable needs the accompanying files. No Python or Tesseract installation is
required. Uninstall an installed copy through **Settings → Apps**.

An unsigned download may show an unknown-publisher warning. If SmartScreen says
**Windows protected your PC**, use **More info → Run anyway** only when you trust
the download. If Windows reports detected malware, report the exact message
instead of disabling protection.

## Linux

1. Extract the **Linux-x86_64.tar.gz** archive into a folder you can write to.
2. Open **KnotStudio** inside the extracted folder. Keep its accompanying files.
3. If your file manager asks how to handle it, choose to run it as a program.

The archive preserves executable permissions. If extraction software removes
them, enable **Allow executing file as program** in the file's properties.
The package includes Python, Tk and OCR. It needs a desktop session and the
standard Ubuntu desktop libraries; a headless server is not the target.
Wayland sessions need XWayland for the Tk interface.

## Help and updates

Choose **Help → User guide** in the app, or read the [user guide](USER_GUIDE.md).
Get updates from [Releases](https://github.com/32805433/KnotStudio/releases).
Replace a portable folder as a whole when upgrading; save diagram files outside
that folder. Report problems through [Issues](https://github.com/32805433/KnotStudio/issues),
including the app version, operating system, and exact error message.
