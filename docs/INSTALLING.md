# Install and open Knot Studio

Download [Knot Studio for Mac](https://github.com/32805433/KnotStudio/releases/download/v0.1.0/KnotStudio-0.1.0-macOS-arm64.zip)
from the [Releases page](https://github.com/32805433/KnotStudio/releases).

## Requirements

Version 0.1.0 requires an **Apple silicon Mac (M1 or later)** running
**macOS 15.7.5 or later**. Check **Apple menu → About This Mac** if unsure.
The packaged app does not support Intel Macs, Windows, or Linux.

Recognition runs on your Mac and works offline.

## Installation

1. Locate **KnotStudio-0.1.0-macOS-arm64.zip**, the macOS app archive.
   The separate source-code archive is for developers.
2. Double-click the ZIP to extract it.
3. Drag **Knot Studio.app** into **Applications**.
4. Open the app and choose **Open…** to load a picture. You can use your own image
   or save one from the [example pictures](EXAMPLES.md).

## If macOS blocks opening the app

Version 0.1.0 is **not notarized by Apple**. If macOS says it cannot check Knot
Studio for malicious software or cannot verify its developer, follow the steps
below only if you trust the source of your copy.

1. Try opening **Knot Studio.app** once.
2. Open **System Settings → Privacy & Security**.
3. Find the message about Knot Studio and click **Open Anyway**, then confirm
   opening it. macOS may ask for your login password.

This creates an exception for this app. Do not disable system-wide security
protections. For details and the risks of opening an unnotarized app, see
[Apple's opening instructions](https://support.apple.com/102445).
On a managed Mac, your administrator may restrict this option.

If the warning instead says the app **contains malware**, **will damage your
computer**, or **is damaged**, report the exact message through
[Issues](https://github.com/32805433/KnotStudio/issues) before trying to open it.

## Help and updates

- Read the [user guide](USER_GUIDE.md) or choose **Help → User guide** in the app.
- Get future versions from [Releases](https://github.com/32805433/KnotStudio/releases).
- Report problems through [Issues](https://github.com/32805433/KnotStudio/issues),
  including your macOS version and the error message.

Reconstructed diagrams can contain mistakes. Compare the output with the input
before relying on its crossings, orientations, or PD code.
