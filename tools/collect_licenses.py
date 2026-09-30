#!/usr/bin/env python3
"""Collect installed redistribution notices for a native build, without downloads.

Run with the Python environment used to freeze the app. ``collect(output,
 tesseract=..., tessdata=...)`` writes inventory.json and INVENTORY.txt beside
copied upstream notices. Missing required evidence causes a failure *after*
writing the inventory, so a builder can inspect the issues before packaging.
This is a notice collector, not a substitute for corresponding-source duties.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import plistlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGES = {
    "numpy": "https://github.com/numpy/numpy",
    "scipy": "https://github.com/scipy/scipy",
    "scikit-image": "https://github.com/scikit-image/scikit-image",
    "opencv-python-headless": "https://github.com/opencv/opencv-python",
    "networkx": "https://github.com/networkx/networkx",
    "Pillow": "https://github.com/python-pillow/Pillow",
    "imageio": "https://github.com/imageio/imageio",
    "tifffile": "https://github.com/cgohlke/tifffile",
    "lazy-loader": "https://github.com/scientific-python/lazy-loader",
    "packaging": "https://github.com/pypa/packaging",
    "PyInstaller": "https://github.com/pyinstaller/pyinstaller",
    "pyinstaller-hooks-contrib": "https://github.com/pyinstaller/pyinstaller-hooks-contrib",
}
LEGAL_NAME = re.compile(r"(?:licen[cs]e|copying|copyright|notice)(?:[._-].*|$)", re.I)
MODEL_SHA256 = "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"
MODEL_LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
MODEL_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/4.1.0/eng.traineddata"
MODEL_LICENSE_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/LICENSE"
SOURCE_NOTE = (
    "This inventory preserves installed notices and identifies upstream sources. "
    "It does not constitute a source offer or establish that all redistribution "
    "obligations have been met. Distribute complete corresponding source where "
    "required; an upstream URL alone does not replace that obligation."
)


class LicenseCollectionError(RuntimeError):
    """Required local license evidence could not be collected."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]", "_", value)


def _notice_path(path: Path) -> bool:
    # A runtime package can itself be named "licenses" (e.g. packaging).
    # Only a distribution metadata license directory implies legal evidence.
    return bool(LEGAL_NAME.match(path.name)) or any(
        part.endswith(".dist-info") and i + 1 < len(path.parts)
        and path.parts[i + 1].lower() in {"licenses", "licences"}
        for i, part in enumerate(path.parts)
    )


def _otool(path: Path, flag: str) -> str:
    return subprocess.check_output(
        ["/usr/bin/otool", flag, str(path)], text=True, stderr=subprocess.PIPE
    )


def _homebrew_root(path: Path) -> Path | None:
    parts = path.resolve().parts
    if "Cellar" not in parts:
        return None
    idx = parts.index("Cellar")
    return Path(*parts[:idx + 3]) if len(parts) > idx + 2 else None


def _resolve_link(link: str, loader: Path, executable: Path, rpaths: list[str]) -> Path | None:
    def expand(value: str) -> Path:
        value = value.replace("@loader_path", str(loader.parent))
        value = value.replace("@executable_path", str(executable.parent))
        return Path(value)

    if link.startswith("@rpath/"):
        for base in rpaths:
            candidate = expand(base) / link[len("@rpath/"):]
            if candidate.exists():
                return candidate.resolve()
        return None
    candidate = expand(link)
    return candidate.resolve() if candidate.is_absolute() and candidate.exists() else None


def native_dependencies(executable: Path) -> tuple[dict[Path, list[Path]], list[str]]:
    """Resolve actual Mach-O links recursively, grouped by installed Homebrew keg.

    Apple system libraries are provided by macOS and excluded. Returned paths
    are for build use only; the saved inventory converts them to keg-relative
    references. Unresolved or non-Homebrew non-system links are reported.
    """
    executable = executable.resolve()
    queue = [executable]
    visited: set[Path] = set()
    groups: dict[Path, list[Path]] = {}
    issues: list[str] = []
    while queue:
        binary = queue.pop()
        if binary in visited:
            continue
        visited.add(binary)
        root = _homebrew_root(binary)
        if root is None:
            issues.append(f"Native binary {binary.name}: no Homebrew keg provenance.")
        else:
            groups.setdefault(root, []).append(binary)
        try:
            links = _otool(binary, "-L").splitlines()[1:]
            load_commands = _otool(binary, "-l")
        except (OSError, subprocess.CalledProcessError):
            issues.append(f"Cannot inspect native links for {binary.name}.")
            continue
        rpaths = re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path (.*?) \(offset", load_commands)
        for line in links:
            link = line.strip().split(" (", 1)[0]
            if link.startswith(("/usr/lib/", "/System/Library/")):
                continue
            resolved = _resolve_link(link, binary, executable, rpaths)
            if resolved is None:
                issues.append(f"Unresolved native dependency {Path(link).name} of {binary.name}.")
            elif resolved != binary:
                queue.append(resolved)
    return groups, sorted(set(issues))


def collect(output: Path, tesseract: Path | None = None, tessdata: Path | None = None,
            *, strict: bool = True, ocr_root: Path | None = None,
            runtime_notices: Path | None = None) -> dict[str, Any]:
    """Create a portable license tree and inventories; return the inventory.

    ``output`` is a dedicated directory for collected dependency notices.
    Pass original installed Tesseract and English model paths, before relocating
    binaries. Omit Tesseract only for a build that does not bundle OCR.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    components: list[dict[str, Any]] = []
    issues: list[str] = []

    def save(component: dict[str, Any], relative: str, content: bytes,
             source_reference: str) -> None:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        component["notices"].append({"path": relative, "sha256": _sha(content),
                                     "source_reference": source_reference})

    def file(component: dict[str, Any], source: Path, relative: str, reference: str) -> None:
        if not source.is_file():
            issues.append(f"{component['name']}: required notice missing ({reference}).")
            return
        content = source.read_bytes()
        if not content.strip():
            issues.append(f"{component['name']}: required notice is empty ({reference}).")
            return
        save(component, relative, content, reference)

    def component(name: str, version: str, kind: str, url: str) -> dict[str, Any]:
        result: dict[str, Any] = {"name": name, "version": version, "kind": kind,
                                  "source_url": url, "notices": []}
        components.append(result)
        return result

    for name, url in PACKAGES.items():
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            issues.append(f"Required build/runtime distribution is missing: {name}.")
            continue
        entry = component(name, dist.version, "python-distribution", url)
        entry["license_expression"] = dist.metadata.get("License-Expression")
        prefix = f"python/{_safe(name)}/{_safe(dist.version)}"
        for relative in sorted(dist.files or [], key=str):
            rel = Path(str(relative))
            if rel.is_absolute() or ".." in rel.parts or not _notice_path(rel):
                continue
            source = Path(dist.locate_file(relative))
            if source.is_file():
                file(entry, source, f"{prefix}/{rel.as_posix()}", rel.as_posix())
        if not entry["notices"]:
            issues.append(f"{name} {dist.version}: no installed LICENSE/COPYING/NOTICE files.")
        if name == "PyInstaller":
            if not any(b"Bootloader Exception" in (output / n["path"]).read_bytes()
                       for n in entry["notices"]):
                issues.append("PyInstaller: bootloader exception text was not found.")
            entry["scope"] = "Unmodified embedded bootloader, loader, and runtime hooks; retain the bootloader exception."

    base = Path(sys.base_prefix)
    if sys.platform != 'darwin':
        try:
            from tools.portable_licenses import collect_runtime
        except ModuleNotFoundError:
            from portable_licenses import collect_runtime
        collect_runtime(base, runtime_notices, component, file, issues)
    py_version = ".".join(map(str, sys.version_info[:3]))
    minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    if sys.platform == 'darwin':
        py = component("Python", py_version, "runtime", f"https://www.python.org/downloads/release/python-{py_version.replace('.', '')}/")
        file(py, base / "lib" / minor / "LICENSE.txt", "runtime/Python/LICENSE.txt", f"lib/{minor}/LICENSE.txt")
    # The CPython core license alone omits notices for bundled third-party code.
    doc_rel = "Resources/English.lproj/Documentation/_sources/license.rst.txt"
    if sys.platform == 'darwin':
        file(py, base / doc_rel, "runtime/Python/bundled-library-notices.rst.txt", doc_rel)
    for name in (("Tcl", "Tk") if sys.platform == 'darwin' else ()):
        framework = base / "Frameworks" / f"{name}.framework"
        versions = framework / "Versions"
        candidates = sorted(p for p in versions.glob("*") if p.name != "Current" and p.is_dir())
        if len(candidates) != 1:
            issues.append(f"{name}: expected one bundled framework version with local notices.")
            continue
        version_dir = candidates[0]
        info = version_dir / "Resources" / "Info.plist"
        version = version_dir.name
        if info.is_file():
            meta = plistlib.loads(info.read_bytes())
            version = meta.get("CFBundleShortVersionString") or meta.get("CFBundleVersion") or version
        entry = component(name, str(version), "runtime-framework", "https://www.tcl-lang.org/")
        license_file = version_dir / "Resources" / "license.terms"
        file(entry, license_file, f"runtime/{name}/license.terms", license_file.relative_to(base).as_posix())

    if tesseract is not None:
        tesseract = Path(tesseract)
        if not tesseract.is_file():
            issues.append("Tesseract executable is missing.")
        elif sys.platform != 'darwin':
            try:
                from tools.portable_licenses import collect_native
            except ModuleNotFoundError:
                from portable_licenses import collect_native
            collect_native(tesseract, ocr_root, component, file, issues)
        else:
            groups, native_issues = native_dependencies(tesseract)
            issues.extend(native_issues)
            for keg, binaries in sorted(groups.items(), key=lambda pair: str(pair[0])):
                name, version = keg.parent.name, keg.name
                formula_path = keg / ".brew" / f"{name}.rb"
                formula = formula_path.read_text() if formula_path.is_file() else ""
                source = re.search(r'^\s*url\s+"([^"]+)"', formula, re.M)
                homepage = re.search(r'^\s*homepage\s+"([^"]+)"', formula, re.M)
                declared = re.search(r'^\s*license\s+(.+)$', formula, re.M)
                entry = component(name, version, "native-homebrew", source.group(1) if source else f"https://formulae.brew.sh/formula/{name}")
                entry["homepage"] = homepage.group(1) if homepage else None
                entry["formula_license_declaration"] = declared.group(1).strip() if declared else None
                entry["binaries"] = sorted(b.relative_to(keg).as_posix() for b in binaries)
                prefix = f"native/{_safe(name)}/{_safe(version)}"
                for candidate in sorted(keg.rglob("*")):
                    rel = candidate.relative_to(keg)
                    if candidate.is_file() and _notice_path(rel) and candidate.suffix.lower() not in {".png", ".jpg"}:
                        file(entry, candidate, f"{prefix}/{rel.as_posix()}", rel.as_posix())
                if not entry["notices"] and name == "leptonica":
                    header = keg / "include/leptonica/allheaders.h"
                    text = header.read_bytes() if header.is_file() else b""
                    end = text.find(b"*/")
                    notice = text[:end + 2] + b"\n" if end >= 0 else b""
                    if b"Redistribution and use" in notice and b"THIS SOFTWARE" in notice:
                        save(entry, f"{prefix}/LICENSE-from-installed-header.txt", notice,
                             "include/leptonica/allheaders.h (initial license comment)")
                        entry["notice_origin"] = "Full license comment retained verbatim from installed upstream header; bottle had no separate license file."
                if not entry["notices"]:
                    issues.append(f"{name} {version}: no installed redistribution notice found.")
                if not formula:
                    issues.append(f"{name} {version}: installed source/version formula metadata missing.")
            if tessdata is None:
                tessdata = tesseract.resolve().parent.parent / "share/tessdata"
    if tessdata is not None:
        model = Path(tessdata)
        if model.is_dir():
            model = model / "eng.traineddata"
        entry = component("tessdata_fast English model", "4.1.0", "ocr-language-data", MODEL_URL)
        if not model.is_file():
            issues.append("English OCR model is missing.")
        else:
            actual = _sha(model.read_bytes())
            entry["model"] = {"file": "eng.traineddata", "sha256": actual}
            if actual != MODEL_SHA256:
                issues.append("English model differs from the verified tessdata_fast 4.1.0 release; establish its source and license before distribution.")
        vendored = Path(__file__).resolve().parents[1] / "licenses/upstream/tessdata-fast-4.1.0-LICENSE"
        file(entry, vendored, "models/tessdata-fast-4.1.0/LICENSE", MODEL_LICENSE_URL)
        if vendored.is_file() and _sha(vendored.read_bytes()) != MODEL_LICENSE_SHA256:
            issues.append("English model's vendored license differs from the verified upstream license text.")
        entry["license_expression"] = "Apache-2.0"
        entry["attribution"] = "Tesseract OCR project contributors; tessdata_fast English LSTM language model."

    inventory = {"schema_version": 1, "scope": "Installed notices for the build interpreter, selected Python distributions, and recursively linked OCR binaries.",
                 "source_distribution_note": SOURCE_NOTE, "components": components,
                 "issues": sorted(set(issues)), "complete": not issues}
    encoded = json.dumps(inventory, indent=2, ensure_ascii=False) + "\n"
    # Paths in provenance refer to package/keg members; never the build user's home.
    if "/Users/" in encoded or re.search(r"[A-Za-z]:\\Users\\", encoded):
        raise LicenseCollectionError("Generated inventory unexpectedly contains a personal absolute path.")
    (output / "inventory.json").write_text(encoded)
    lines = ["KnotStudio bundled dependency notices", "", SOURCE_NOTE, ""]
    for entry in components:
        lines += [f"{entry['name']} {entry['version']}", f"  Source: {entry['source_url']}"]
        lines += [f"  Notice: {item['path']}" for item in entry["notices"]]
    if issues:
        lines += ["", "UNRESOLVED REQUIRED EVIDENCE:"] + [f"- {issue}" for issue in sorted(set(issues))]
    else:
        lines += ["", "All required local notice files were collected. This is not a legal compliance certification."]
    (output / "INVENTORY.txt").write_text("\n".join(lines) + "\n")
    if issues and strict:
        raise LicenseCollectionError("Missing license evidence; see inventory.json: " + "; ".join(sorted(set(issues))))
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tesseract", type=Path)
    parser.add_argument("--tessdata", type=Path)
    parser.add_argument("--ocr-root", type=Path)
    parser.add_argument("--runtime-notices", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true", help="Write flagged evidence for inspection; do not publish it as a complete bundle.")
    args = parser.parse_args()
    try:
        result = collect(args.output, tesseract=args.tesseract, tessdata=args.tessdata,
                         strict=not args.allow_incomplete, ocr_root=args.ocr_root,
                         runtime_notices=args.runtime_notices)
    except LicenseCollectionError as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Collected notices for {len(result['components'])} components; {len(result['issues'])} unresolved issues.")


if __name__ == "__main__":
    main()
