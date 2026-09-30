#!/usr/bin/env python3
"""Compatibility entry point for the shared native macOS desktop builder."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_desktop import main

if __name__ == '__main__':
    if sys.platform != 'darwin':
        raise SystemExit('Build the macOS application on macOS; use build_desktop.py on Windows/Linux.')
    main()
