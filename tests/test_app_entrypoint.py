"""Headless release commands communicate success to the calling build tool."""
from pathlib import Path
import sys

import pytest

from recognizer import app, release_selftest


@pytest.mark.parametrize('ok, expected_status', [(True, 0), (False, 1)])
def test_selftest_report_path_gui_flag_and_exit_status(tmp_path, monkeypatch, ok, expected_status):
    report = tmp_path / 'verification.json'
    calls = []

    def run(output, gui=False):
        calls.append((output, gui))
        return {'ok': ok}

    monkeypatch.setattr(release_selftest, 'run', run)
    assert app.main(['--self-test', str(report), '--gui']) == expected_status
    assert calls == [(Path(report), True)]


def test_headless_recognition_requires_an_output_path():
    with pytest.raises(SystemExit) as error:
        app.main(['--recognize', 'image.png'])
    assert error.value.code == 2


def test_headless_recognition_creates_output_parents(tmp_path, monkeypatch):
    from recognizer import pipeline
    result = {'diagram': {'crossings': [], 'edges': []}}
    monkeypatch.setattr(pipeline, 'recognize', lambda image: result)
    output = tmp_path/'nested'/'result.json'
    assert app.main(['--recognize', 'image.png', '--output', str(output)]) == 0
    import json
    assert json.loads(output.read_text()) == result


def test_headless_recognition_missing_input_reports_failure(tmp_path, capsys):
    output = tmp_path/'result.json'
    assert app.main(['--recognize', str(tmp_path/'missing.png'), '-o', str(output)]) == 1
    assert 'missing.png' in capsys.readouterr().err
    assert not output.exists()


def test_frozen_app_can_launch_when_home_log_directory_is_unwritable(tmp_path, monkeypatch):
    from recognizer import ui
    started = []
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(ui, 'main', lambda argv: started.append(argv))
    monkeypatch.setattr(app, 'log_directory', lambda: tmp_path/'Library'/'Logs'/'Knot Studio')
    # A file where the Logs directory should be reproduces a real startup failure.
    (tmp_path/'Library').write_text('not a directory')
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    before = sys.stdout, sys.stderr
    assert app.main([]) == 0
    assert started == [[]]
    assert (sys.stdout, sys.stderr) == before
    assert list(tmp_path.glob('KnotStudio-*.log'))


def test_source_startup_error_does_not_point_to_nonexistent_frozen_log(monkeypatch, capsys):
    from recognizer import ui
    from tkinter import messagebox
    shown = []
    def fail(argv):
        raise RuntimeError('startup failure')
    monkeypatch.delattr(sys, 'frozen', raising=False)
    monkeypatch.setattr(ui, 'main', fail)
    monkeypatch.setattr(messagebox, 'showerror', lambda *args: shown.append(args))
    assert app.main([]) == 1
    assert 'startup failure' in capsys.readouterr().err
    assert shown and 'terminal output' in shown[0][1]
