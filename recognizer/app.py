"""Desktop release entry point, including an offline bundle verification mode."""
import argparse
from pathlib import Path
import sys
import traceback


def _open_log():
    """Logging must not prevent a GUI launch on a restricted home directory."""
    try:
        directory = Path.home()/'Library'/'Logs'/'Knot Studio'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/'app.log'
        if path.exists() and path.stat().st_size > 2_000_000:
            path.replace(directory/'app.previous.log')
        return path.open('a', encoding='utf-8', buffering=1), path
    except OSError:
        try:
            import tempfile
            log = tempfile.NamedTemporaryFile(mode='a', encoding='utf-8',
                                              prefix='KnotStudio-', suffix='.log',
                                              delete=False, buffering=1)
            return log, Path(log.name)
        except OSError:
            return None, None


def main(argv=None):
    parser = argparse.ArgumentParser(description='Knot Studio')
    parser.add_argument('input', nargs='?', help='Image or editable diagram JSON')
    parser.add_argument('--self-test', metavar='REPORT', help='Write offline bundle checks as JSON')
    parser.add_argument('--gui', action='store_true', help='Also test Tk in self-test mode')
    parser.add_argument('--recognize', metavar='IMAGE', help='Recognize without opening the editor')
    parser.add_argument('--output', '-o', metavar='JSON', help='Output for --recognize')
    args = parser.parse_args(argv)
    if args.self_test:
        from .release_selftest import run
        result = run(Path(args.self_test), gui=args.gui)
        return 0 if result.get('ok') else 1
    if args.recognize:
        import json
        from .pipeline import recognize
        if not args.output:
            parser.error('--recognize requires --output')
        try:
            result = recognize(args.recognize)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        except (OSError, ValueError) as error:
            print(f'Knot Studio: {error}', file=sys.stderr)
            return 1
        return 0 if result.get('diagram') is not None else 1
    log = log_path = None
    original_stdout, original_stderr = sys.stdout, sys.stderr
    if getattr(sys, 'frozen', False):
        log, log_path = _open_log()
        if log is not None:
            sys.stdout = sys.stderr = log
    try:
        from .ui import main as gui_main
        gui_main([args.input] if args.input else [])
        return 0
    except Exception:
        traceback.print_exc()
        try:
            from tkinter import messagebox
            messagebox.showerror('Knot Studio could not start',
                                 (f'Please consult {log_path} for details.' if log_path else
                                  'The error details are in the terminal output.' if sys.stderr else
                                  'No diagnostic log could be written. Please report how you started the app.'))
        except Exception:
            pass
        return 1
    finally:
        if log is not None:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            log.close()


if __name__ == '__main__':
    raise SystemExit(main())
