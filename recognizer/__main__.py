"""Command-line recognition and the native Python diagram editor."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',nargs='?',type=Path)
    parser.add_argument('--output','-o',type=Path,help='Write recognition result as JSON.')
    parser.add_argument('--png',type=Path,help='Export the reconstructed diagram as PNG.')
    parser.add_argument('--svg',type=Path,help='Export the reconstructed diagram as SVG.')
    parser.add_argument('--gui',action='store_true',help='Launch the Tk desktop editor.')
    parser.add_argument('--keep-labels',action='store_true',help='Keep source labels unchanged (the default; retained for compatibility).')
    parser.add_argument('--board-photo',choices=('auto','off','board','light','dark'),default='auto',
                        help='Classical board-photo handling: automatic, off, automatic polarity, whiteboard, or blackboard.')
    parser.add_argument('--experimental-ink',action='store_true',
                        help='Try the small learned photo-ink filter after classical recognition fails; inspect any recovered diagram.')
    parser.add_argument('--time-budget',type=float,default=10.,
                        help='Shared recognition budget in seconds (default 10).')
    args=parser.parse_args()
    if args.gui or args.image is None:
        from .ui import main as gui_main
        gui_main([str(args.image)] if args.image else [])
        return
    from .pipeline import recognize
    options={'board_photo':args.board_photo, 'time_budget':args.time_budget}
    if args.experimental_ink:
        options['learned_ink']=True
    if args.keep_labels:
        options['remove_labels']=False
    result=recognize(args.image,options=options)
    text=json.dumps(result,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+'\n')
    else:
        print(text)
    if result['diagram'] is not None:
        if args.png:
            from .render import render_png
            render_png(result['diagram'],args.png)
        if args.svg:
            from .render import save_svg
            save_svg(result['diagram'],args.svg)


if __name__=='__main__':main()
