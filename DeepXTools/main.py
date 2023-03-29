import argparse
from pathlib import Path

from core.lib import argparse as lib_argparse


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    run_parser = subparsers.add_parser("run", help="Run the application.")
    run_subparsers = run_parser.add_subparsers()

    mask_editor_parser = run_subparsers.add_parser("MaskEditor", help="Run Mask Editor.")
    mask_editor_parser.add_argument('--ui-data-dir', required=True, action=lib_argparse.FixPathAction, help="UI data directory.")
    def mask_editor_run(args):
        from MaskEditor import QxMaskEditorApp
        app = QxMaskEditorApp(settings_path=Path(args.ui_data_dir) / 'MaskEditor.ui')
        app.exec()
        app.dispose()
    mask_editor_parser.set_defaults(func=mask_editor_run)

    deep_roto_parser = run_subparsers.add_parser("DeepRoto", help="Run Deep Roto.")
    deep_roto_parser.add_argument('--ui-data-dir', required=True, action=lib_argparse.FixPathAction, help="UI data directory.")
    deep_roto_parser.add_argument('--open-path', required=False, action=lib_argparse.FixPathAction, help="Open .dxr project path.")
    def deep_roto_run(args):
        from DeepRoto import MxDeepRoto, QxDeepRotoApp
        deep_roto = MxDeepRoto(open_path=Path(args.open_path) if args.open_path is not None else None)

        app = QxDeepRotoApp(deep_roto=deep_roto, settings_path=Path(args.ui_data_dir) / 'DeepRoto.ui')
        app.exec()
        app.dispose()

        deep_roto.dispose()

    deep_roto_parser.set_defaults(func=deep_roto_run)

    def bad_args(args):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()


# import code
# code.interact(local=dict(globals(), **locals()))
