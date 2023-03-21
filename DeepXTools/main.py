import argparse
from pathlib import Path
from core.lib import argparse as lib_argparse


# from core.lib import torch as lib_torch

# print( lib_torch.get_avail_gpu_devices() )

# import torch
# import code
# code.interact(local=dict(globals(), **locals()))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    run_parser = subparsers.add_parser("run", help="Run the application.")
    run_subparsers = run_parser.add_subparsers()

    mask_editor_parser = run_subparsers.add_parser("MaskEditor", help="Run Mask Editor.")
    mask_editor_parser.add_argument('--workspace-dir', required=True, action=lib_argparse.FixPathAction, help="Workspace directory.")
    def mask_editor_run(args):
        from MaskEditor import QxMaskEditorApp
        app = QxMaskEditorApp(workspace_path=Path(args.workspace_dir), settings_path=Path(args.workspace_dir) / 'MaskEditor.ui')
        app.exec()
        app.dispose()
    mask_editor_parser.set_defaults(func=mask_editor_run)

    deep_roto_parser = run_subparsers.add_parser("DeepRoto", help="Run Deep Roto.")
    deep_roto_parser.add_argument('--workspace-dir', required=True, action=lib_argparse.FixPathAction, help="Workspace directory.")
    def deep_roto_run(args):
        from DeepRoto import MxDeepRoto, QxDeepRotoApp
        deep_roto = MxDeepRoto(workspace_path=Path(args.workspace_dir))
        
        app = QxDeepRotoApp(deep_roto=deep_roto, settings_path=Path(args.workspace_dir) / 'DeepRoto.ui')
        app.exec()
        app.dispose()

        # app = QxDeepRotoApp(deep_roto=deep_roto, settings_path=Path(args.workspace_dir) / 'DeepRoto.ui')
        # app.exec()
        # app.dispose()
        
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
