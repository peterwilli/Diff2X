import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        epilog=f'Upscale images',
        prog='python -m diff2x',
    )
    sps = parser.add_subparsers(dest='action', required=True)
    serve_parser = sps.add_parser('serve', help='Host diff2x as service')
    serve_test_parser = sps.add_parser('serve_test', help='See if server is running as intended')
    for current_parser in [serve_parser, serve_test_parser]:
        current_parser.add_argument(
            'config_file',
            metavar='YAML_CONFIG_FILE',
            nargs='?',
            type=argparse.FileType('r'),
            help='The YAML config file to use, default is stdin.',
            default=sys.stdin,
        )
    args = parser.parse_args()
    run_cli(args)

def serve_test(cfg):
    from jina import Document
    import time
    with _serve(cfg) as f:
        input = Document(uri="/app/diff2X/misc/thingy-test.png")
        upscale_doc = f.post(on = '/upscale', inputs = input, on_done = print)
        while True:
            progress_doc = f.post(on = '/upscale_progress', inputs = input)
            time.sleep(1)

def serve(cfg):
    with _serve(cfg) as f:
        f.block()

def _serve(cfg):
    from .executor import Diff2Executor, ProgressExecutor
    from jina import Flow
    return Flow.load_config(cfg)

def run_cli(args):
    if args.action == "serve":
        serve(args.config_file)
    elif args.action == "serve_test":
        serve_test(args.config_file)

if __name__ == '__main__':
    main()