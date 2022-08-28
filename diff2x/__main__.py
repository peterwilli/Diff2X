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
    upscale_parser = sps.add_parser('upscale', help='Upscale image with Diff2X')
    upscale_parser.add_argument(
        'image_file',
        type=str,
        help='Image file to upscale, default is stdin.',
        default=sys.stdin,
    )
    upscale_parser.add_argument(
        'output_file',
        type=str,
        help='Output file, default is stdout',
        default=sys.stdout,
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

def _upscale(input_file, output_file):
    from .diff2x import upscale_image
    def on_image_update(image):
        img_out = open(output_file, 'wb')
        image.save(img_out)
        img_out.flush()
        img_out.close()
    result = upscale_image(input_file, on_image_update = on_image_update)
    result.wait()
    result.final_image.save(output_file)

def run_cli(args):
    if args.action == "serve":
        serve(args.config_file)
    elif args.action == "serve_test":
        serve_test(args.config_file)
    elif args.action == "upscale":
        _upscale(args.image_file, args.output_file)

if __name__ == '__main__':
    main()