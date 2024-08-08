from ollama_benchmark.client import OllamaClient


def make_args(subparsers):
    parser = subparsers.add_parser(
        "unload_models",
        help="Unload one or several models from memory",
    )
    parser.add_argument('models', nargs='*')


def main(args):
    client = OllamaClient(
        host=args.host,
        timeout=args.timeout,
    )
    if not args.models:
        client.unload_all()
    else:
        for model in args.models:
            client.unload(model)
