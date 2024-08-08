from ollama_benchmark import errors
from ollama_benchmark.client import OllamaClient


def make_args(subparsers):
    parser = subparsers.add_parser(
        "pull_models",
        help="Pull one or several models",
    )
    parser.add_argument('models', nargs='+')


def main(args):
    client = OllamaClient(
        host=args.host,
        timeout=args.timeout,
    )
    for model in args.models:
        try:
            client.pull_model(model)
            print(f'Pulled {model}')
        except errors.OllamaModelDoesNotExistError as err:
            print(err)
