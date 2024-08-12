import argparse

from ollama_benchmark import settings
from ollama_benchmark import utils
from ollama_benchmark import errors
from ollama_benchmark.loggers import logger
from ollama_benchmark.speed import main as speed
from ollama_benchmark.load import main as load
from ollama_benchmark.judge import main as judge
from ollama_benchmark.embedding import main as embedding
from ollama_benchmark import chat
from ollama_benchmark import pull_models
from ollama_benchmark import unload_models
from ollama_benchmark import print_questions

ACTIONS = {
    'speed': speed.main,
    'embedding': embedding.main,
    'load': load.main,
    'chat': chat.main,
    'pull_models': pull_models.main,
    'unload_models': unload_models.main,
    'judge': judge.main,
    'questions': print_questions.main,
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--host', default=settings.HOST,
        help="Ollama server address as host:port",
    )
    parser.add_argument(
        '--timeout',
        default=settings.TIMEOUT, type=int,
        help="Maximum duration before ollama client timeout",
    )
    parser.add_argument('-v', '--verbosity', type=int, default=0, choices=(0, 1, 2))
    parser.add_argument(
        '-q', '--quiet',
        action="store_true",
        help="Completly disable any output (wip)",
    )

    subparsers = parser.add_subparsers(dest="action")

    speed.make_parser(subparsers)
    embedding.make_parser(subparsers)
    load.make_args(subparsers)
    chat.make_args(subparsers)
    judge.make_args(subparsers)
    pull_models.make_args(subparsers)
    unload_models.make_args(subparsers)
    print_questions.make_args(subparsers)

    args = parser.parse_args()
    if args.action not in ACTIONS:
        parser.print_help()
        exit(1)

    verbosity = args.verbosity
    stdout = None
    if args.quiet:
        verbosity = 0
        stdout = open('/dev/null', 'w')
    logger.setLevel(40-(10+verbosity*10))

    action = ACTIONS[args.action]
    try:
        action(args)
    except KeyboardInterrupt:
        print("Stopped", file=stdout)
        exit(1)
    except errors.OllamaBenchmarkError as err:
        print(err, file=stdout)
        exit(1)


if __name__ == '__main__':
    main()
