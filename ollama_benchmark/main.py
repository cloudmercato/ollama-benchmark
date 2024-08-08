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

    parser.add_argument('--host', default=settings.HOST)
    parser.add_argument('--timeout', default=settings.TIMEOUT)
    parser.add_argument('-v', '--verbosity', type=int, default=0, choices=(0, 1, 2))

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

    logger.setLevel(40-(10+args.verbosity*10))
    logger.info('Log level: %s', logger.level)

    action = ACTIONS[args.action]
    try:
        action(args)
    except KeyboardInterrupt:
        print("Stopped")
        exit(1)
    except errors.OllamaBenchmarkError as err:
        print(err)


if __name__ == '__main__':
    main()
