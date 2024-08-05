import argparse
from ollama_benchmark import settings
from ollama_benchmark import utils
from ollama_benchmark.loggers import logger
from ollama_benchmark.client import OllamaClient
from ollama_benchmark.speed import main as speed
from ollama_benchmark.load import main as load
from ollama_benchmark.embedding import main as embedding


def print_questions():
    print("   ID | Category | # Turns | Turns")
    questions = utils.data_manager.list_questions()
    row_template = "{question_id:5} | {category:^8} | {num_turns:3} | {turns}"
    for question in questions:
        row = row_template.format(
            num_turns=len(question['turns']),
            **question
        )
        print(row)


def pull_models(args):
    client = OllamaClient(
        host=args.host,
        timeout=args.timeout,
    )
    for model in args.models:
        client.client.pull(model)


def unload_models(args):
    client = OllamaClient(
        host=args.host,
        timeout=args.timeout,
    )
    client.unload_all()


def generate(prompt=None):
    prompt = prompt or input('> ')
    client = OllamaClient(
        host=args.host,
        timeout=args.timeout,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', default=settings.HOST)
    parser.add_argument('--timeout', default=settings.TIMEOUT)
    parser.add_argument('-v', '--verbosity', type=int, default=0, choices=(0, 1, 2))

    subparsers = parser.add_subparsers(dest="action")

    speed.make_parser(subparsers)
    embedding.make_parser(subparsers)
    load.make_args(subparsers)
    judge_parser = subparsers.add_parser("judge", help="Evaluate chat quality")
    subparsers.add_parser("unload", help="Unload all models from memory")
    subparsers.add_parser("questions", help="")

    pull_models_parser = subparsers.add_parser("pull-models", help="")
    pull_models_parser.add_argument('models', action='append')

    args = parser.parse_args()

    logger.setLevel(40-(10+args.verbosity*10))
    logger.info('Log level: %s', logger.level)

    if args.action == 'speed':
        speed.main(args)
    elif args.action == 'embedding':
        embedding.main(args)
    elif args.action == 'load':
        load.main(args)
    elif args.action == 'unload':
        unload_models(args)
    elif args.action == 'judge':
        print("Not implemented yet")
    elif args.action == 'questions':
        print_questions()
    elif args.action == 'pull-models':
        pull_models(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
