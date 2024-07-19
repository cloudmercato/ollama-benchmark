import argparse
from ollama_benchmark import settings
from ollama_benchmark.loggers import logger
from ollama_benchmark.speed.main import main as speed
from ollama_benchmark import utils


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', default=settings.HOST)
    parser.add_argument('--timeout', default=settings.TIMEOUT)
    parser.add_argument('-v', '--verbosity', type=int, default=0, choices=(0, 1, 2))

    subparsers = parser.add_subparsers(dest="action")

    question_parser = subparsers.add_parser("questions", help="")

    speed_parser = subparsers.add_parser("speed", help="")
    speed_parser.add_argument('--model', default=settings.MODEL)
    speed_parser.add_argument('--questions', default=['all'], nargs='*')
    speed_parser.add_argument('--max-workers', default=1, type=int)
    speed_parser.add_argument('--max_turns', default=None, type=int, required=False)
    speed_parser.add_argument('--list-questions', action="store_true")

    args = parser.parse_args()

    logger.setLevel(40-(10+args.verbosity*10))
    logger.info('Log level: %s', logger.level)

    if args.action == 'speed':
        speed(speed_parser, args)
    elif args.action == 'questions':
        print_questions()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
