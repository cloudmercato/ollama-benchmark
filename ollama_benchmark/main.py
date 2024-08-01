import argparse
from ollama_benchmark import settings
from ollama_benchmark import utils
from ollama_benchmark.loggers import logger
from ollama_benchmark.client import OllamaClient
from ollama_benchmark.speed.main import main as speed


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

    question_parser = subparsers.add_parser("questions", help="")

    speed_parser = subparsers.add_parser("speed", help="")
    speed_parser.add_argument('--model', default=settings.MODEL)
    speed_parser.add_argument(
        '--mirostat', type=int, default=0, choices=[0, 1, 2],
        help='Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)'
    )
    speed_parser.add_argument(
        '--mirostat_eta', type=float, default=0.1,
        help='Influences how quickly the algorithm responds to feedback from the generated text. (Default: 0.1)'
    )
    speed_parser.add_argument(
        '--mirostat_tau', type=float, default=5.0,
        help='Controls the balance between coherence and diversity of the output. (Default: 5.0)'
    )
    speed_parser.add_argument(
        '--num_ctx', type=int, default=2048,
        help='Sets the size of the context window used to generate the next token. (Default: 2048)'
    )
    speed_parser.add_argument(
        '--repeat_last_n', type=int, default=64,
        help='Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)'
    )
    speed_parser.add_argument('--repeat_penalty', type=float, default=1.1,
        help='Sets how strongly to penalize repetitions. (Default: 1.1)'
    )
    speed_parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='The temperature of the model. (Default: 0.8)'
    )
    speed_parser.add_argument(
        '--seed', type=int, default=0,
        help='Sets the random number seed to use for generation. (Default: 0)'
    )
    speed_parser.add_argument(
        '--stop', type=str, default=None,
        help='Sets the stop sequences to use. (Default: None)'
    )
    speed_parser.add_argument(
        '--tfs_z', type=float, default=1.0,
        help='Tail free sampling is used to reduce the impact of less probable tokens from the output. (Default: 1)'
    )
    speed_parser.add_argument(
        '--num_predict', type=int, default=128,
        help='Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)'
    )
    speed_parser.add_argument(
        '--top_k', type=int, default=40,
        help='Reduces the probability of generating nonsense. (Default: 40)'
    )
    speed_parser.add_argument(
        '--top_p', type=float, default=0.9,
        help='Works together with top-k. (Default: 0.9)'
    )
    speed_parser.add_argument(
        '--min_p', type=float, default=0.0,
        help='Alternative to the top_p, and aims to ensure a balance of quality and variety. (Default: 0.0)'
    )
    speed_parser.add_argument(
        '--do-not-prewarm', action="store_false", dest="prewarm",
        help='Run a prewarm request to load model into GPU',
    )

    speed_parser.add_argument('--questions', default=['all'], nargs='*')
    speed_parser.add_argument('--max-workers', default=1, type=int)
    speed_parser.add_argument('--max_turns', default=None, type=int, required=False)

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
