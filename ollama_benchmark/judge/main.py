import argparse
import json
import logging

from ollama_benchmark import settings
from ollama_benchmark import utils
from ollama_benchmark import errors
from ollama_benchmark.judge.tester import Tester
from ollama_benchmark.judge.tester import JUDGE_SYSTEM_PROMPT, JUDGE_PROMPT

logger = logging.getLogger("ollama_benchmark")


def make_args(subparsers):
    parser = subparsers.add_parser("judge", help="Evaluate answer quality with LLM-as-a-Judge")
    parser.add_argument(
        '--system-prompt', type=argparse.FileType('r'), required=False,
        help='Path to system prompt to use.'
    )
    parser.add_argument('--judge-host', required=None)
    parser.add_argument('--judge-model', default=settings.JUDGE_MODEL)
    parser.add_argument('--question', default='81')
    parser.add_argument('--max_turns', default=None, type=int, required=False)
    utils.add_tester_arguments(parser)
    utils.add_ollama_config_arguments(parser)
    parser.add_argument(
        '--judge-mirostat', type=int, default=0, choices=[0, 1, 2],
        help='Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)'
    )
    parser.add_argument(
        '--judge-mirostat_eta', type=float, default=0.1,
        help='Influences how quickly the algorithm responds to feedback from the generated text. (Default: 0.1)'
    )
    parser.add_argument(
        '--judge-mirostat_tau', type=float, default=5.0,
        help='Controls the balance between coherence and diversity of the output. (Default: 5.0)'
    )
    parser.add_argument(
        '--judge-num_ctx', type=int, default=2048,
        help='Sets the size of the context window used to generate the next token. (Default: 2048)'
    )
    parser.add_argument(
        '--judge-repeat_last_n', type=int, default=64,
        help='Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)'
    )
    parser.add_argument(
        '--judge-repeat_penalty', type=float, default=1.1,
        help='Sets how strongly to penalize repetitions. (Default: 1.1)'
    )
    parser.add_argument(
        '--judge-temperature', type=float, default=0.8,
        help='The temperature of the model. (Default: 0.8)'
    )
    parser.add_argument(
        '--judge-seed', type=int, default=0,
        help='Sets the random number seed to use for generation. (Default: 0)'
    )
    parser.add_argument(
        '--judge-tfs_z', type=float, default=1.0,
        help='Tail free sampling is used to reduce the impact of less probable tokens from the output. (Default: 1)'
    )
    parser.add_argument(
        '--judge-num_predict', type=int, default=128,
        help='Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)'
    )
    parser.add_argument(
        '--judge-top_k', type=int, default=40,
        help='Reduces the probability of generating nonsense. (Default: 40)'
    )
    parser.add_argument(
        '--judge-top_p', type=float, default=0.9,
        help='Works together with top-k. (Default: 0.9)'
    )
    parser.add_argument(
        '--judge-min_p', type=float, default=0.0,
        help='Alternative to the top_p, and aims to ensure a balance of quality and variety. (Default: 0.0)'
    )

    parser.add_argument(
        '--judge-system-prompt', type=argparse.FileType('r'), required=False,
        help='Path to alternative judge system prompt to use.'
    )
    parser.add_argument(
        '--judge-prompt', type=argparse.FileType('r'), required=False,
        help='Path to alternative judge prompt to use.'
    )
    parser.add_argument(
        '--show-judge-system-prompt', action="store_true",
        help='Display default judge system prompt',
    )
    parser.add_argument(
        '--show-judge-prompt', action="store_true",
        help='Display default judge prompt',
    )

    parser.add_argument(
        '--save-messages', type=argparse.FileType('w'), required=False,
        help='Path where to save the messages from the conversation as JSON.'
    )
    parser.add_argument(
        '--load-messages', type=argparse.FileType('r'), required=False,
        help='Path where to get the messages given to the judge.'
    )


def print_results(args, results, ollama_options, ollama_judge_options):
    utils.print_main()

    overall_results = {
        'model': args.model,
        'judge_model': args.judge_model,
        'question_id': args.question,
        'max_turns': args.max_turns,
    }
    for key, value in overall_results.items():
        print(f"{key}: {value}")
    for key, value in ollama_options.items():
        print(f"{key}: {value}")
    for key, value in ollama_judge_options.items():
        print(f"judge_{key}: {value}")

    skipped = ('judgements', 'self_judgement', 'messages')
    for result in results:
        for key, value in result.items():
            if key in skipped:
                continue
            print(f"{key}: {value}")
    messages = [r['messages'] for r in results]
    print(f"messages: {json.dumps(messages)}")
    # Scores
    judgements = [
        judgement
        for run in results
        for judgement in run['judgements']
    ]
    rating_keys = (
        'relevance',
        'coherence',
        'quality',
        'language',
        'originality',
        'neutrality',
        'total_rating',
    )
    for key in rating_keys:
        total_ratings = [i[key] for i in judgements]
        data = {
            f'{key}_mean': utils.mean(total_ratings),
            f'{key}_stdev': utils.stdev(total_ratings),
        }
        for key_, value in data.items():
            print(f"{key_}: {value}")
        print(f"{key}: {total_ratings}")
    for i, judgement in enumerate(judgements):
        if 'evaluation' in judgement:
            print(f"{i};evaluation: {judgement['evaluation']}")
        if 'feedback' in judgement:
            print(f"{i};feedback: {judgement.get('feedback')}")
    # Not self-judge with loaded messages
    if not args.load_messages:
        self_judgements = [
            run['self_judgement']
            for run in results
        ]
        total_self_ratings = [i['total_rating'] for i in self_judgements
                              if 'total_rating' in i]
        data = {
            'total_self_rating_mean': utils.mean(total_self_ratings),
            'total_self_rating_stdev': utils.stdev(total_self_ratings),
        }
        for key, value in data.items():
            print(f"{key}: {value}")
        print(f"total_self_ratings: {total_self_ratings}")
        for i, judgement in enumerate(self_judgements):
            if 'evaluation' in judgement:
                print(f"{i};self_evaluation: {judgement['evaluation']}")
            print(f"{i};self_feedback: {judgement.get('feedback')}")


def main(args):
    if args.show_judge_system_prompt:
        print(JUDGE_SYSTEM_PROMPT)
        exit(0)
    elif args.show_judge_prompt:
        print(JUDGE_PROMPT)
        exit(0)

    ollama_options = {
        'mirostat': args.mirostat,
        'mirostat_eta': args.mirostat_eta,
        'mirostat_tau': args.mirostat_tau,
        'num_ctx': args.num_ctx,
        'repeat_last_n': args.repeat_last_n,
        'repeat_penalty': args.repeat_penalty,
        'temperature': args.temperature,
        'seed': args.seed,
        'stop': args.stop,
        'tfs_z': args.tfs_z,
        'num_predict': args.num_predict,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'min_p': args.min_p,
    }
    ollama_judge_options = {
        'mirostat': args.judge_mirostat,
        'mirostat_eta': args.judge_mirostat_eta,
        'mirostat_tau': args.judge_mirostat_tau,
        'num_ctx': args.judge_num_ctx,
        'repeat_last_n': -1,
        'repeat_penalty': 1.1,
        'temperature': args.judge_temperature,
        'seed': args.judge_seed,
        'stop': None,
        'tfs_z': args.judge_tfs_z,
        'num_predict': args.judge_num_predict,
        'top_k': args.judge_top_k,
        'top_p': args.judge_top_p,
        'min_p': args.judge_min_p,
    }
    tester = Tester(
        host=args.host,
        timeout=args.timeout,
        model=args.model,
        system_prompt=args.system_prompt,
        ollama_options=ollama_options,
        ollama_judge_options=ollama_judge_options,
        pull=args.pull,
        prewarm=args.prewarm,
        max_workers=1,
        question=args.question,
        judge_model=args.judge_model,
        judge_system_prompt=args.judge_system_prompt,
        judge_prompt=args.judge_prompt,
        monitoring_enabled=False,
        load_messages=args.load_messages,
    )
    try:
        tester.check_config()
    except errors.ConfigurationError as err:
        print(f"Configuration error: {err}")
        exit(1)

    run_results = tester.run_suite()

    print_results(
        args,
        run_results['results'],
        ollama_options,
        ollama_judge_options,
    )

    if args.save_messages:
        messages = [r['messages'] for r in run_results['results']]
        args.save_messages.write(json.dumps(messages))
