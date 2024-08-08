import json
import logging
from collections import defaultdict

from ollama_benchmark import utils
from ollama_benchmark.speed.tester import Tester

logger = logging.getLogger("ollama_benchmark")


def make_parser(subparsers):
    parser = subparsers.add_parser("speed", help="Evaluate chat speed performance")
    parser.add_argument('--questions', default=['81'], nargs='*')
    parser.add_argument('--max-workers', default=1, type=int)
    parser.add_argument('--max_turns', default=None, type=int, required=False)
    utils.add_tester_arguments(parser)
    utils.add_ollama_config_arguments(parser)
    utils.add_monitoring_arguments(parser)


def print_results(args, questions, results, real_duration, options):
    utils.print_main()

    overall_results = {
        'model': args.model,
        'question_ids': json.dumps(questions),
        'max_workers': args.max_workers,
        'max_turns': args.max_turns,
    }
    for key, value in overall_results.items():
        print(f"{key}: {value}")
    for key, value in options.items():
        print(f"{key}: {value}")

    totals_keys = (
        'eval_duration_mean', 'prompt_eval_duration_mean',
        'eval_count_mean', 'prompt_eval_count_mean',
        'eval_rate_mean', 'prompt_eval_rate_mean',
        'total_durations',
    )
    totals = defaultdict(list)
    skipped = ('question', 'question_id', 'responses')
    for i, result in enumerate(results):
        for key, value in result.items():
            if key in skipped:
                continue
            print(f"{i};{result['question_id']};{key}: {value}")
            if key in totals_keys:
                totals[key].append(value)

    skipped = ('total_durations', )
    for key, values in totals.items():
        if key in skipped:
            continue
        mean = utils.mean(values)
        stdev = utils.stdev(values)
        print(f"{key}: {mean}")
        print(f"{key.replace('mean', 'stdev')}: {stdev}")

    total_durations = [j for i in totals['total_durations'] for j in i]
    total_duration = sum(total_durations)
    print(f"total_duration: {total_duration}")
    print(f"real_duration: {real_duration}")


def main(args):
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
    tester = Tester(
        host=args.host,
        timeout=args.timeout,
        model=args.model,
        ollama_options=ollama_options,
        pull=args.pull,
        prewarm=args.prewarm,
        max_workers=args.max_workers,
        questions=args.questions,
        monitoring_enabled=args.monitoring_enabled,
    )
    run_results = tester.run_suite()

    print_results(
        args,
        tester.get_tasks(),
        run_results['results'],
        run_results['real_duration'],
        ollama_options,
    )

    if tester.monitoring_enabled:
        with open(args.monitoring_output, 'w') as fd:
            fd.write(json.dumps(
                tester.get_monitoring_results(),
                cls=utils.JSONEncoder,
            ))
