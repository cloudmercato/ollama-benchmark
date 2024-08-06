import json
import logging
from collections import defaultdict

from ollama_benchmark import utils
from ollama_benchmark.embedding.tester import Tester

logger = logging.getLogger("ollama_benchmark")


def make_parser(subparsers):
    parser = subparsers.add_parser("embedding", help="Evaluate embedding peformance")
    parser.add_argument('--sample-sizes', default=[], type=int, nargs='*')
    parser.add_argument('--sample-langs', default=[], nargs='*')
    parser.add_argument('--max-workers', default=1, type=int)
    parser.add_argument('--num-tasks', default=1, type=int)
    parser.add_argument('--num-inputs', default=1, type=int)
    utils.add_tester_arguments(parser)
    utils.add_ollama_config_arguments(parser)


def print_results(args, results, options):
    utils.print_main()

    overall_results = {
        'model': args.model,
        'sample_sizes': json.dumps(args.sample_sizes),
        'sample_langs': json.dumps(args.sample_langs),
        'max_workers': args.max_workers,
    }
    for key, value in overall_results.items():
        print(f"{key}: {value}")
    for key, value in options.items():
        print(f"{key}: {value}")
    # Duration
    values = [r['duration'] for r in results['results']]
    duration = {
        'min': utils.min(values),
        'max': utils.max(values),
        'mean': utils.mean(values),
        'stdev': utils.stdev(values),
        'perc95': utils.perc95(values),
    }
    for key, value in duration.items():
        print(f"duration_{key}: {value}")
    total_duration = sum(values)
    print(f"total_duration: {total_duration}")
    print(f"real_duration: {results['real_duration']}")
    # Rate
    values = [(1/r['duration']) for r in results['results']]
    rates = {
        'min': utils.min(values),
        'max': utils.max(values),
        'mean': utils.mean(values),
        'stdev': utils.stdev(values),
        'perc95': utils.perc95(values),
    }
    for key, value in rates.items():
        print(f"rate_{key}: {value}")
    # Errors
    values = [r['errors'] for r in results['results']]
    print(f"errors: {sum(values)}")
    errors = {
        'mean': utils.mean(values),
        'stdev': utils.stdev(values),
    }
    for key, value in errors.items():
        print(f"errors_per_worker_{key}: {value}")


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
        langs=args.sample_langs,
        sizes=args.sample_sizes,
        num_tasks=args.num_tasks,
        num_inputs=args.num_inputs,
    )
    run_results = tester.run_suite()

    print_results(
        args,
        run_results,
        ollama_options,
    )

    if tester.monitoring_enabled:
        with open(args.monitoring_output, 'w') as fd:
            fd.write(json.dumps(
                tester.get_monitoring_results(),
                cls=utils.JSONEncoder,
            ))
