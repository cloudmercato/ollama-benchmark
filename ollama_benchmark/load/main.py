import json
import logging

from ollama_benchmark import utils
from ollama_benchmark.load.tester import Tester

logger = logging.getLogger("ollama_benchmark")


def make_args(subparsers):
    parser = subparsers.add_parser("load", help="Evaluate model loading speed")
    parser.add_argument('--max-workers', default=1, type=int)
    parser.add_argument('models', nargs='+')
    utils.add_tester_arguments(parser)


def print_results(args, results):
    utils.print_main()

    overall_results = {
        'models': json.dumps(args.models),
        'max_workers': args.max_workers,
    }
    for key, value in overall_results.items():
        print(f"{key}: {value}")
    # Duration
    values = [r['duration'] for r in results['results']]
    duration = {
        'min': min(values),
        'max': max(values),
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
        'min': min(values),
        'max': max(values),
        'mean': utils.mean(values),
        'stdev': utils.stdev(values),
        'perc95': utils.perc95(values),
    }
    for key, value in rates.items():
        print(f"rate_{key}: {value}")
    # Errors
    values = [r['error'] for r in results['results'] if r['error']]
    print(f"errors: {len(values)}")


def main(args):
    tester = Tester(
        host=args.host,
        timeout=args.timeout,
        model=None,
        pull=args.pull,
        prewarm=False,
        max_workers=args.max_workers,
        models=args.models,
    )
    run_results = tester.run_suite()

    print_results(
        args,
        run_results,
    )

    if tester.monitoring_enabled:
        with open(args.monitoring_output, 'w') as fd:
            fd.write(json.dumps(
                tester.get_monitoring_results(),
                cls=utils.JSONEncoder,
            ))
