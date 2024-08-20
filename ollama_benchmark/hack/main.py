import logging

from ollama_benchmark import settings
from ollama_benchmark import utils
from ollama_benchmark import errors
from ollama_benchmark.hack.tester import Tester, HACK_IDS

logger = logging.getLogger("ollama_benchmark")


def make_args(subparsers):
    parser = subparsers.add_parser("hack", help="Run hacking common hacking technics")
    parser.add_argument('--hacks', nargs='+')
    parser.add_argument('--judge-model', default="llama3")
    parser.add_argument('--show-hacks', action="store_true", help="Display all hacks and their IDs.")
    utils.add_tester_arguments(parser)
    utils.add_ollama_config_arguments(parser)


def print_results(args, results, ollama_options):
    utils.print_main()

    overall_results = {
        'model': args.model,
        'hack_ids': args.hacks,
    }
    for key, value in overall_results.items():
        print(f"{key}: {value}")
    for key, value in ollama_options.items():
        print(f"{key}: {value}")

    skipped = ('hack_id', 'ollama_version')
    for result in results:
        for key, value in result.items():
            if key in skipped:
                continue
            print(f"{result['hack_id']};{key}: {value}")


def main(args):
    if args.show_hacks:
        print('Hack IDs')
        print('\n'.join(HACK_IDS))
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
    tester = Tester(
        host=args.host,
        model=args.model,
        ollama_options=ollama_options,
        judge_model=args.judge_model,
        pull=args.pull,
        prewarm=args.prewarm,
        monitoring_enabled=False,
        hacks=args.hacks,
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
    )
