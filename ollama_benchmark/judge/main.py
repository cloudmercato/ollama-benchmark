import logging
from ollama_benchmark import settings
from ollama_benchmark import utils
from ollama_benchmark.judge.tester import Tester

logger = logging.getLogger("ollama_benchmark")


def make_args(subparsers):
    parser = subparsers.add_parser("judge", help="Evaluate answer quality with LLM-as-a-Judge")
    parser.add_argument('--judge-model', default=settings.JUDGE_MODEL)
    parser.add_argument('--question', default='81')
    parser.add_argument('--max_turns', default=None, type=int, required=False)
    utils.add_tester_arguments(parser)
    utils.add_ollama_config_arguments(parser)


def print_results(args, results, ollama_options):
    utils.print_main()
    judgements = [
        judgement
        for run in results
        for judgement in run['judgements']
    ]

    overall_results = {
        'model': args.model,
        'judge_model': args.judge_model,
        'question_id': args.question,
        'max_turns': args.max_turns,
    }
    for key, value in overall_results.items():
        print(f"{key}: {value}")
    skipped = ('judgements', )
    for result in results:
        for key, value in result.items():
            if key in skipped:
                continue
            print(f"{key}: {value}")

    total_ratings = [i['total_rating'] for i in judgements]
    data = {
        'total_rating_mean': utils.mean(total_ratings),
        'total_rating_stdev': utils.stdev(total_ratings),
    }
    for key, value in data.items():
        print(f"{key}: {value}")
    print(f"total_ratings: {total_ratings}")

    for i, judgement in enumerate(judgements):
        print(f"{i};evaluation: {judgement['evaluation']}")
        print(f"{i};feedback: {judgement.get('feedback')}")


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
        max_workers=1,
        question=args.question,
        judge_model=args.judge_model,
        monitoring_enabled=False,
    )
    run_results = tester.run_suite()

    print_results(
        args,
        run_results['results'],
        ollama_options,
    )
