import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from ollama_benchmark import settings
from ollama_benchmark import utils
from ollama_benchmark.client import OllamaClient
from ollama_benchmark.speed import run_test

logger = logging.getLogger("ollama_benchmark")


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


def main(parser, args):
    client = OllamaClient(
        host=args.host,
        timeout=args.timeout,
    )

    tester = run_test.Tester(
        client,
    )
    questions = args.questions

    if 'all' in args.questions:
        questions = [
            q['question_id']
            for q in utils.data_manager.list_questions()
        ]

    pool_kwargs = {'max_workers': args.max_workers}
    futures = []
    options = {
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
    if args.prewarm:
        tester.prewarm(args.model)

    with ThreadPoolExecutor(**pool_kwargs) as executor:
        t0 = time.time()
        for question_id in questions:
            future = executor.submit(
                tester.run,
                question_id=question_id,
                model=args.model,
                max_turns=args.max_turns,
                options=options,
            )
            logger.info('Submitted %s', question_id)
            futures.append(future)
        results = [
            future.result()
            for future in futures
        ]
        real_duration = (time.time() - t0) * 1000
    print_results(args, questions, results, real_duration, options)
