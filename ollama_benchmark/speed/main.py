import logging
from concurrent.futures import ThreadPoolExecutor
from ollama_benchmark import settings
from ollama_benchmark import utils
from ollama_benchmark.client import OllamaClient
from ollama_benchmark.speed import run_test

logger = logging.getLogger("ollama_benchmark")


def print_results(args, questions, results):
    overall_results = {
        'model': args.model,
        'question_ids': questions,
        'max_workers': args.max_workers,
        'max_turns': args.max_turns,
    }
    for key, value in overall_results.items():
        print(f"{key}: {value}")

    skipped = ('question', 'question_id', 'responses')
    for result in results:
        for key, value in result.items():
            if key in skipped:
                continue
            print(f"{result['question_id']};{key}: {value}")


def main(parser, args):
    client = OllamaClient(
        host=args.host,
        timeout=args.timeout,
    )

    tester = run_test.Tester(
        client,
    )
    questions = args.questions
    if args.list_questions:
        print("   ID | Category | # Turns | Turns")
        questions = utils.data_manager.list_questions()
        row_template = "{question_id:5} | {category:^8} | {num_turns:3}     | {turns}"
        for question in questions:
            row = row_template.format(
                num_turns=len(question['turns']),
                **question
            )
            print(row)
        return

    if 'all' in args.questions:
        questions = [
            q['question_id']
            for q in utils.data_manager.list_questions()
        ]

    pool_kwargs = {'max_workers': args.max_workers}
    futures = []
    with ThreadPoolExecutor(**pool_kwargs) as executor:
        for question_id in questions:
            future = executor.submit(
                tester.run,
                question_id=question_id,
                model=args.model,
                max_turns=args.max_turns,
            )
            logger.info('Submitted %s', question_id)
            futures.append(future)
        results = [
            future.result()
            for future in futures
        ]
    print_results(args, questions, results)
