from ollama_benchmark import utils


def make_args(subparsers):
    parser = subparsers.add_parser(
        "questions",
        help="Print the questions available for tests",
    )

    parser.add_argument(
        '--list-categories', action="store_true",
    )
    parser.add_argument(
        '--filter-categories', nargs='*', dest="categories"
    )
    parser.add_argument(
        '--list-sources', action="store_true",
    )
    parser.add_argument(
        '--filter-sources', nargs='*', dest="sources"
    )
    parser.add_argument(
        '--list-langs', action="store_true",
    )
    parser.add_argument(
        '--filter-langs', nargs='*', dest="langs"
    )


TEMPLATES = {
    0: (
        "   ID | Category | # Turns",
        "{question_id:5} | {category:^8} | {num_turns:3}"
    ),
    1: (
        "   ID | Category | # Turns | Lang | Source",
        "{question_id:5} | {category:^8} | {num_turns:7} | {language:4} | {source}"
    ),
    2: (
        "   ID | Category | # Turns | Lang | Source             | Turns",
        "{question_id:5} | {category:^8} | {num_turns:7} | {language:4} | {source:18} | {turns}"
    ),
}


def list_categories(questions):
    template = "{category:10} | {count:6}"

    questions_categories = [q['category'] for q in questions]
    categories = sorted(set(questions_categories))
    print("Category   | Count")
    for category in categories:
        count = len([c for c in questions_categories if c == category])
        row = template.format(category=category, count=count)
        print(row)


def list_sources(questions):
    template = "{source:18} | {count}"

    questions_sources = [q['source'] for q in questions]
    sources = sorted(set(questions_sources))
    print("Source             | Count")
    for source in sources:
        count = len([c for c in questions_sources if c == source])
        row = template.format(source=source, count=count)
        print(row)


def list_langs(questions):
    template = "{lang:4} | {count}"

    questions_langs = [q['language'] for q in questions]
    langs = sorted(set(questions_langs))
    print("Lang | Count")
    for lang in langs:
        count = len([c for c in questions_langs if c == lang])
        row = template.format(lang=lang, count=count)
        print(row)


def main(args):
    questions = utils.data_manager.list_questions()
    if args.list_categories:
        list_categories(questions)
        exit(0)
    if args.list_sources:
        list_sources(questions)
        exit(0)
    if args.list_langs:
        list_langs(questions)
        exit(0)

    header, body_template = TEMPLATES[args.verbosity]
    print(header)

    for question in questions:
        if args.categories and question['category'] not in args.categories:
            continue
        if args.sources and question['source'] not in args.sources:
            continue
        if args.langs and question['language'] not in args.langs:
            continue
        row = body_template.format(
            num_turns=len(question['turns']),
            **question
        )
        print(row)
