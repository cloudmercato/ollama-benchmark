from ollama_benchmark import utils


def make_args(subparsers):
    parser = subparsers.add_parser(
        "questions",
        help="Print the questions available for tests",
    )


def main(args):
    print("   ID | Category | # Turns | Turns")
    questions = utils.data_manager.list_questions()
    row_template = "{question_id:5} | {category:^8} | {num_turns:3} | {turns}"
    for question in questions:
        row = row_template.format(
            num_turns=len(question['turns']),
            **question
        )
        print(row)
