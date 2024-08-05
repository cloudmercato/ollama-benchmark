import os
import datetime
import json
from base64 import b64encode
import urllib.request
import math
import statistics

from ollama_benchmark import __version__
from ollama_benchmark import settings
from ollama_benchmark import questions

QUESTION_URL = 'https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl'


def print_main():
    data = {
        'version': __version__,
    }
    for key, value in data.items():
        print(f"{key}: {value}")


class DataManager:
    def __init__(self, data_dir=settings.DATA_DIR):
        self.data_dir = data_dir

    @property
    def question_file_name(self):
        if not hasattr(self, '_question_file_name'):
            self._question_file_name = os.path.join(self.data_dir, 'question.jsonl')
        return self._question_file_name

    def download_questions(self):
        os.makedirs(os.path.dirname(self.question_file_name), exist_ok=True)
        urllib.request.urlretrieve(QUESTION_URL, self.question_file_name)

    @property
    def simple_questions(self):
        if not hasattr(self, '_simple_questions'):
            self._simple_questions = []
            try:
                fd = open(self.question_file_name, 'r')
            except FileNotFoundError:
                self.download_questions()
                fd = open(self.question_file_name, 'r')
            for line in fd:
                self._simple_questions.append(json.loads(line))
            fd.close()
        return self._simple_questions

    @property
    def image_questions(self):
        return questions.IMAGE_QUESTIONS

    @property
    def questions(self):
        if not hasattr(self, '_questions'):
            self._questions = []
            self._questions += self.simple_questions
            self._questions += self.image_questions
        return self._questions

    def list_questions(self):
        return self.questions

    def get_question(self, id_):
        for question in self.questions:
            if str(question['question_id']) == str(id_):
                return question
        msg = "Question '%s' does not exist." % id_
        raise Exception(msg)

    def get_question_b64_images(self, id_):
        question = self.get_question(id_)
        b64s = []
        for i, url in enumerate(question['image_urls']):
            filename = '{}-{}.{}'.format(i, id_, url.split('.')[-1])
            full_filename = os.path.join(self.data_dir, filename)
            try:
                fd = open(full_filename, 'rb')
            except FileNotFoundError:
                os.makedirs(os.path.dirname(full_filename), exist_ok=True)
                urllib.request.urlretrieve(url, full_filename)
                fd = open(full_filename, 'rb')
            content = fd.read()
            fd.close()
            b64s.append(b64encode(content))
        return b64s


data_manager = DataManager()


def mean(values):
    if not values:
        return
    return statistics.mean(values)


def stdev(values):
    if not values:
        return
    if len(values) == 1:
        return .0
    return statistics.stdev(values)


def perc95(values):
    if not values:
        return
    if len(values) == 1:
        return values[0]
    perc = 95
    size = len(values)
    return sorted(values)[int(math.ceil((size * perc) / 100)) - 1]


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()


def add_test_argument(parser):
    parser.add_argument('--model', default=settings.MODEL)
    parser.add_argument(
        '--mirostat', type=int, default=0, choices=[0, 1, 2],
        help='Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)'
    )
    parser.add_argument(
        '--mirostat_eta', type=float, default=0.1,
        help='Influences how quickly the algorithm responds to feedback from the generated text. (Default: 0.1)'
    )
    parser.add_argument(
        '--mirostat_tau', type=float, default=5.0,
        help='Controls the balance between coherence and diversity of the output. (Default: 5.0)'
    )
    parser.add_argument(
        '--num_ctx', type=int, default=2048,
        help='Sets the size of the context window used to generate the next token. (Default: 2048)'
    )
    parser.add_argument(
        '--repeat_last_n', type=int, default=64,
        help='Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)'
    )
    parser.add_argument('--repeat_penalty', type=float, default=1.1,
        help='Sets how strongly to penalize repetitions. (Default: 1.1)'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='The temperature of the model. (Default: 0.8)'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Sets the random number seed to use for generation. (Default: 0)'
    )
    parser.add_argument(
        '--stop', type=str, default=None,
        help='Sets the stop sequences to use. (Default: None)'
    )
    parser.add_argument(
        '--tfs_z', type=float, default=1.0,
        help='Tail free sampling is used to reduce the impact of less probable tokens from the output. (Default: 1)'
    )
    parser.add_argument(
        '--num_predict', type=int, default=128,
        help='Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)'
    )
    parser.add_argument(
        '--top_k', type=int, default=40,
        help='Reduces the probability of generating nonsense. (Default: 40)'
    )
    parser.add_argument(
        '--top_p', type=float, default=0.9,
        help='Works together with top-k. (Default: 0.9)'
    )
    parser.add_argument(
        '--min_p', type=float, default=0.0,
        help='Alternative to the top_p, and aims to ensure a balance of quality and variety. (Default: 0.0)'
    )
    parser.add_argument(
        '--do-not-pull', action="store_false", dest="pull",
        help='Do not pull the model',
    )
    parser.add_argument(
        '--do-not-prewarm', action="store_false", dest="prewarm",
        help='Run a prewarm request to load model into GPU',
    )

    parser.add_argument(
        '--disable-monitoring', action="store_false", dest="monitoring",
    )
    parser.add_argument(
        '--monitoring-interval', type=int, default=5,
    )
    parser.add_argument(
        '--monitoring-probers', action='append'
    )
    parser.add_argument(
        '--monitoring-output', default="/dev/stderr"
    )
    return parser
