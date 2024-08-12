import os
import json
import pickle
import urllib.request
from base64 import b64encode

from ollama_benchmark import settings
from ollama_benchmark import questions
from ollama_benchmark import errors

MLBENCH_URL = 'https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl'
ODISSEY_MATH_URL = 'https://github.com/protagolabs/odyssey-math/raw/main/final-odyssey-math-with-levels.jsonl'
CODEULTRAFEEDBACK_URL = 'hf://datasets/coseal/CodeUltraFeedback/data/train-00000-of-00001.parquet'
CHC_BENCH_BASE_URL = "hf://datasets/m-a-p/CHC-Bench/"
CHC_BENCH_URLS = {
    'coding': 'data/coding-00000-of-00001-80d4feca8d41e2a2.parquet',
    'hard_case': 'data/hard_case-00000-of-00001-eac0d1d53614f880.parquet',
    'math': 'data/math-00000-of-00001-1c703fb37d6a3d4c.parquet',
    'read_compre': 'data/read_compre-00000-of-00001-d1450b0de0c8feab.parquet',
    'roleplaying': 'data/roleplaying-00000-of-00001-f6b77c19d1f54f6d.parquet',
    'science': 'data/science-00000-of-00001-7bdc8f67487119a0.parquet',
    'social': 'data/social-00000-of-00001-6e74f2e94f6031ae.parquet',
    'writting': 'data/writting-00000-of-00001-21e24be5d854b6e3.parquet'
}
CHC_BENCH_CATEGORIES = {
    '代码': 'coding',
    '人文历史': 'humanities',
    '写作类': 'writing',
    '数学题': 'math',
    '科学类': 'stem',
    '角色扮演': 'roleplay',
    '阅读理解': 'extraction',
    '难题集萃': 'reasoning',
}

read_parquet = None
try:
    from pandas import read_parquet
except ImportError:
    try:
        from polars import read_parquet
    except ImportError:
        pass


class DataManager:
    def __init__(self, data_dir=settings.DATA_DIR):
        self.data_dir = data_dir

    def _download_parquet(self, url, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df = read_parquet(url)
        df.to_pickle(filename)

    @property
    def mlbench_file_name(self):
        if not hasattr(self, '_mlbench_file_name'):
            self._mlbench_file_name = os.path.join(self.data_dir, 'mlbench.jsonl')
        return self._mlbench_file_name

    def download_mlbench(self):
        os.makedirs(os.path.dirname(self.mlbench_file_name), exist_ok=True)
        urllib.request.urlretrieve(MLBENCH_URL, self.mlbench_file_name)

    @property
    def mb_questions(self):
        if not hasattr(self, '_mb_questions'):
            self._mb_questions = []
            try:
                fd = open(self.mlbench_file_name, 'r')
            except FileNotFoundError:
                self.download_mlbench()
                fd = open(self.mlbench_file_name, 'r')
            for line in fd:
                data = json.loads(line)
                data.update({
                    'source': 'mlbench',
                    'language': 'en',
                })
                self._mb_questions.append(data)
            fd.close()
        return self._mb_questions

    @property
    def odyssey_math_file_name(self):
        if not hasattr(self, '_odyssey_math_file_name'):
            self._odyssey_math_file_name = os.path.join(self.data_dir, 'math_odyssey.jsonl')
        return self._odyssey_math_file_name

    def download_odyssey_math(self):
        os.makedirs(os.path.dirname(self.odyssey_math_file_name), exist_ok=True)
        urllib.request.urlretrieve(ODISSEY_MATH_URL, self.odyssey_math_file_name)

    @property
    def om_questions(self):
        if not hasattr(self, '_om_questions'):
            self._om_questions = []
            try:
                fd = open(self.odyssey_math_file_name, 'r')
            except FileNotFoundError:
                self.download_odyssey_math()
                fd = open(self.odyssey_math_file_name, 'r')
            for line in fd:
                data = json.loads(line)
                for key, problem in data.items():
                    self._om_questions.append({
                        'question_id': f"om_{key}".lower(),
                        'turns': [problem['question']],
                        'category': 'math',
                        'source': 'odyssey-math',
                        'language': 'en',
                    })
            fd.close()
        return self._om_questions

    @property
    def codeultrafeedback_file_name(self):
        if not hasattr(self, '_codeultrafeedback_file_name'):
            self._codeultrafeedback_file_name = os.path.join(self.data_dir, 'codeultrafeedback.pkl')
        return self._codeultrafeedback_file_name

    def download_codeultrafeedback(self):
        self._download_parquet(
            CODEULTRAFEEDBACK_URL,
            self.codeultrafeedback_file_name,
        )

    @property
    def cud_questions(self):
        if not hasattr(self, '_cud_questions'):
            self._cud_questions = []
            try:
                fd = open(self.codeultrafeedback_file_name, 'rb')
            except FileNotFoundError:
                self.download_codeultrafeedback()
                fd = open(self.codeultrafeedback_file_name, 'rb')
            df = pickle.load(fd)
            for row in df.iloc:
                data = {
                    'question_id': f"cuf_{row.name}",
                    'turns': [row.instruction],
                    'category': 'coding',
                    'source': 'codeultrafeedback',
                    'language': 'en',
                }
                self._cud_questions.append(data)
            fd.close()
        return self._cud_questions

    def _get_chc_bench_file_name(self, key):
        filename = os.path.join(self.data_dir, f'chc-bench-{key}.pkl')
        return filename

    def download_chc_bench(self):
        for key, url_suffix in CHC_BENCH_URLS.items():
            url = CHC_BENCH_BASE_URL + url_suffix
            filename = self._get_chc_bench_file_name(key)
            self._download_parquet(url, filename)

    @property
    def chc_questions(self):
        if not hasattr(self, '_chc_questions'):
            self._chc_questions = []
            for key in CHC_BENCH_URLS:
                filename = self._get_chc_bench_file_name(key)
                try:
                    fd = open(filename, 'rb')
                except FileNotFoundError:
                    self.download_chc_bench()
                    fd = open(filename, 'rb')
                df = pickle.load(fd)
                for row in df.iloc:
                    category = CHC_BENCH_CATEGORIES.get(row.category, row.category)
                    data = {
                        'question_id': f"chc_{category}_{row.name}",
                        'turns': [row.query],
                        'category': category,
                        'source': 'chc_bench',
                        'language': 'zh',
                    }
                    self._chc_questions.append(data)
                fd.close()
        return self._chc_questions

    @property
    def cm_questions(self):
        qs = questions.IMAGE_QUESTIONS[::]
        for question in qs:
            question.setdefault('language', 'en')
            question.update({
                'category': 'vision',
                'source': 'cloud-mercato',
            })
        return qs

    @property
    def questions(self):
        if not hasattr(self, '_questions'):
            self._questions = []
            self._questions += self.mb_questions
            self._questions += self.cm_questions
            self._questions += self.om_questions
            if read_parquet is not None:
                self._questions += self.cud_questions
                self._questions += self.chc_questions
        return self._questions

    def list_questions(self):
        return self.questions

    def get_question(self, id_):
        prefix = id_.split('_', 1)[0]
        prefix = 'mb' if prefix.isdigit() else prefix
        attr_name = f"{prefix}_questions"
        questions = getattr(self, attr_name)
        for question in questions:
            if str(question['question_id']) == str(id_):
                return question
        msg = "Question '%s' does not exist." % id_
        raise errors.QuestionDoesNotExist(msg)

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
