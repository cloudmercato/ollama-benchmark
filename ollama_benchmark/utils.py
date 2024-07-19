import os
import json
from base64 import b64encode
import urllib.request
from ollama_benchmark import settings
from ollama_benchmark import questions

QUESTION_URL = 'https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl'


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
