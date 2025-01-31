import json
import time

from ollama_benchmark import utils
from ollama_benchmark import client
from ollama_benchmark import errors
from ollama_benchmark.tester import BaseTester
from ollama_benchmark.judge import prompts


class Tester(BaseTester):
    def __init__(
        self,
        judge_model,
        system_prompt=None,
        judge_host=None,
        judge_system_prompt=None,
        judge_prompt=None,
        quick_judge=False,
        ollama_judge_options=None,
        question=None,
        max_turns=1,
        load_messages=None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.judge_model = judge_model
        self.system_prompt = system_prompt
        self.ollama_judge_options = ollama_judge_options
        self.judge_system_prompt = judge_system_prompt or prompts.JUDGE_SYSTEM_PROMPT
        self.judge_prompt = judge_prompt or prompts.JUDGE_PROMPT
        self.quick_judge = quick_judge
        if judge_host:
            self.judge_client = client.OllamaClient(
                host=judge_host,
                timeout=judge_timeout,
            )
            self.judge_host = judge_host
        else:
            self.judge_client = self.client
            self.judge_host = self.ollama_host
        self.question = question
        self.max_turns = max_turns
        self.load_messages_file = load_messages

    @property
    def load_messages(self):
        if not self.load_messages_file:
            return None
        if not hasattr(self, '_load_messages'):
            try:
                self._load_messages = json.load(self.load_messages_file)
            except json.decoder.JSONDecodeError as err:
                msg = f"Invalid JSON messages file: {err}"
                raise errors.ConfigurationError(msg)
        return self._load_messages

    def check_config(self):
        for text in ('{question}', '{answer}'):
            if text not in self.judge_prompt:
                msg = f"The judge prompt must contains '{text}'"
                raise errors.ConfigurationError(msg)
        if self.load_messages_file and not self.load_messages:
            msg = "No loaded messages"
            raise errors.ConfigurationError(msg)

    def pull_model(self):
        self.client.pull_model(self.model)
        self.judge_client.pull_model(self.judge_model)

    def get_tasks(self):
        return [self.question]

    def get_tasks_kwargs(self):
        if self.load_messages:
            return [
                (i, {'question_id': 0})
                for i, msg in enumerate(self.load_messages)
            ]
        return [
            (
                task,
                {'question_id': task},
            )
            for task in self.get_tasks()
        ]

    def run_turns(self, question_id):
        data_manager = utils.DataManager()
        question = data_manager.get_question(question_id)
        responses = []
        messages = []
        if self.system_prompt is not None:
            messages.append({
                "role": "system",
                "content": self.system_prompt.read(),
            })

        for turn_id, prompt in enumerate(question['turns'][:self.max_turns]):
            self.logger.debug('turn #%s', turn_id)
            self.logger.info('> %s', prompt)
            message = {
                "role": "user",
                "content": prompt,
            }
            if 'image_urls' in question:
                message['images'] = data_manager.get_question_b64_images(question_id)
            messages.append(message)
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=self.ollama_options,
            )
            self.logger.info('< %s', response['message']['content'])
            messages.append({
                "role": "assistant",
                "content": response['message']['content'],
            })
            responses.append(response)
        return messages

    def quick_evaluate_turns(self, messages):
        judgements = []
        messages = messages[::]
        if messages and messages[0]['role'] == "system":
            system = messages.pop(0)

        while messages:
            prompt = messages.pop(0)
            answer = messages.pop(0)
            judge_prompt = self.judge_prompt.format(
                question=prompt['content'],
                answer=answer['content'],
            )
            judge_messages = [{
                "role": "system",
                "content": self.judge_system_prompt,
            }, {
                "role": "user",
                "content": judge_prompt,
            }]
            self.logger.debug('judge system > %s', self.judge_system_prompt)
            self.logger.info('judge > %s', judge_prompt)
            response = self.judge_client.chat(
                model=self.judge_model,
                messages=judge_messages,
                options=self.ollama_judge_options,
            )
            self.logger.info('judge < %s', response['message']['content'])
            judgements.append(response)
        return judgements

    def evaluate_turns(self, messages):
        judgements = []
        orig_msgs = messages[::]
        data = {}
        for rate_type, judge_prompt_temp in prompts.JUDGE_PROMPTS.items():
            messages = orig_msgs[::]
            if messages and messages[0]['role'] == "system":
                system = messages.pop(0)

            while messages:
                prompt = messages.pop(0)
                answer = messages.pop(0)
                judge_prompt = judge_prompt_temp.format(
                    question=prompt['content'],
                    answer=answer['content'],
                )
                judge_messages = [{
                    "role": "system",
                    "content": self.judge_system_prompt,
                }, {
                    "role": "user",
                    "content": judge_prompt,
                }]
                self.logger.debug('judge system > %s', self.judge_system_prompt)
                self.logger.info('judge > %s', judge_prompt)
                response = self.judge_client.chat(
                    model=self.judge_model,
                    messages=judge_messages,
                    options=self.ollama_judge_options,
                )
                self.logger.info('judge < %s', response['message']['content'])
                data[rate_type] = response
        judgements.append(data)
        return judgements

    def self_evaluate_turns(self, messages):
        messages = messages[::]
        messages.append({
            "role": "user",
            "content": prompts.SELF_ESTEEM_PROMPT,
        })
        self.logger.info('> %s', prompts.SELF_ESTEEM_PROMPT)
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=self.ollama_options,
        )
        self.logger.info('< %s', response['message']['content'])
        return response

    def _parse_quick_judgement(self, judgement):
        data = {}
        for line in judgement.splitlines():
            line = line.strip()
            if ':' not in line or not line:
                continue
            key, value = [i.strip() for i in line.split(':', 1)]
            key = key.replace('\\', '')
            if not value:
                continue
            if value[0].isdigit():
                value = float(value.split('/')[0].strip())
            data[key] = value
        try:
            dict(data)
        except ValueError:
            data = {}
        if 'total_rating' not in data:
            data = {}
        self.logger.info('Judgment values: %s', data)
        return data

    def _parse_judgement(self, judgement):
        data = {}
        for category, response in judgement.items():
            for line in response['message']['content'].splitlines():
                line = line.strip()
                if not line:
                    continue
                if not (line.startswith(category) and ':' in line):
                    continue

                key, value = [s.strip() for s in line.split(':', 1)]
                key = key.replace('\\', '')
                if not value:
                    continue
                if value[0].isdigit():
                    value = float(value.split('/')[0].strip())
                data[category] = value
                break

        values = list(data.values())
        if len(values) < 6:
            return {}
        data['total_rating'] = sum(values)
        return data

    def run(self, question_id):
        if self.load_messages:
            messages = self.load_messages[question_id]
            message_duration = 0
            self_judgement_content = {}
        else:
            t0 = time.time()
            messages = self.run_turns(question_id)
            message_duration = time.time() - t0

            self_judgement = self.self_evaluate_turns(messages)
            self_judgement_content = self._parse_quick_judgement(self_judgement['message']['content'])

            if self.ollama_host == self.judge_host:
                self.client.unload(self.model)

        t0 = time.time()
        if self.quick_judge:
            judgements = self.quick_evaluate_turns(messages)
            judge_duration = time.time() - t0

            judgements_content = [
                self._parse_quick_judgement(j['message']['content'])
                for j in judgements
            ]
        else:
            judgements = self.evaluate_turns(messages)
            judge_duration = time.time() - t0

            judgements_content = [
                self._parse_judgement(j)
                for j in judgements
            ]

        result = {
            'ollama_version': self.client.get_version(),
            'messages': messages,
            'judgements': judgements_content,
            'self_judgement': self_judgement_content,
            'message_duration': message_duration,
            'judge_duration': judge_duration,
            'work_duration': message_duration + judge_duration,
        }
        return result
