import json
import time
from ollama_benchmark import utils
from ollama_benchmark import errors
from ollama_benchmark.tester import BaseTester

JUDGE_SYSTEM_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer on a scale of 0 to 20, where 0 means that the system_answer is not helpful at all, and 20 means that the system_answer completely and helpfully addresses the user_question.

Here is an example of rating with comment:
5: The system_answer is terrible: completely irrelevant to the question asked, or very partial
8: The system_answer is mostly not helpful: misses some key aspects of the question
12: The system_answer is mostly helpful: provides support, but still could be improved
18: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

Here's a methodology for rating:
    - 5 points on relevance: Does the response directly and fully answer the question asked? Is it consistent with the context?
    - 4 points on coherence: Is the response logical and coherent in itself? Are there no contradictions or absurdities?
    - 4 points on quality of argumentation: Is the response supported by clear and relevant arguments? Does it use concrete examples?
    - 3 points on language: Is the response expressed in clear, precise language appropriate to the context? Is the spelling, grammar or code syntax correct?
    - 2 points on originality: Does the response bring an original point of view or a new perspective on the question?
    - 2 points on neutrality: Is the response objective and neutral? Does it avoid personal biases or subjective opinions?
For a total_rating of 20 points. Never gives more points than what described above.

Provide your feedback as a key:value text, easy for parsing, with no other text than the output. Here's an example of output:

    relevance:1/5
    coherence:2/4
    quality:3/4
    language:1/3
    originality:1/2
    neutrality:1/2
    total_rating:7/20
    evaluation:(your rationale for the rating, as a text)
    feedback:(How you think it could be improved)
"""
JUDGE_PROMPT = """
Now here are the question and answer.

Question: {question}
Answer: {answer}
"""
SELF_ESTEEM_PROMPT = """
From 0 to 20, how do you evaluate the quality of your answer ?

Provide your feedback as a key:value text file just the result, no other text.Here's an example of output:

    evaluation:(your rationale for the rating, as a text)
    total_rating:5/20
"""


class Tester(BaseTester):
    def __init__(
        self,
        judge_model,
        system_prompt=None,
        judge_system_prompt=None,
        judge_prompt=None,
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
        self.judge_system_prompt = judge_system_prompt or JUDGE_SYSTEM_PROMPT
        self.judge_prompt = judge_prompt or JUDGE_PROMPT
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
        if self.model != self.judge_model:
            self.client.pull_model(self.judge_model)

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

    def evaluate_turns(self, messages):
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
            response = self.client.chat(
                model=self.judge_model,
                messages=judge_messages,
                options=self.ollama_judge_options,
            )
            self.logger.info('judge < %s', response['message']['content'])
            judgements.append(response)
        return judgements

    def self_evaluate_turns(self, messages):
        messages = messages[::]
        messages.append({
            "role": "user",
            "content": SELF_ESTEEM_PROMPT,
        })
        self.logger.info('> %s', SELF_ESTEEM_PROMPT)
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=self.ollama_options,
        )
        self.logger.info('< %s', response['message']['content'])
        return response

    def _parse_judgement(self, judgement):
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
            self_judgement_content = self._parse_judgement(self_judgement['message']['content'])

            if self.model == self.judge_model:
                self.client.unload(self.model)

        t0 = time.time()
        judgements = self.evaluate_turns(messages)
        judge_duration = time.time() - t0

        judgements_content = [
            self._parse_judgement(j['message']['content'])
            for j in judgements
        ]

        # Capture messages to a JSON file
        try:
            file_name = f"messages_{question_id}.json"
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(messages, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Messages saved to {file_name}")
        except Exception as e:
            self.logger.error(f"Failed to save messages: {e}")

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
