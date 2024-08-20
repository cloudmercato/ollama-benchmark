import importlib
import time
from ollama_benchmark import utils
from ollama_benchmark import errors
from ollama_benchmark.tester import BaseTester

HACK_IDS = (
    'naive_last_word',
    'emulated_code',
    'emulated_code_escape',
    'hangman',
)


def get_hack_class(hack_id):
    module_path = f'ollama_benchmark.hack.hacks.{hack_id}'
    module = importlib.import_module(module_path)
    driver_class = getattr(module, 'Hack')
    return driver_class


class Tester(BaseTester):
    def __init__(
        self,
        hacks,
        judge_model='llama3',
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.hacks = hacks or HACK_IDS
        self.judge_model = judge_model

    def pull_model(self):
        self.client.pull_model(self.model)
        if self.model != self.judge_model:
            self.client.pull_model(self.judge_model)

    def get_tasks(self):
        return self.hacks

    def get_tasks_kwargs(self):
        return [
            (
                task,
                {'hack_id': task},
            )
            for task in self.get_tasks()
        ]

    def run(self, hack_id):
        hack_class = get_hack_class(hack_id)
        hack = hack_class(
            client=self.client,
            model=self.model,
            ollama_options=self.ollama_options,
            judge_model=self.judge_model,
        )

        t0 = time.time()
        messages, ok = hack.run()
        duration = time.time() - t0

        result = {
            'ollama_version': self.client.get_version(),
            'hack_id': hack_id,
            'messages': messages,
            'ok': ok,
            'duration': duration,
        }
        return result
