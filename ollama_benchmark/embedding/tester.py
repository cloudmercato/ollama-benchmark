import time
import random
from ollama_benchmark import utils
from ollama_benchmark import errors
from ollama_benchmark.tester import BaseTester
from ollama_benchmark.embedding import samples


class Tester(BaseTester):
    def __init__(
        self,
        langs,
        sizes,
        num_tasks,
        num_inputs,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.langs = langs
        self.sizes = sizes
        self.num_tasks = num_tasks
        self.num_inputs = num_inputs
        self.sample_manager = samples.SampleManager()
        self.sample_refs = self.sample_manager.get_samples_refs(
            langs=langs,
            sizes=sizes,
        )

    def get_tasks(self):
        for _ in range(self.num_tasks):
            sample = random.sample(
                self.sample_refs,
                self.num_inputs,
            )
            yield sample

    def get_tasks_kwargs(self):
        return (
            (
                task,
                {
                    'input_': [
                        self.sample_manager.get_sample(*s)
                        for s in task
                     ]
                }
            )
            for task in self.get_tasks()
        )

    def run(self, input_):
        self.logger.info('> %s', input_)
        t0 = time.time()
        result = {'errors': 0}
        try:
            self.client.embed(
                model=self.model,
                input_=input_,
                options=self.ollama_options,
            )
        except errors.ClientError as err:
            self.logger.debug('Client error : %s', err.args[0])
            result['error_duration'] = time.time() - t0
            result["errors"] += 1
        result['duration'] = time.time() - t0
        return result
