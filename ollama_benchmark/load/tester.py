import time
from ollama_benchmark.tester import BaseTester


class Tester(BaseTester):
    def __init__(
        self,
        models,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.models = models

    def get_tasks(self):
        for model in self.models:
            yield model

    def get_tasks_kwargs(self):
        return (
            (task, {'model': task})
            for task in self.get_tasks()
        )

    def pull_model(self):
        for model in self.models:
            self.client.pull_model(model)

    def prewarm(self):
        self.client.unload_all()

    def run(self, model):
        self.logger.info('Load model %s', model)
        t0 = time.time()
        result = {'error': None}
        try:
            self.client.load(model, keep_alive=-1)
        except self.client.err_class as err:
            self.logger.debug('Client error : %s', err.args[0])
            result["error"] = err
        result['duration'] = time.time() - t0

        try:
            self.client.unload(model)
        except self.client.err_class as err:
            self.logger.debug('Client error : %s', err.args[0])
            result["error"] = err

        return result
