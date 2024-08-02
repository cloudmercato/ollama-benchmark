import logging
from ollama_benchmark import utils

logger = logging.getLogger("ollama_benchmark")


class Tester:
    def __init__(self, client):
        self.client = client

    def pull_model(self, model):
        logger.info("Pulling model %s", model)
        self.client.client.pull(model)
        logger.debug("Pulled model %s", model)

    def prewarm(self, model):
        prompt = "Hello world"
        messages = [{
            "role": "user",
            "content": prompt,
        }]
        logger.debug('Prewarm request > %s', prompt)
        response = self.client.client.chat(
            model=model,
            messages=messages,
        )
        logger.info('< %s', response['message']['content'])

    def run(self, question_id, model, max_turns=1, options=None):
        data_manager = utils.DataManager()
        question = data_manager.get_question(question_id)

        responses = []
        messages = []
        for turn_id, prompt in enumerate(question['turns'][:max_turns]):
            logger.debug('turn #%s', turn_id)
            logger.info('> %s', prompt)
            message = {
                "role": "user",
                "content": prompt,
            }
            if 'image_urls' in question:
                message['images'] = data_manager.get_question_b64_images(question_id)
            messages.append(message)
            response = self.client.client.chat(
                model=model,
                messages=messages,
                options=options,
            )
            logger.info('< %s', response['message']['content'])
            messages.append({
                "role": "assistant",
                "content": response['message']['content'],
            })
            responses.append(response)

        total_durations = [(r['total_duration']/10**6) for r in responses]
        load_durations = [(r['load_duration']/10**6) for r in responses]
        prompt_eval_counts = [r['prompt_eval_count'] for r in responses]
        prompt_eval_durations = [(r['prompt_eval_duration']/10**6) for r in responses]
        prompt_eval_rates = [
            (c/(d/1000)) for c, d in zip(prompt_eval_counts, prompt_eval_durations)
        ]
        eval_counts = [r['eval_count'] for r in responses]
        eval_durations = [(r['eval_duration']/10**6) for r in responses]
        eval_rates = [
            (c/(d/1000)) for c, d in zip(eval_counts, eval_durations)
        ]

        result = {
            'question': question,
            'question_id': question_id,
            'responses': responses,

            'total_durations': total_durations,

            'total_duration_mean': utils.mean(total_durations),
            'total_duration_stdev': utils.stdev(total_durations),
            'total_duration_min': min(total_durations),
            'total_duration_max': max(total_durations),

            'load_duration_mean': utils.mean(load_durations),
            'load_duration_stdev': utils.stdev(load_durations),
            'load_duration_min': min(load_durations),
            'load_duration_max': max(load_durations),

            'prompt_eval_duration_mean': utils.mean(prompt_eval_durations),
            'prompt_eval_duration_stdev': utils.stdev(prompt_eval_durations),
            'prompt_eval_duration_min': min(prompt_eval_durations),
            'prompt_eval_duration_max': max(prompt_eval_durations),

            'prompt_eval_rate_mean': utils.mean(prompt_eval_rates),
            'prompt_eval_rate_stdev': utils.stdev(prompt_eval_rates),
            'prompt_eval_rate_min': min(prompt_eval_rates),
            'prompt_eval_rate_max': max(prompt_eval_rates),

            'eval_count_mean': utils.mean(eval_counts),
            'eval_count_stdev': utils.stdev(eval_counts),
            'eval_count_min': min(eval_counts),
            'eval_count_max': max(eval_counts),

            'prompt_eval_count_mean': utils.mean(prompt_eval_counts),
            'prompt_eval_count_stdev': utils.stdev(prompt_eval_counts),
            'prompt_eval_count_min': min(prompt_eval_counts),
            'prompt_eval_count_max': max(prompt_eval_counts),

            'eval_duration_mean': utils.mean(eval_durations),
            'eval_duration_stdev': utils.stdev(eval_durations),
            'eval_duration_min': min(eval_durations),
            'eval_duration_max': max(eval_durations),

            'eval_rate_mean': utils.mean(eval_rates),
            'eval_rate_stdev': utils.stdev(eval_rates),
            'eval_rate_min': min(eval_rates),
            'eval_rate_max': max(eval_rates),
        }
        if len(responses) > 1:
            result.update({
                'question': question,
            })
        return result
