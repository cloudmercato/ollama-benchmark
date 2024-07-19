import logging
import statistics
from ollama_benchmark import utils

logger = logging.getLogger("ollama_benchmark")


class Tester:
    def __init__(self, client):
        self.client = client

    def run(self, question_id, model, max_turns=1):
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
            )
            logger.info('< %s', response['message']['content'])
            messages.append({
                "role": "assistant",
                "content": response['message']['content'],
            })
            responses.append(response)

        total_durations = [r['total_duration'] for r in responses]
        load_durations = [r['load_duration'] for r in responses]
        prompt_eval_durations = [r['prompt_eval_duration'] for r in responses]
        eval_counts = [r['eval_count'] for r in responses]
        eval_durations = [r['eval_duration'] for r in responses]

        result = {
            'question': question,
            'question_id': question_id,
            'responses': responses,

            'total_duration_mean': total_durations[0],
            'total_duration_stdev': 0,
            'total_duration_min': min(total_durations),
            'total_duration_max': max(total_durations),

            'load_duration_mean': load_durations[0],
            'load_duration_stdev': 0,
            'load_duration_min': min(load_durations),
            'load_duration_max': max(load_durations),

            'prompt_eval_duration_mean': prompt_eval_durations[0],
            'prompt_eval_duration_stdev': 0,
            'prompt_eval_duration_min': min(prompt_eval_durations),
            'prompt_eval_duration_max': max(prompt_eval_durations),

            'eval_count_mean': eval_counts[0],
            'eval_count_stdev': 0,
            'eval_count_min': min(eval_counts),
            'eval_count_max': max(eval_counts),

            'eval_duration_mean': eval_durations[0],
            'eval_duration_stdev': 0,
            'eval_duration_min': min(eval_durations),
            'eval_duration_max': max(eval_durations),
        }
        if len(responses) > 1:
            result.update({
                'question': question,
                'responses': responses,
                'total_duration_mean': statistics.mean(total_durations),
                'total_duration_stdev': statistics.stdev(total_durations),
                'load_duration_mean': statistics.mean(load_durations),
                'load_duration_stdev': statistics.stdev(load_durations),
                'prompt_eval_duration_mean': statistics.mean(prompt_eval_durations),
                'prompt_eval_duration_stdev': statistics.stdev(prompt_eval_durations),
                'eval_count_mean': statistics.mean(eval_counts),
                'eval_count_stdev': statistics.stdev(eval_counts),
                'eval_duration_mean': statistics.mean(eval_durations),
                'eval_duration_stdev': statistics.stdev(eval_durations),
            })
        return result
