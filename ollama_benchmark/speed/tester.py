from ollama_benchmark import utils
from ollama_benchmark import errors
from ollama_benchmark.tester import BaseTester


class Tester(BaseTester):
    def __init__(
        self,
        questions,
        max_turns=1,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.questions = questions
        self.max_turns = max_turns

    def get_tasks(self):
        questions = self.questions
        if 'all' in questions:
            questions = [
                q['question_id']
                for q in utils.data_manager.list_questions()
            ]
        return questions

    def get_tasks_kwargs(self):
        return [
            (
                task,
                {'question_id': task},
            )
            for task in self.get_tasks()
        ]

    def run(self, question_id):
        data_manager = utils.DataManager()
        question = data_manager.get_question(question_id)

        responses = []
        messages = []
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
            response = self.client.client.chat(
                model=self.model,
                messages=messages,
                options=self.ollama_options,
            )
            self.logger.info('< %s', response['message']['content'])

            if self.tokenizer_model:
                try:
                    response['prompt_eval_count'] = self.count_tokens(messages)
                    response['eval_count'] = self.count_tokens([response['message']])
                except errors.TokenizerError as err:
                    self.logger.warning("Cannot tokenize: %s", err)
                    self.tokenizer_model = None

            messages.append({
                "role": "assistant",
                "content": response['message']['content'],
            })
            responses.append(response)

        total_durations = [(r['total_duration']/10**6) for r in responses]
        load_durations = [(r['load_duration']/10**6) for r in responses]
        prompt_eval_counts = [r['prompt_eval_count'] for r in responses
                              if 'prompt_eval_count' in r]
        prompt_eval_durations = [(r['prompt_eval_duration']/10**6) for r in responses
                                 if 'prompt_eval_duration' in r]
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
            'total_duration_min': utils.min(total_durations),
            'total_duration_max': utils.max(total_durations),

            'load_duration_mean': utils.mean(load_durations),
            'load_duration_stdev': utils.stdev(load_durations),
            'load_duration_min': utils.min(load_durations),
            'load_duration_max': utils.max(load_durations),

            'prompt_eval_duration_mean': utils.mean(prompt_eval_durations),
            'prompt_eval_duration_stdev': utils.stdev(prompt_eval_durations),
            'prompt_eval_duration_min': utils.min(prompt_eval_durations),
            'prompt_eval_duration_max': utils.max(prompt_eval_durations),

            'prompt_eval_rate_mean': utils.mean(prompt_eval_rates),
            'prompt_eval_rate_stdev': utils.stdev(prompt_eval_rates),
            'prompt_eval_rate_min': utils.min(prompt_eval_rates),
            'prompt_eval_rate_max': utils.max(prompt_eval_rates),

            'eval_count_mean': utils.mean(eval_counts),
            'eval_count_stdev': utils.stdev(eval_counts),
            'eval_count_min': utils.min(eval_counts),
            'eval_count_max': utils.max(eval_counts),

            'prompt_eval_count_mean': utils.mean(prompt_eval_counts),
            'prompt_eval_count_stdev': utils.stdev(prompt_eval_counts),
            'prompt_eval_count_min': utils.min(prompt_eval_counts),
            'prompt_eval_count_max': utils.max(prompt_eval_counts),

            'eval_duration_mean': utils.mean(eval_durations),
            'eval_duration_stdev': utils.stdev(eval_durations),
            'eval_duration_min': utils.min(eval_durations),
            'eval_duration_max': utils.max(eval_durations),

            'eval_rate_mean': utils.mean(eval_rates),
            'eval_rate_stdev': utils.stdev(eval_rates),
            'eval_rate_min': utils.min(eval_rates),
            'eval_rate_max': utils.max(eval_rates),
        }
        if len(responses) > 1:
            result.update({
                'question': question,
            })
        return result
