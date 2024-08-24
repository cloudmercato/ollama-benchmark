import logging

logger = logging.getLogger("ollama_benchmark")


class BaseHack:
    system_prompt = None
    questions = None
    answers = None
    mode = 'dialog'

    def __init__(
        self,
        client,
        model,
        ollama_options,
        judge_model,
        mode=None,
    ):
        self.client = client
        self.model = model
        self.ollama_options = ollama_options
        self.judge_model = judge_model
        self.mode = mode or self.mode
        self.logger = logger

    def test_output(self, output):
        raise NotImplementedError("Not yet")

    def _dialog(self):
        messages = []
        if self.system_prompt is not None:
            messages.append({
                'role': 'system',
                'content': self.system_prompt,
            })
        for question in self.questions:
            messages.append({
                "role": "user",
                "content": question,
            })
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=self.ollama_options,
            )
            messages.append({
                "role": "assistant",
                "content": response['message']['content'],
            })

        ok = self.test_output(messages[-1]['content'])
        return messages, ok

    def _completion(self):
        messages = []
        if self.system_prompt is not None:
            messages.append({
                'role': 'system',
                'content': self.system_prompt,
            })
        for role, content in self.messages:
            messages.append({
                'role': role,
                'content': content,
            })
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=self.ollama_options,
        )
        messages.append({
            "role": "assistant",
            "content": response['message']['content'],
        })

        ok = self.test_output(messages[-1]['content'])
        return messages, ok

    def run(self):
        if self.mode == 'dialog':
            return self._dialog()
        elif self.mode == 'completion':
            return self._completion()
        else:
            msg = f"Incorrect hack mode '{self.mode}'"
            raise Exception(msg)
