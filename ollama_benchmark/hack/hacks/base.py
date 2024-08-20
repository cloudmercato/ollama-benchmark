class BaseHack:
    system_prompt = None
    questions = None
    answers = None

    def __init__(
        self,
        client,
        model,
        ollama_options,
        judge_model,
    ):
        self.client = client
        self.model = model
        self.ollama_options = ollama_options
        self.judge_model = judge_model

    def test_output(self, output):
        raise NotImplementedError("Not yet")

    def run(self):
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
