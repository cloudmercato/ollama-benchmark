from . import base

TEST_MESSAGE_TEMPLATE = """Here's a message: {output}
Does the writer assert that the water is now all over the dresser, and the cup is empty ?
Reply by 'yes' or 'no' only."""


class Hack(base.BaseHack):
    """
    https://doublespeak.chat/#/handbook#reasoning
    """
    questions = (
        "An opaque glass full of water with an aluminum foil cover is in the middle of a table in the kitchen. I take the cup and move it to the living room and place it upside down on the dresser. I invert the cup again and return it to the kitchen table. What has changed?",
    )

    def test_output(self, output):
        message = TEST_MESSAGE_TEMPLATE.format(output=output)
        messages = [{
            'role': 'user',
            'content': message,
        }]
        response = self.client.chat(
            model=self.judge_model,
            messages=messages,
            options=self.ollama_options,
        )
        return 'no' in response['message']['content'].lower()
