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
Now here are the question and answer:

Question: {question}
Answer: {answer}
"""
RELEVANCE_PROMPT = """
Up to 5 points can you rate only the relevance of the answer.
Here are the question and answer:

Question: {question}
Answer: {answer}
"""
COHERENCE_PROMPT = """
Up to 4 points can you rate only the coherence of the answer.
Here are the question and answer:

Question: {question}
Answer: {answer}
"""
QUALITY_PROMPT = """
Up to 4 points can you rate only the quality of the answer.
Here are the question and answer:

Question: {question}
Answer: {answer}
"""
LANGUAGE_PROMPT = """
Up to 3 points can you rate only the language of the answer.
Here are the question and answer:

Question: {question}
Answer: {answer}
"""
ORIGINALITY_PROMPT = """
Up to 2 points can you rate only the orignality of the answer.
Here are the question and answer:

Question: {question}
Answer: {answer}
"""
NEUTRALITY_PROMPT = """
Up to 2 points can you rate only the neutrality of the answer.
Here are the question and answer:

Question: {question}
Answer: {answer}
"""
JUDGE_PROMPTS = {
    'relevance': RELEVANCE_PROMPT,
    'coherence': COHERENCE_PROMPT,
    'quality': QUALITY_PROMPT,
    'language': LANGUAGE_PROMPT,
    'originality': ORIGINALITY_PROMPT,
    'neutrality': NEUTRALITY_PROMPT,
}

SELF_ESTEEM_PROMPT = """
From 0 to 20, how do you evaluate the quality of your answer ?

Provide your feedback as a key:value text file just the result, no other text.Here's an example of output:

    evaluation:(your rationale for the rating, as a text)
    total_rating:5/20
"""
