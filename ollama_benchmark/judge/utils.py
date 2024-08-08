"""
Reference:
    https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/download_mt_bench_pregenerated.py

"""
import os
import json
import urllib.request

FILENAMES = [
    "data/mt_bench/model_answer/alpaca-13b.jsonl",
    "data/mt_bench/model_answer/baize-v2-13b.jsonl",
    "data/mt_bench/model_answer/chatglm-6b.jsonl",
    "data/mt_bench/model_answer/claude-instant-v1.jsonl",
    "data/mt_bench/model_answer/claude-v1.jsonl",
    "data/mt_bench/model_answer/dolly-v2-12b.jsonl",
    "data/mt_bench/model_answer/falcon-40b-instruct.jsonl",
    "data/mt_bench/model_answer/fastchat-t5-3b.jsonl",
    "data/mt_bench/model_answer/gpt-3.5-turbo.jsonl",
    "data/mt_bench/model_answer/gpt-4.jsonl",
    "data/mt_bench/model_answer/gpt4all-13b-snoozy.jsonl",
    "data/mt_bench/model_answer/guanaco-33b.jsonl",
    "data/mt_bench/model_answer/guanaco-65b.jsonl",
    "data/mt_bench/model_answer/h2ogpt-oasst-open-llama-13b.jsonl",
    "data/mt_bench/model_answer/koala-13b.jsonl",
    "data/mt_bench/model_answer/llama-13b.jsonl",
    "data/mt_bench/model_answer/mpt-30b-chat.jsonl",
    "data/mt_bench/model_answer/mpt-30b-instruct.jsonl",
    "data/mt_bench/model_answer/mpt-7b-chat.jsonl",
    "data/mt_bench/model_answer/nous-hermes-13b.jsonl",
    "data/mt_bench/model_answer/oasst-sft-4-pythia-12b.jsonl",
    "data/mt_bench/model_answer/oasst-sft-7-llama-30b.jsonl",
    "data/mt_bench/model_answer/palm-2-chat-bison-001.jsonl",
    "data/mt_bench/model_answer/rwkv-4-raven-14b.jsonl",
    "data/mt_bench/model_answer/stablelm-tuned-alpha-7b.jsonl",
    "data/mt_bench/model_answer/tulu-30b.jsonl",
    "data/mt_bench/model_answer/vicuna-13b-v1.3.jsonl",
    "data/mt_bench/model_answer/vicuna-33b-v1.3.jsonl",
    "data/mt_bench/model_answer/vicuna-7b-v1.3.jsonl",
    "data/mt_bench/model_answer/wizardlm-13b.jsonl",
    "data/mt_bench/model_answer/wizardlm-30b.jsonl",
    "data/mt_bench/model_judgment/gpt-4_single.jsonl",
    "data/mt_bench/model_judgment/gpt-4_pair.jsonl",
]
HG_FILE_URL = "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/"
DEFAULT_DATA_DIR = '/tmp/ollama/'


def download_files(to=DEFAULT_DATA_DIR):
    for name in FILENAMES:
        dest_file = os.path.join(to, name)
        url = HG_FILE_URL + name

        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        urllib.request.urlretrieve(url, dest_file)


def load_questions(question_file):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions
