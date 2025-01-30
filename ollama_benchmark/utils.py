import datetime
import json
import builtins
import math
import statistics

from ollama_benchmark import __version__
from ollama_benchmark import settings
from ollama_benchmark.data_manager import DataManager


def print_main():
    data = {
        'version': __version__,
    }
    for key, value in data.items():
        print(f"{key}: {value}")


data_manager = DataManager()


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return
    return statistics.mean(values)


def stdev(values):
    values = [v for v in values if v is not None]
    if not values:
        return
    if len(values) == 1:
        return .0
    return statistics.stdev(values)


def min(values):
    values = [v for v in values if v is not None]
    if not values:
        return
    return builtins.min(values)


def max(values):
    values = [v for v in values if v is not None]
    if not values:
        return
    return builtins.max(values)


def perc95(values):
    values = [v for v in values if v is not None]
    if not values:
        return
    if len(values) == 1:
        return values[0]
    perc = 95
    size = len(values)
    return sorted(values)[int(math.ceil((size * perc) / 100)) - 1]


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()


def add_tester_arguments(parser):
    parser.add_argument(
        '--do-not-pull', action="store_false", dest="pull",
        help='Do not pull the model',
    )
    parser.add_argument(
        '--do-not-prewarm', action="store_false", dest="prewarm",
        help='Run a prewarm request to load model into GPU',
    )
    return parser


def add_monitoring_arguments(parser):
    parser.add_argument(
        '--disable-monitoring', action="store_false", dest="monitoring_enabled",
    )
    parser.add_argument(
        '--monitoring-interval', type=int, default=5,
    )
    parser.add_argument(
        '--monitoring-probers', action='append'
    )
    parser.add_argument(
        '--monitoring-output', default="/dev/stderr"
    )
    return parser


def add_ollama_config_arguments(parser):
    parser.add_argument('--model', default=settings.MODEL)
    parser.add_argument(
        '--mirostat', type=int, default=0, choices=[0, 1, 2],
        help='Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)'
    )
    parser.add_argument(
        '--mirostat_eta', type=float, default=0.1,
        help='Influences how quickly the algorithm responds to feedback from the generated text. (Default: 0.1)'
    )
    parser.add_argument(
        '--mirostat_tau', type=float, default=5.0,
        help='Controls the balance between coherence and diversity of the output. (Default: 5.0)'
    )
    parser.add_argument(
        '--num_ctx', type=int, default=2048,
        help='Sets the size of the context window used to generate the next token. (Default: 2048)'
    )
    parser.add_argument(
        '--repeat_last_n', type=int, default=64,
        help='Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)'
    )
    parser.add_argument(
        '--repeat_penalty', type=float, default=1.1,
        help='Sets how strongly to penalize repetitions. (Default: 1.1)'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='The temperature of the model. (Default: 0.8)'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Sets the random number seed to use for generation. (Default: 0)'
    )
    parser.add_argument(
        '--stop', type=str, default=None,
        help='Sets the stop sequences to use. (Default: None)'
    )
    parser.add_argument(
        '--tfs_z', type=float, default=1.0,
        help='Tail free sampling is used to reduce the impact of less probable tokens from the output. (Default: 1)'
    )
    parser.add_argument(
        '--num_predict', type=int, default=-1,
        help='Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)'
    )
    parser.add_argument(
        '--top_k', type=int, default=40,
        help='Reduces the probability of generating nonsense. (Default: 40)'
    )
    parser.add_argument(
        '--top_p', type=float, default=0.9,
        help='Works together with top-k. (Default: 0.9)'
    )
    parser.add_argument(
        '--min_p', type=float, default=0.0,
        help='Alternative to the top_p, and aims to ensure a balance of quality and variety. (Default: 0.0)'
    )
    return parser


def add_test_argument(parser):
    parser = add_tester_arguments(parser)
    parser = add_monitoring_arguments(parser)
    parser = add_ollama_config_arguments(parser)
    return parser
