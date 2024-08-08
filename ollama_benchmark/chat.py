import time
import logging
from ollama_benchmark import utils
from ollama_benchmark import errors
from ollama_benchmark import client

logger = logging.getLogger('ollama_benchmark')


def make_args(subparsers):
    parser = subparsers.add_parser(
        "chat",
        help="Evaluate performance while chatting"
    )
    parser.add_argument('--unload-after', action="store_true")
    utils.add_ollama_config_arguments(parser)
    utils.add_tester_arguments(parser)


def print_help(func_name):
    print()
    print('Console help:')
    for func_name in FUNCS:
        print(f"\\{func_name}")
        aliases = []
        for alias in FUNCS_ALIASES:
            if FUNCS_ALIASES[alias] == func_name:
                aliases.append(alias)
        print('  \\' + ' \\'.join(aliases))

    print()


FUNCS = {
    'quit': lambda p: exit(0),
    'help': print_help,
}
FUNCS_ALIASES = {
    'q': 'quit',
    'exit': 'quit',
    'e': 'quit',
    'h': 'help',
    '?': 'help',
}


def main(args):
    cl = client.OllamaClient(
        host=args.host,
        timeout=args.timeout,
    )

    if args.prewarm:
        logger.info('Loading model %s', args.model)
        t0 = time.time()
        cl.load(args.model)
        duration = time.time() - t0
        print('load_model_duration: ', duration)

    ollama_options = {
        'mirostat': args.mirostat,
        'mirostat_eta': args.mirostat_eta,
        'mirostat_tau': args.mirostat_tau,
        'num_ctx': args.num_ctx,
        'repeat_last_n': args.repeat_last_n,
        'repeat_penalty': args.repeat_penalty,
        'temperature': args.temperature,
        'seed': args.seed,
        'stop': args.stop,
        'tfs_z': args.tfs_z,
        'num_predict': args.num_predict,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'min_p': args.min_p,
    }

    def loop(messages):
        prompt = input("> ")

        if not prompt.strip():
            return

        if prompt.startswith('\\'):
            func_name = prompt[1:]
            if func_name in FUNCS_ALIASES:
                func_name = FUNCS_ALIASES[func_name]
            if func_name in FUNCS:
                FUNCS[func_name](func_name)
            return

        message = {
            "role": "user",
            "content": prompt,
        }
        messages.append(message)
        t0 = time.time()
        response = cl.chat(
            model=args.model,
            messages=messages,
            options=ollama_options,
        )
        print('<', response['message']['content'])
        print('total_duration: ', response['total_duration']/10**9)
        print('load_duration: ', response['load_duration']/10**9)
        print('prompt_eval_count: ', response['prompt_eval_count'])
        print('prompt_eval_duration: ', response['prompt_eval_duration']/10**9)
        print('eval_count: ', response['eval_count'])
        print('eval_duration: ', response['eval_duration']/10**9)
        print('request_duration: ', time.time() - t0)
        messages.append({
            "role": "assistant",
            "content": response['message']['content'],
        })

    t0 = time.time()
    messages = []
    try:
        while 1:
            try:
                loop(messages)
            except errors.OllamaConnectionError as err:
                print(err, '\n')
    except KeyboardInterrupt:
        print('')
    duration = time.time() - t0
    print('whole_duration: ', duration)

    if args.unload_after:
        cl.unload(args.model)
