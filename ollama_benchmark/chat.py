import time
from ollama_benchmark import utils
from ollama_benchmark import client


def make_args(subparsers):
    parser = subparsers.add_parser("chat", help="Evaluate performance while chatting")
    utils.add_ollama_config_arguments(parser)
    utils.add_tester_arguments(parser)

FUNCS = {
    'quit': lambda p: exit(0),
    'q': lambda p: exit(0),
}
FUNCS.update({
    'q': FUNCS['q'],
    'exit': FUNCS['q'],
    'e': FUNCS['q'],
})

def main(args):
    cl = client.OllamaClient(
        host=args.host,
        timeout=args.timeout,
    )

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
        print('time: ', time.time() - t0)
        messages.append({
            "role": "assistant",
            "content": response['message']['content'],
        })

    messages = []
    while 1:
        try:
            loop(messages)
        except KeyboardInterrupt:
            break
        except client.OllamaConnectionError as err:
            print(err, '\n')
