class OllamaBenchmarkError(Exception):
    pass


class ClientError(OllamaBenchmarkError):
    pass


class OllamaConnectionError(ClientError):
    pass


class OllamaModelUnfoundError(ClientError):
    pass


class OllamaModelDoesNotExistError(ClientError):
    pass


class OllamaTimeoutError(ClientError):
    pass
