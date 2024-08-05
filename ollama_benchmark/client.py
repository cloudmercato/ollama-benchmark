from ollama import Client
from ollama_benchmark import settings


class ClientError(Exception):
    pass


class OllamaClient:
    err_class = ClientError

    def __init__(
        self,
        host=settings.HOST,
        follow_redirects=True,
        timeout=settings.TIMEOUT,
    ):
        self.host = host
        self.follow_redirects = follow_redirects
        self.timeout = timeout

    @property
    def client(self):
        if not hasattr(self, '_client'):
            self._client = Client(
                host=self.host,
                follow_redirects=self.follow_redirects,
                timeout=self.timeout,
            )
        return self._client

    def embed(self, model, input_, options):
        response = self.client._request('POST', '/api/embed', json={
            'model': model,
            'input': input_,
            'options': options,
        })
        if response.status_code >= 300:
            raise ClientError(response.status_code)
        return response.json()
