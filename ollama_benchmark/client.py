from ollama import Client
from ollama_benchmark import settings
import httpx


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

    def get_version(self):
        response = self.client._request('GET', '/api/version')
        if response.status_code >= 300:
            raise ClientError(response.status_code)
        return response.json()['version']

    def list_running_models(self):
        response = self.client._request('GET', '/api/ps')
        if response.status_code >= 300:
            raise ClientError(response.status_code)
        return response.json()['models']

    def load(self, model, keep_alive=None):
        try:
            self.client.generate(
                model=model,
                keep_alive=keep_alive,
            )
        except httpx.ConnectError as err:
            raise ClientError(err)

    def unload(self, model):
        self.client.generate(
            model=model,
            keep_alive=0,
        )

    def unload_all(self):
        models = self.list_running_models()
        for model in models:
            self.unload(model['name'])

    def embed(self, model, input_, options):
        response = self.client._request('POST', '/api/embed', json={
            'model': model,
            'input': input_,
            'options': options,
        })
        if response.status_code >= 300:
            raise ClientError(response.status_code)
        return response.json()
