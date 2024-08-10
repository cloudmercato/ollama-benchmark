import time
import logging
import platform
from concurrent.futures import ThreadPoolExecutor
from ollama_benchmark import client
from ollama_benchmark import errors

try:
    from probes import ProbeManager
    has_probes = True
except ImportError:
    has_probes = False


class BaseTester:
    def __init__(
        self,
        host,
        model,
        timeout=None,
        ollama_options=None,
        max_workers=1,
        pull=True,
        prewarm=True,
        monitoring_enabled=True,
        monitoring_probers=None,
        monitoring_interval=5,
    ):
        self.client = client.OllamaClient(
            host=host,
            timeout=timeout,
        )
        self.logger = logging.getLogger("ollama_benchmark")
        self.model = model
        self.max_workers = max_workers
        self.ollama_options = ollama_options or {}
        self.pull = pull
        self.prewarm_ = prewarm
        self.monitoring_enabled = monitoring_enabled
        self.monitoring_probers = monitoring_probers
        self.monitoring_interval = monitoring_interval

    def check_config(self):
        pass

    def pull_model(self):
        self.client.pull_model(self.model)

    def start_monitoring(self, probers, interval=5):
        if not probers:
            probers = [
                'probes.probers.system.CpuProber',
                'probes.probers.system.MemoryProber',
            ]
            sys_plat = platform.system()
            if sys_plat == 'Darwin':
                probers += ['probes.probers.macos.MacosProber']
        self.probe_manager = ProbeManager(
            interval=interval,
            probers=probers,
        )
        self.probe_manager.start()

    def stop_monitoring(self):
        self.probe_manager.stop()

    def get_monitoring_results(self):
        return self.probe_manager.get_results(),

    def prewarm(self):
        try:
            self.client.prewarm(self.model)
        except errors.OllamaTimeoutError as err:
            self.logger.warning("Error in prewarm: %s", err)

    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def get_tasks(self):
        raise NotImplementedError()

    def get_tasks_kwargs(self):
        raise NotImplementedError()

    def run_suite(self):
        if self.pull:
            self.pull_model()
        if self.prewarm_:
            self.prewarm()

        if self.monitoring_enabled:
            try:
                self.start_monitoring(
                    self.monitoring_probers,
                    self.monitoring_interval,
                )
            except Exception as err:
                self.logger.warning('Error with monitoring: %s', err)
                self.monitoring_enabled = False

        tasks = self.get_tasks_kwargs()
        futures = []
        pool_kwargs = {'max_workers': self.max_workers}
        with ThreadPoolExecutor(**pool_kwargs) as executor:
            t0 = time.time()
            for task, task_kw in tasks:
                future = executor.submit(
                    self.run,
                    **task_kw
                )
                self.logger.info('Submitted %s', task)
                futures.append(future)
            results = [
                future.result()
                for future in futures
            ]
            real_duration = (time.time() - t0)

        run_results = {
            'results': results,
            'real_duration': real_duration,
        }

        if self.monitoring_enabled:
            self.stop_monitoring()
            run_results['monitoring'] = self.get_monitoring_results()

        return run_results
