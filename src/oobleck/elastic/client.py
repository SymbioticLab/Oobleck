import etcd3
import rpyc

from etcd3 import Etcd3Client
from rpyc import Connection

from ast import literal_eval
from typing import Optional, Dict, List, Tuple

from oobleck.elastic import constants


class ElasticAgentClientMixin(object):
    """A mixin that is used by agent processes
    to query data required for training.
    Communicate with etcd and master process via rpyc.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        master_ip = kwargs.get("master_ip", "localhost")
        self.etcd: Etcd3Client = etcd3.client(master_ip)
        self.client: Connection = rpyc.connect()
        self.agent_id = self.client._channel.stream.sock.getsockname()
        self.client.root.register_agent(self.agent_id)

    def __del__(self):
        # cancel all etcd wathces here
        self.etcd.close()
        self.client.close()

    def wait_for_training_start(self):
        """Wait until get a "oobleck/training_in_progress" etcd event.
        It is a blocking operation.
        """
        iterator, cancel = self.etcd.watch(constants.OOBLECK_TRAINING_IN_PROGRESS)
        for event in iterator:
            if event._event.kv.value.decode() != str(True):
                continue

            cancel()


class ElasticWorkerClientMixin(object):
    """A mixin that is used by worker processes
    to query data required for training.
    Only communicate with etcd.
    """

    def __init__(self):
        super().__init__()

    def get_world_info(self) -> Optional[Dict[str, List[int]]]:
        result = self.etcd.get(constants.OOBLECK_WORLD_INFO)[0]
        if result == None:
            return None
        return literal_eval(result)

    def get_torch_master_info(self) -> Tuple[str, int]:
        master_info = self.etcd.get(constants.OOBLECK_TORCH_MASTER_INFO)
        if master_info[0] == None:
            raise ValueError("No master info in etcd.")
        return literal_eval(master_info[0])


class ElasticClientMonitorMixin(object):
    def __init__(self):
        super().__init__()

        self.etcd = etcd3.client()

    def on_reconfiguration_requested(self, event):
        """
        We lose some other GPU, thus topology reconfiguration is needed.
        Pause training, reconfigure topology, and resume training.

        Args:
            event (_type_): an event of reconfiguration_needed
        """
