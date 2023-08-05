import asyncio
import getpass
import logging

import simple_parsing

import oobleck.elastic.message_util as message_util
from oobleck.elastic.training_util import DistributedJobConfiguration, OobleckArguments

logger = logging.getLogger(__name__)


class OobleckClient:
    def __init__(self):
        pass

    async def connect_to_master(self, master_ip: str, master_port: int):
        logger.info("Connecting to master...")
        self.conn_ = await asyncio.wait_for(
            asyncio.open_connection(master_ip, master_port),
            timeout=message_util.TIMEOUT,
        )

    async def request_job_launch(self, job_config: DistributedJobConfiguration):
        reader, writer = self.conn_
        await message_util.send_request_type(
            writer, message_util.RequestType.LAUNCH_JOB
        )
        await message_util.send(
            writer, job_config, need_pickle=True, drain=True, close=False
        )
        response, request_type = await message_util.recv_response(reader)
        assert request_type == message_util.RequestType.LAUNCH_JOB
        if response == message_util.Response.SUCCESS:
            logger.info("Job launch request is accepted by the master.")
        else:
            logger.error("Job launch request is rejected by the master.")
            raise RuntimeError("Job launch request is rejected by the master.")


async def main(args: DistributedJobConfiguration):
    client = OobleckClient()
    await client.connect_to_master(args.master_ip, args.master_port)
    await client.request_job_launch(args)


"""
Send a training job run request to the master.
Information to send
1. List of node IPs where agents and workers will be running
2. Job configuration (oobleck.elastic.training_util.OobleckArguments)
"""
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args: DistributedJobConfiguration = simple_parsing.parse(
        DistributedJobConfiguration, add_config_path_arg=True
    )

    if not args.username:
        args.username = getpass.getuser()
    asyncio.run(main(args))
