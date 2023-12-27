import asyncio
import getpass
import logging

import simple_parsing as sp

import oobleck.elastic.message_util as message_util
from oobleck.elastic.training_util import (
    DistributedArguments,
    JobArguments,
    ModelArguments,
    OobleckArguments,
)

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

    async def request_job_launch(self, args: OobleckArguments):
        reader, writer = self.conn_
        await message_util.send_request_type(
            writer, message_util.RequestType.LAUNCH_JOB
        )
        await message_util.send(writer, args)
        response, request_type = await message_util.recv_response(reader)
        assert request_type == message_util.RequestType.LAUNCH_JOB
        if response == message_util.Response.SUCCESS:
            logger.info("Job launch request is accepted by the master.")
        else:
            logger.error("Job launch request is rejected by the master.")
            raise RuntimeError("Job launch request is rejected by the master.")


async def main(args: OobleckArguments):
    client = OobleckClient()
    dist_args = args.dist
    await client.connect_to_master(dist_args.master_ip, dist_args.master_port)
    await client.request_job_launch(args)


"""
Send a training job run request to the master.
Information to send
1. List of node IPs where agents and workers will be running
2. Job configuration (oobleck.elastic.training_util.OobleckArguments)
"""
if __name__ == "__main__":
    parser = sp.ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(DistributedArguments, dest="dist")
    parser.add_arguments(JobArguments, dest="job")
    parser.add_arguments(ModelArguments, dest="model")

    parsed_args = parser.parse_args()
    dist_args: DistributedArguments = getattr(parsed_args, "dist")
    job_args: JobArguments = getattr(parsed_args, "job")
    model_args: ModelArguments = getattr(parsed_args, "model")

    if dist_args.username is None:
        dist_args.username = getpass.getuser()

    args = OobleckArguments(dist_args, job_args, model_args)
    asyncio.run(main(args))
