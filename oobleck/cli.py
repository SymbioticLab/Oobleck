"""Executable Tyro CLI for Oobleck's refactor-era public operations."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Union

from oobleck.config import (
    AgentConfig,
    ChaosConfig,
    DrainConfig,
    InspectMembershipConfig,
    MasterServiceConfig,
    ProfileCommandConfig,
    TrainingLaunchConfig,
)
from oobleck.elastic import (
    AsyncioTcpControlTransport,
    MasterControlService,
    MessageEnvelope,
    run_agent_service,
    ssh_agent_command,
    load_initial_hostfile,
    inspect_membership,
    request_drain,
)
from oobleck.planning import (
    CompatibilityFingerprint,
    ModelProfile,
    ModelProfiler,
    ProfilingWorkload,
    create_pipeline_templates,
    save_templates,
)

Command = Union[
    MasterServiceConfig,
    AgentConfig,
    TrainingLaunchConfig,
    ProfileCommandConfig,
    DrainConfig,
    InspectMembershipConfig,
    ChaosConfig,
]


async def _master(config: MasterServiceConfig) -> None:
    """Run the membership master until cancellation, then close every stream."""

    service = MasterControlService(
        AsyncioTcpControlTransport(max_frame_bytes=config.max_frame_bytes),
        lease_timeout_s=config.lease_timeout_s,
        lease_check_interval_s=max(0.05, config.heartbeat_interval_s / 2),
        max_nodes=config.max_nodes,
    )
    server = await service.start(config.host, config.port)
    print(", ".join(str(item.getsockname()) for item in server.sockets or ()))
    try:
        await server.serve_forever()
    finally:
        await service.close()


async def _agent(config: AgentConfig) -> None:
    """Run a node agent and emit machine-readable membership changes."""

    async def announce(message: MessageEnvelope) -> None:
        """Print a compact generation notice for operators and launch scripts."""

        print(
            json.dumps(
                {
                    "generation": message.generation,
                    "nodes": [item["agent_id"] for item in message.payload["nodes"]],
                    "reasons": message.payload["reasons"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    await run_agent_service(config, on_membership=announce)


def _launch(config: TrainingLaunchConfig) -> int:
    """Run locally or bootstrap initial remote agents and supervise their exits."""

    if not config.training_script.is_file():
        raise FileNotFoundError(f"training script does not exist: {config.training_script}")
    if config.initial_hostfile is None:
        result = subprocess.run(
            [sys.executable, str(config.training_script), *config.script_args],
            check=False,
        )
        return result.returncode

    if config.agent_script is None or not config.agent_script.is_file():
        raise FileNotFoundError(f"agent script does not exist: {config.agent_script}")
    hosts = load_initial_hostfile(config.initial_hostfile)
    if len(hosts) > config.max_nodes:
        raise ValueError("initial hostfile exceeds configured max_nodes")
    if any(len(host.gpu_ids) != config.tensor_parallel_size for host in hosts):
        raise ValueError("initial hostfile GPU width does not match tensor_parallel_size")
    processes = [
        subprocess.Popen(
            ssh_agent_command(
                host,
                agent_script=config.agent_script,
                worker_script=config.training_script,
                master_host=config.master_host,
                master_port=config.master_port,
                worker_args=config.script_args,
                remote_python=config.remote_python,
                socket_directory=config.socket_directory,
                ssh_command=config.ssh_command,
            )
        )
        for host in hosts
    ]
    try:
        while True:
            codes = [process.poll() for process in processes]
            failed = next((code for code in codes if code not in (None, 0)), None)
            if failed is not None:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    process.wait()
                return failed
            if all(code == 0 for code in codes):
                return 0
            time.sleep(0.1)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise


def _profile(config: ProfileCommandConfig) -> None:
    """Measure or load a profile, validate compatibility, and save templates."""

    if config.profile is None and config.measurement_factory is None:
        raise ValueError(
            "provide --profile with a versioned ModelProfile JSON or "
            "--measurement-factory module:function"
        )
    if config.profile is not None:
        profile = ModelProfile.load(config.profile)
    else:
        module_name, separator, attribute = str(config.measurement_factory).partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("measurement_factory must have the form module:function")
        factory = getattr(importlib.import_module(module_name), attribute)
        workload = factory(config)
        if not isinstance(workload, ProfilingWorkload):
            raise TypeError("measurement factory must return ProfilingWorkload")
        if config.hardware is not None:
            hardware = config.hardware
        else:
            import torch

            hardware = (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else "cpu"
            )
        if config.cornstarch_version is not None:
            cornstarch_version = config.cornstarch_version
        else:
            try:
                cornstarch_version = importlib.metadata.version("cornstarch")
            except importlib.metadata.PackageNotFoundError:
                cornstarch_version = "source-tree"
        fingerprint = CompatibilityFingerprint(
            config.model,
            config.dtype,
            config.tensor_parallel_size,
            hardware,
            cornstarch_version,
        )
        profile_path = config.profile_output or config.output.with_suffix(".profile.json")
        profile = ModelProfiler(config.model, fingerprint=fingerprint).measure_and_record(
            profile_path,
            config.microbatch_size,
            workload,
            warmup_steps=config.warmup_steps,
            measurement_steps=config.measurement_steps,
        )
    fingerprint = profile.fingerprint
    requested = (
        config.model,
        config.dtype,
        config.tensor_parallel_size,
        config.microbatch_size,
    )
    actual = (
        fingerprint.model,
        fingerprint.dtype,
        fingerprint.tensor_parallel_size,
        profile.microbatch_size,
    )
    if requested != actual:
        raise ValueError(f"profile compatibility mismatch: requested={requested}, actual={actual}")
    device_memory_bytes = config.device_memory_bytes
    if device_memory_bytes is None:
        import torch

        if not torch.cuda.is_available():
            raise ValueError("device_memory_bytes is required when profiling without a CUDA device")
        device_memory_bytes = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory
    templates = create_pipeline_templates(
        config.model,
        profile.layers,
        config.resource_counts,
        config.tensor_parallel_size,
        fingerprint=fingerprint,
        device_memory_bytes=device_memory_bytes,
    )
    save_templates(config.output, tuple(templates.values()), fingerprint)


async def _inspect(config: InspectMembershipConfig) -> None:
    """Print the master's current checksummed membership as JSON."""

    snapshot = await inspect_membership(config.master_host, config.master_port)
    print(json.dumps(asdict(snapshot), sort_keys=True))


async def _drain(config: DrainConfig) -> None:
    """Request graceful removal of a live stable node identity."""

    generation = await request_drain(config.master_host, config.master_port, config.node_id)
    print(f"drain accepted at generation {generation}")


def _chaos(config: ChaosConfig) -> None:
    """Kill only a PID proven to be the confirmed disposable example agent."""

    path = f"/proc/{config.pid}/cmdline"
    try:
        command = open(path, "rb").read().decode(errors="replace")
    except OSError as exc:
        raise ValueError(f"cannot inspect target PID {config.pid}: {exc}") from exc
    if "run_agent.py" not in command or config.target_node_id not in command:
        raise ValueError("target is not the validated disposable Oobleck example agent")
    os.kill(config.pid, signal.SIGKILL)


def dispatch(command: Command) -> object:
    """Route a validated Tyro dataclass to its synchronous or async handler."""

    if isinstance(command, MasterServiceConfig):
        return asyncio.run(_master(command))
    if isinstance(command, AgentConfig):
        return asyncio.run(_agent(command))
    if isinstance(command, TrainingLaunchConfig):
        return _launch(command)
    if isinstance(command, ProfileCommandConfig):
        return _profile(command)
    if isinstance(command, DrainConfig):
        return asyncio.run(_drain(command))
    if isinstance(command, InspectMembershipConfig):
        return asyncio.run(_inspect(command))
    if isinstance(command, ChaosConfig):
        return _chaos(command)
    raise TypeError(f"unsupported command {type(command).__name__}")


def main() -> object:
    """Parse the command union with Tyro and dispatch the selected operation."""

    import tyro

    return dispatch(tyro.cli(Command))


if __name__ == "__main__":
    main()
