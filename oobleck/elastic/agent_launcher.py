import subprocess
import torch
import sys
import os

if __name__ == "__main__":
    # Number of GPUs per agent.
    # Agents are assumed to be separated nodes.
    num_gpus_per_agent = 1

    num_gpus = torch.cuda.device_count()

    processes = []
    # launch Python module oobleck.elastic.agent with the following environment:
    # CUDA_VISIBLE_DEVICES=[i, i+1, ... i+num_gpus_per_agent-1]
    for i in range(0, num_gpus, num_gpus_per_agent):
        current_env = os.environ.copy()
        current_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i + j) for j in range(num_gpus_per_agent)]
        )

        # spawn the process
        cmd = [sys.executable, "-m", "oobleck.elastic.agent"]

        print(f"Launching: {cmd} with device {current_env['CUDA_VISIBLE_DEVICES']}")

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    # Wait for all processes to be done
    for process in processes:
        process.wait()
    print("All agents terminated")
