#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
sys.path.append(os.getcwd())
import yaml
import utils
import subprocess
from kubejobs.jobs import KubernetesJob

# Monkey patch the KubernetesJob.run method to use kubectl create instead of kubectl apply
original_run = KubernetesJob.run

def patched_run(self):
    job_yaml = self.generate_yaml()

    # Save the generated YAML to a temporary file
    with open("temp_job.yaml", "w") as temp_file:
        temp_file.write(job_yaml)

    # Run kubectl create instead of kubectl apply
    cmd = ["kubectl", "create", "-f", "temp_job.yaml"]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Remove the temporary file
        os.remove("temp_job.yaml")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command '{' '.join(cmd)}' failed with return code {e.returncode}.")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        # Remove the temporary file
        os.remove("temp_job.yaml")
        return e.returncode  # return the exit code
    except Exception as e:
        print(f"An unexpected error occurred while running '{' '.join(cmd)}'.")
        # Remove the temporary file
        os.remove("temp_job.yaml")
        return 1  # return the exit code

# Apply the patch
KubernetesJob.run = patched_run

def argument_parser():
    parser = argparse.ArgumentParser(description="Backend Runner")
    parser.add_argument("config", type=str)
    parser.add_argument("--job-name", "-n", type=str, default="neel-crosscoder")
    parser.add_argument("--gpu-type", type=str, default="NVIDIA-H100-80GB-HBM3")
    parser.add_argument("--gpu-limit", type=int, default=None)
    parser.add_argument("--cpu-limit", type=int, default=32)
    parser.add_argument("--namespace", type=str, default="eidf097ns")
    args = parser.parse_args()
    return args

def main():
    args = argument_parser()
    configs = yaml.safe_load(open(args.config, "r"))
    job_name = args.job_name
    is_completed = utils.check_if_completed(job_name, namespace=args.namespace)
    cpu_limit = args.cpu_limit
    if is_completed is True:
        # "sleep $((RANDOM % 300 + 300)) && " \
        base_args = "apt -y update && apt -y upgrade && " \
        "apt-get -y install git-lfs unzip psmisc wget git sudo python3 python-is-python3 pip bc htop nano nodejs npm curl && " \
        "mkdir user && " \
        "cd user && " \
        "git lfs install && " \
        "pip install -U pip && " \
        "pip install datasets && " \
        "mkdir home && " \
        "cd home && " \
        "git clone https://github.com/Neelectric/open-r1_olmo.git && " \
        "pip install gpustat && " \
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python " \
        "HF_HUB_DISABLE_PROGRESS_BARS=1 CURL_CA_BUNDLE=\"\" "
        command = "&& sleep infinity "
        secret_env_vars = configs["env_vars"]
        # Create a Kubernetes Job with a name, container image, and command
        print(f"Creating job for: {command}")
        job = KubernetesJob(name=job_name, cpu_request=cpu_limit, ram_request="260Gi",
                            image="nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04",
                            gpu_type="nvidia.com/gpu",
                            gpu_limit=configs["gpu_limit"] if args.gpu_limit is None else args.gpu_limit,
                            gpu_product=configs["gpu_product"] if args.gpu_type is None else args.gpu_type,
                            backoff_limit=1,
                            command=["/bin/bash", "-c", "--"],
                            args=[base_args + command],
                            secret_env_vars=secret_env_vars,
                            user_email="p.minervini@ed.ac.uk",
                            namespace=args.namespace,
                            kueue_queue_name=f"{args.namespace}-user-queue")
        # Run the Job on the Kubernetes cluster
        job.run()

if __name__ == "__main__":
    main()

