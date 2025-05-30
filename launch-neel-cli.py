#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

sys.path.append(os.getcwd())

import yaml
import utils

from kubejobs.jobs import KubernetesJob


def argument_parser():
    parser = argparse.ArgumentParser(description="Backend Runner")
    parser.add_argument("config", type=str)
    parser.add_argument("--job-name", "-n", type=str, default="neel-crosscoder")
    parser.add_argument("--gpu-type", type=str, default=None)
    parser.add_argument("--gpu-limit", type=int, default=None)
    parser.add_argument("--namespace", type=str, default="eidf097ns")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    configs = yaml.safe_load(open(args.config, "r"))

    job_name = args.job_name
    is_completed = utils.check_if_completed(job_name, namespace=args.namespace)

    if is_completed is True:
        # "sleep $((RANDOM % 300 + 300)) && " \
        base_args = "apt -y update && apt -y upgrade && " \
        "apt-get -y install git-lfs unzip psmisc wget git sudo python3 python-is-python3 pip bc htop nano nodejs npm curl && " \
        "mkdir -p /home/user && " \
        "cd /home/user && " \
        "git clone https://github.com/Neelectric/open-r1_olmo.git && " \
        "wget -qO- https://astral.sh/uv/install.sh | sh && " \
        "source $HOME/.local/bin/env && " \
        "cd open-r1_olmo && " \
        "bash setup.bash && " \
        "git lfs install && " \
        "pip install -U pip && " \
        "pip install datasets && " \
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python " \
        "HF_HUB_DISABLE_PROGRESS_BARS=1 CURL_CA_BUNDLE=\"\" "
        command = "&& sleep infinity "

        secret_env_vars = configs["env_vars"]

        # Create a Kubernetes Job with a name, container image, and command
        print(f"Creating job for: {command}")
        job = KubernetesJob(name=job_name, cpu_request="32", ram_request="260Gi",
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
