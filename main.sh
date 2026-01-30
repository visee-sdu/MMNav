#!/usr/bin/env bash

export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export HABITAT_SIM_GPU_RENDERING=1
export HABITAT_SIM_EGL=1

export PYTHONPATH=${PYTHONPATH}:/nav
export CHALLENGE_CONFIG_FILE=configs/multi_agents_multi_goals.yaml
export AGENT_EVALUATION_TYPE=local

CUDA_VISIBLE_DEVICES=0 python nav/main.py -v 0 --dump_location ./data --exp_name test --start_ep 0 --end_ep 400 --evaluation $AGENT_EVALUATION_TYPE 