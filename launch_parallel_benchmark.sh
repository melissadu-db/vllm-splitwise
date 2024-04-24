#!/bin/bash

#  All the batch sizes we want to test are:
#    [
#       1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 
#       256, 512, 1024, 1536, 2048, 3072, 4096
#    ]

GREEN="\033[0;32m"
RESET="\033[0m"

source /home/vllm-private/.venv/bin/activate

echo $(which python)

batch_sizes="96,128,256,512,1024,1536"
tp_size=4
gpu_ids="0,1"

set -e
set -x

echo -e "${GREEN}Launching benchmark for tensor_parallel $tp_size...${RESET}"
for gpu_idx in 0 1; do
    python benchmarks/kernels/benchmark_gptq.py $gpu_idx $gpu_ids $batch_sizes $tp_size &
done


batch_sizes="48,64,96,128,256,512,1024,1536,2048,3072,4096"
tp_size=2
gpu_ids="2,3,4,5"

echo -e "${GREEN}Launching benchmark for tensor_parallel $tp_size...${RESET}"
for gpu_idx in 0 1 2 3; do
    python benchmarks/kernels/benchmark_gptq.py $gpu_idx $gpu_ids $batch_sizes $tp_size &
done


batch_sizes="1024,1536,2048,3072,4096"
tp_size=8
gpu_ids="6,7"

echo -e "${GREEN}Launching benchmark for tensor_parallel $tp_size...${RESET}"
for gpu_idx in 0 1; do
    python benchmarks/kernels/benchmark_gptq.py $gpu_idx $gpu_ids $batch_sizes $tp_size &
done

wait

