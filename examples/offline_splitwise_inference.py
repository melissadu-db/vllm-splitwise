"""Example of running inference with Splitwise.

Run `python examples/offline_splitwise_inference.py --sep-prompt-token --tensor-parallel-size 2`
"""

from vllm import LLM, SamplingParams, AsyncEngineArgs, AsyncLLMEngine
import argparse
import torch
from vllm.utils import random_uuid
import asyncio

assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs to run this example."

# Sample prompts.
prompts = [
    "Life blooms like a flower. Far away or by the road. Waiting", "A quick brown fox", "Artificial intelligence is",
    "To be or not to be,", "one two three four"
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0)


async def generate_and_process(engine, prompt, sampling_params, request_id):
    async for output in engine.generate(prompt, sampling_params, request_id):
        final_output = output
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    print(f"Prompt: {prompt!r}, Generated text: {text_outputs!r}")
    return text_outputs


async def main():
    parser = argparse.ArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    # Create an LLM.
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    request_id = random_uuid()

    # Generate texts from the prompts asynchronously.
    tasks = []
    for prompt in prompts:
        task = asyncio.create_task(generate_and_process(engine, prompt, sampling_params, request_id))
        tasks.append(task)

    # Wait for all tasks to complete.
    await asyncio.gather(*tasks)


asyncio.run(main())
