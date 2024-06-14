"""Example of running inference with Splitwise.

Run `python examples/offline_splitwise_inference.py --sep-prompt-token --tensor-parallel-size 2`
"""

from vllm import LLM, SamplingParams, AsyncEngineArgs, AsyncLLMEngine
import argparse
import torch

assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs to run this example."

def parse_args():
    parser = argparse.ArgumentParser(
        description="Splitwise example")
    
    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()

# Sample prompts.
prompts = [
    # "Hello, my name is",
    "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]

prompts = [
    "Life blooms like a flower. Far away or by the road. Waiting",
    "A quick brown fox",
    "Artificial intelligence is",
    "To be or not to be,",
    "one two three four"
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument("--sep-prompt-token", action="store_true", default=False)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    # Create an LLM.
    llm = LLM(**vars(args))

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")