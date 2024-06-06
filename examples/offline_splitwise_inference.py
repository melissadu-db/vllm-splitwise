from vllm import LLM, SamplingParams
import torch

assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs to run the test."

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

# Create an LLM.
# llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", sep_prompt_token=True, tensor_parallel_size=1)
llm = LLM(model="facebook/opt-125m", sep_prompt_token=False, tensor_parallel_size=1)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")