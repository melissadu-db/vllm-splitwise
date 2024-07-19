from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_p=0.95)

# Create an LLM.
# llm = LLM(model="facebook/opt-125m", disagg_mode='distserve', prefill_tp=4, decode_tp=2, enforce_eager=True)
# llm = LLM(model="facebook/opt-125m", disagg_mode='splitwise', tensor_parallel_size=2, enforce_eager=True)
llm = LLM(model="facebook/opt-125m", tensor_parallel_size=4, disagg_mode='distserve', enforce_eager=True)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
