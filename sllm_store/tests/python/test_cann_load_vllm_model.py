from vllm import LLM, SamplingParams

import os

storage_path = os.getenv("STORAGE_PATH", "/mnt/model_weights/vllm_models")
model_name = "facebook/opt-1.3b"
model_path = os.path.join(storage_path, model_name)

llm = LLM(
    model=model_path,
    load_format="serverless_llm",
    dtype="float16"
)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")