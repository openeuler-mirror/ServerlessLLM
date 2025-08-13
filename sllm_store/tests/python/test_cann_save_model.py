# Import necessary modules
from sllm_store.transformers import save_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

models = [
    # 'facebook/opt-1.3b',
    'facebook/opt-6.7b',
    # 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    # 'Qwen/Qwen2.5-Coder-14B-Instruct',
    # 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    # 'Qwen/Qwen2.5-72B-Instruct'
]

for model_name in models:
    print(model_name)
    # Load the model from Hugging Face (as in your code)
    loaded_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    # Save the model to the NVMe-mounted path in ServerlessLLM's optimized format
    save_model(loaded_model, f'/mnt/model_weights/models/{model_name}')

    # Load and save the tokenizer (using standard Transformers method, as ServerlessLLM focuses on model weights)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f'/mnt/model_weights/models/{model_name}')