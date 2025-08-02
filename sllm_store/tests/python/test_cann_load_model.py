import time
import torch
import torch_npu
# from torch_npu.contrib import transfer_to_npu
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sllm_store.transformers import load_model
from sllm_store.cann_utils import is_cann_available
from sllm_store._C import start_profiling

# start_profiling()

# os.environ['ASCEND_RT_VISIBLE_DEVICES']='0,1'

# --- Configuration ---
model_name = 'facebook/opt-1.3b'
storage_path = "/mnt/model_weights/models/"
target_dtype = torch.float16
# --- End Configuration ---

# --- Helper Functions for Weight Inspection ---
def inspect_weights(model_instance, stage_name, target_device=None):
    """Prints statistics for key model weights."""
    print(f"\n--- {stage_name} Weight Inspection ---")
    
    if target_device:
        print(f"Model current device: {next(model_instance.parameters()).device}")

    # For Qwen2.5, typically model.model.layers is the list of transformer blocks
    # We'll check the first layer's self_attn.q_proj.weight as a representative
    try:    
        # Access the first transformer block's query projection weight
        # Path might be model.model.layers[0].self_attn.q_proj.weight for Qwen2.5
        # Or model.decoder.layers[0].self_attn.q_proj.weight for other architectures
        
        # A safer way to find a relevant weight
        target_param = None
        for name, param in model_instance.named_parameters():
            if "layers.0.self_attn.q_proj.weight" in name:
                target_param = param
                break
        
        if target_param is None:
            print("Could not find a common transformer layer weight (e.g., q_proj).")
            print("Attempting to inspect the first found trainable parameter.")
            for name, param in model_instance.named_parameters():
                if param.requires_grad:
                    target_param = param
                    break

        if target_param is not None:
            # Ensure the tensor is on CPU for statistical stability when comparing across devices
            # and convert to float32 for accurate statistics if it's float16
            param_cpu_float32 = target_param.cpu().float()
            
            print(f"Inspecting weight: '{name}' (dtype: {target_param.dtype}, device: {target_param.device})")
            print(f"  Shape: {param_cpu_float32.shape}")
            print(f"  Mean: {param_cpu_float32.mean().item():.6f}")
            print(f"  Std: {param_cpu_float32.std().item():.6f}")
            print(f"  Min: {param_cpu_float32.min().item():.6f}")
            print(f"  Max: {param_cpu_float32.max().item():.6f}")
            
            # Check for NaNs/Infs
            if torch.isnan(param_cpu_float32).any():
                print("!!! WARNING: NaN values detected in this weight !!!")
            if torch.isinf(param_cpu_float32).any():
                print("!!! WARNING: Inf values detected in this weight !!!")

            # Print first few elements to spot obvious corruption patterns
            print("  First 10 elements (flattened):")
            print(param_cpu_float32.flatten()[:10].tolist())
        else:
            print("No trainable parameters found in the model to inspect.")

    except Exception as e:
        print(f"Error during weight inspection for '{stage_name}': {e}")

def compare_weights(model1, model2, name1="Model 1", name2="Model 2"):
    print(f"\n--- Comparing selected weights between {name1} and {name2} ---")
    
    param1 = None
    param2 = None

    for n1, p1 in model1.named_parameters():
        if "layers.0.self_attn.q_proj.weight" in n1:
            param1_name = n1
            param1 = p1
            break
    
    for n2, p2 in model2.named_parameters():
        if "layers.0.self_attn.q_proj.weight" in n2:
            param2_name = n2
            param2 = p2
            break

    if param1 is None or param2 is None:
        print("Could not find corresponding 'q_proj.weight' in both models for comparison.")
        print("Falling back to comparing first trainable parameters if available.")
        
        # Fallback to first trainable parameter
        for n1, p1 in model1.named_parameters():
            if p1.requires_grad:
                param1_name = n1
                param1 = p1
                break
        for n2, p2 in model2.named_parameters():
            if p2.requires_grad:
                param2_name = n2
                param2 = p2
                break

    if param1 is not None and param2 is not None:
        # Move both to CPU and convert to float32 for accurate comparison
        p1_cpu_float32 = param1.cpu().float()
        p2_cpu_float32 = param2.cpu().float()

        if p1_cpu_float32.shape != p2_cpu_float32.shape:
            print(f"WARNING: Mismatched shapes: {param1_name}: {p1_cpu_float32.shape} vs {param2_name}: {p2_cpu_float32.shape}")
            return
        
        diff = (p1_cpu_float32 - p2_cpu_float32).abs().mean().item()
        print(f"Mean absolute difference for '{param1_name}' vs '{param2_name}': {diff:.6f}")
        
        if diff > 1e-4: # A small tolerance for floating point differences due to precision or operations
            print(f"*** WARNING: Significant difference detected (diff > 1e-4) between {name1} and {name2} for this weight! ***")
        else:
            print(f"Weights are largely similar (diff <= 1e-4).")
    else:
        print("Could not find suitable parameters in both models for comparison.")


# --- Main Script ---

# Check if NPU is available
device_type = "npu" if is_cann_available() else "cuda"
if not is_cann_available() and torch.cuda.is_available():
    device_type = "cuda"
elif not is_cann_available() and not torch.cuda.is_available():
    print("Neither NPU nor CUDA is available. This script requires an accelerator.")
    exit()

print(f"Using {device_type.upper()} devices")

# Warm up the device
if device_type == "npu":
    try:
        num_npus = torch_npu.npu.device_count()
        print(f"Number of NPU devices: {num_npus}")
        for i in range(num_npus):
            torch.ones(1).to(f"npu:{i}")
            torch_npu.npu.synchronize()
        print("NPU devices warmed up")
    except Exception as e:
        print(f"Error warming up NPU devices: {e}")
        # Proceed with a single device if multi-NPU warm-up fails
        device = torch.device("npu:0")
elif device_type == "cuda":
    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_gpus}")
    for i in range(num_gpus):
        torch.ones(1).to(f"cuda:{i}")
        torch.cuda.synchronize()
    print("CUDA devices warmed up")

device = torch.device(f"{device_type}")
print(f"Using primary device: {device}")

# --- 1. Baseline: Load with standard transformers to CPU ---
print("\n--- STEP 1: Loading model with standard transformers (CPU baseline) ---")
start_baseline = time.time()
try:
    # Load to CPU first to rule out device-specific issues initially
    original_model_cpu = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=target_dtype,
        trust_remote_code=True # Qwen models often require this
    )
    print(f"Original model loaded successfully to CPU. Time: {time.time() - start_baseline:.2f}s")
    inspect_weights(original_model_cpu, "Original CPU Load")

except Exception as e:
    print(f"FAILED: Standard transformers model loading failed: {e}")
    print("This indicates an issue with the base model files or Hugging Face setup.")
    exit()

# --- 2. Load model with sllm_store ---
print(f"\n--- STEP 2: Loading model with sllm_store.transformers.load_model ---")
sllm_model_initial = None
sllm_model_on_npu = None
try:
    start_sllm_load = time.time()
    sllm_model_initial = load_model(
        model_name,
        device_map="auto", # sllm_store will decide where to initially place
        torch_dtype=target_dtype,
        storage_path=storage_path,
        fully_parallel=True
    )
    # Print out the device map
    if hasattr(sllm_model_initial, 'hf_device_map'):
        print("Loaded model's hf_device_map:")
        print(sllm_model_initial.hf_device_map)
    else:
        print("hf_device_map not attached to model.")
    
    print(f"sllm_store model loaded successfully. Time: {time.time() - start_sllm_load:.2f}s")
    
    # Inspect weights immediately after sllm_store load (may still be on CPU or mixed)
    inspect_weights(sllm_model_initial, "After sllm_store Load (Initial Placement)")
    
    # Compare with the original CPU model
    compare_weights(original_model_cpu, sllm_model_initial, "Original CPU", "sllm_store Initial")

    # --- 3. Move sllm_store model to target device (NPU/CUDA) ---
    print(f"\n--- STEP 3: Skipping move as model is already dispatched via Accelerate ---")
    sllm_model_on_npu = sllm_model_initial
    start_to_device = time.time()
    # sllm_model_on_npu = sllm_model_initial.to(device)
    print(f"sllm_store model moved to {device.type.upper()}. Time: {time.time() - start_to_device:.2f}s")
    
    # Inspect weights after moving to the target device
    inspect_weights(sllm_model_on_npu, f"After .to({device.type.upper()})")

    # Compare the sllm_store model after .to() with the original CPU model
    compare_weights(original_model_cpu, sllm_model_on_npu, "Original CPU", "sllm_store on NPU")
    
    # Compare the sllm_store model before and after .to() (if it started on CPU)
    if str(next(sllm_model_initial.parameters()).device).startswith('cpu'):
        compare_weights(sllm_model_initial, sllm_model_on_npu, "sllm_store Initial (CPU)", "sllm_store on NPU")

except Exception as e:
    print(f"FAILED: sllm_store model loading or device transfer failed: {e}")
    # If this fails, the issue is likely within sllm_store or torch_npu's handling
    exit()

# --- 4. Load the tokenizer ---
print("\n--- STEP 4: Loading tokenizer ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"FAILED: Tokenizer loading failed: {e}")
    exit()

# --- 5. Prepare input and run inference ---
print("\n--- STEP 5: Preparing input and running inference ---")
inputs = tokenizer('Hello, my dog is cute', return_tensors='pt').to("npu")
print(f"Input tokens on device: {inputs['input_ids'].device}")
print(f"Input tokens decoded: {tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)}")

# for name, param in sllm_model_on_npu.named_parameters():
#   print(name)
#   print(param)

try:
    start_inference = time.time()
    outputs = sllm_model_on_npu.generate(**inputs, max_new_tokens=100)
    print(f"Inference completed. Time: {time.time() - start_inference:.2f}s")

    # Check output tensor for issues
    print(f"Output tensor shape: {outputs.shape}")
    print(f"Output tensor dtype: {outputs.dtype}")
    if torch.isnan(outputs).any():
        print("!!! WARNING: NaN values detected in output tensor !!!")
    if torch.isinf(outputs).any():
        print("!!! WARNING: Inf values detected in output tensor !!!")
    
    # Decode and clean up output
    print("\n--- Inference Response ---")
    print("Response (raw decoded):", tokenizer.decode(outputs[0], skip_special_tokens=True))

except Exception as e:
    print(f"FAILED: Inference failed: {e}")
    print("This could be due to corrupted weights, NPU issues during computation, or other runtime errors.")
    # If the output is "corrupted", this is where it would manifest.
    # The weight checks above help determine if it was corrupted *before* inference.

print("\n--- Testing Complete ---")