# Ascend NPU Deployment

ServerlessLLM Store (`sllm-store`) is a Python library that supports fast model checkpoint loading from multi-tier storage (i.e., DRAM, SSD, HDD) into GPUs/NPUs.

ServerlessLLM Store provides a model manager and two key functions:

- `save_model`: Convert a HuggingFace model into a loading-optimized format and save it to a local path.
- `load_model`: Load a model into given GPUs.

This document provides instructions for deploying the `sllm_store` loading acceleration library from ServerlessLLM on an Ascend NPU environment. It outlines how to use the library with CANN to speed up model loading.

## Environment Setup

- **Check NPU status:** `npu-smi info`
- **Monitor NPU status in real-time:** `watch -d -n 1 npu-smi info`
- **Check CANN version:** `ascend-dmi -c`

---

### Set CANN Environment

The version used for the CANN test is **`8.0.0`**. Using other versions may cause bugs.

```shell
# Set up CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set up CANN environment
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/8.0.0
export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/lib64:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=$ASCEND_TOOLKIT_HOME/opp
export ASCEND_AICPU_PATH=$ASCEND_TOOLKIT_HOME/aicpu
export PYTHONPATH=$ASCEND_TOOLKIT_HOME/python/site-packages:$PYTHONPATH

# Optional: Set log level for CANN (for debugging)
export ASCEND_SLOG_PRINT_TO_STDOUT=1 # Enable logging to standard output
export ASCEND_GLOBAL_LOG_LEVEL=1 # Log level, typically 1 for INFO, 3 for ERROR
```

### Download `torch` and `torch_npu`

First, install `torch` and `torch_npu` to ensure the compilation process can find the necessary `torch_npu` functions.

```shell
pip install torch==2.4.0
pip install torch_npu==2.4.0.post2
```

### Set Up Conda Environment

If you don't have Conda, download it first.

```shell
conda create -n sllm-worker python=3.10 -y
conda activate sllm-worker
conda install -c conda-forge gcc=13 gxx cmake -y
conda install -c conda-forge ninja

# Set USE_CANN environment variable to enable using CANN
export USE_CANN=1
```

---

### Download from Source

1.  Clone the repository and navigate to the `store` directory.

<!-- end list -->

```shell
git clone https://gitee.com/openeuler/ServerlessLLM.git
```

2.  Download the `store` library from the source.

<!-- end list -->

```shell
rm -rf build
pip install .
```

---

## Usage Examples

1.  Convert a model to the ServerlessLLM format and save it locally.

<!-- end list -->

```python
from sllm_store.transformers import save_model

# Load a model from HuggingFace model hub.
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b', torch_dtype=torch.float16)

# Replace './models' with your local path.
save_model(model, './models/facebook/opt-1.3b')
```

2.  In a separate terminal process, start the checkpoint store server.

<!-- end list -->

```shell
# 'mem_pool_size' is the maximum size of the memory pool in GB. It should be larger than the model size.
export ASCEND_RT_VISIBLE_DEVICES="0" # Set the NPU device to use, e.g., "0,1,..."
sllm-store start --storage-path ./models --mem-pool-size 4GB
```

3.  Load the model and perform inference.

<!-- end list -->

```python
import time
import torch
import torch_npu
from sllm_store.transformers import load_model

# Warm up the GPU
num_npus = torch_npu.npu.device_count()
for i in range(num_npus):
    torch.ones(1).to(f"npu:{i}")
    torch_npu.npu.synchronize()

start = time.time()
model = load_model("facebook/opt-1.3b", device_map="auto", torch_dtype=torch.float16, storage_path="./models/", fully_parallel=True)
# Please note the loading time depends on the model size and the hardware bandwidth.
print(f"Model loading time: {time.time() - start:.2f}s")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
inputs = tokenizer('Hello, my dog is cute', return_tensors='pt').to("npu")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Built-in Test Cases

Navigate to `cd sllm_store/tests/python/`. There are four files available for testing `sllm_store` with CANN and NPU:

```
|
|-- test_cann_basic.py
|-- test_cann_load_model.py
|-- test_cann_load_vllm_model.py
|-- test_cann_save_model.py
```

---

## Applying vLLM Patches on NPU

To use vLLM, your versions must align with the following:

```shell
vllm-ascend==0.7.3.post1
torch-npu==2.5.1
torch==2.5.1
```

Tested vLLM version: [https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/installation.html](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/installation.html)

A known issue when installing vLLM on NPU is the incompatibility of `torch-npu` and `torch` versions with ServerlessLLM on the NPU. This requires manually fixing the version issues.

### Install vLLM Patches

1.  Download vLLM ascend from the source code; using `pip install` may cause issues.
2.  Check patch status (optional):

<!-- end list -->

```shell
./sllm_store/vllm_patch/check_patch.sh
```

3.  Apply patches:

<!-- end list -->

```shell
./sllm_store/vllm_patch/patch.sh
patch -p1 < sllm_load_npu.patch
patch -p1 < vllm_ascend.patch
```

Remove patches (if needed):

```shell
./sllm_store/vllm_patch/remove_patch.sh
```

---

### Note

> The patch files are located in the `sllm_store/vllm_patch/sllm_load.patch` directory of the ServerlessLLM repository.

Download a model from HuggingFace and save it in the ServerlessLLM format:

```shell
python3 sllm_store/example/cann_save_vllm_model.py --model-name facebook/opt-1.3b --storage-path $PWD/models --tensor-parallel-size 1
```

You can also transfer a model from a local path instead of downloading it from the network by passing the `--local-model-path` parameter.

After downloading the model, start the checkpoint store server and load the model within vLLM using the `sllm` load format.

Start the checkpoint store server in a separate process:

```shell
# 'mem_pool_size' is the maximum size of the memory pool in GB. It should be larger than the model size.
sllm-store start --storage-path $PWD/models --mem-pool-size 4GB
```

Load the model in vLLM:

```python
from vllm import LLM, SamplingParams

import os

storage_path = os.getenv("STORAGE_PATH", "./models")
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
```

### NNAL Issues

**If you encounter NNAL-related issues when running the vLLM example, install the NNAL package (if missing).**

NNAL provides the ATB library, including `libatb.so`. Download and install it. The version must match your CANN version. Use `$(uname -i)` to get the architecture, such as `aarch64` or `x86_64`.

```shell
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-nnal_8.1.RC1_linux-"$(uname -i)".run

chmod +x ./Ascend-cann-nnal_8.1.RC1_linux-"$(uname -i)".run

./Ascend-cann-nnal_8.1.RC1_linux-"$(uname -i)".run --install
```

Load the NNAL/ATB environment script to add the library's path (e.g., `libatb.so`). This must be done in every new terminal session before running any scripts.

```shell
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## Troubleshooting

Here are some issues you might encounter during compilation and usage.

### Torch Version Issues

The default `requirement.txt` specifies:

```
torch==2.5.1
torch-npu==2.5.1
```

If you encounter a compilation error related to `torch` or `torch-npu`, switch to:

```shell
pip install torch==2.4.0
pip install torch-npu==2.4.0.post2.
```

If you then encounter issues with `torchvision` or CANN when running `sllm_store`, switch back to:

```shell
pip install torch==2.5.1
pip install torch-npu==2.5.1
```

---

### Inference Issues -\> `load_model()`

If you encounter `torch_npu`-related issues:

1.  Downgrade `numpy` to `<= 2.0` -\> `pip install numpy==1.26.4`
2.  Install the necessary packages via `pip`.

Start the backend server:

```shell
sllm-store start --storage-path /root/PROJECT/ServerlessLLM-NPU/sllm_store/tests/python/models --mem-pool-size 4GB
```

When running `test_cann_load_model`, if you encounter an issue where the gRPC server cannot connect, set the following environment variables on the server. Proxies can block the connection.

```shell
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1" # It is often best to set both for compatibility
```

If you encounter an issue like this:

```shell
(sllm-worker) [root@devserver-bms-2956fa59 sllm_store]# sllm-store start
Traceback (most recent call last):
  File "/root/miniconda3/envs/sllm-worker/bin/sllm-store", line 5, in <module>
    from sllm_store.cli import main
  File "/root/PROJECT/ServerlessLLM-NPU/sllm_store/sllm_store/cli.py", line 24, in <module>
    from sllm_store.server import serve
  File "/root/PROJECT/ServerlessLLM-NPU/sllm_store/sllm_store/server.py", line 13, in <module>
    ctypes.CDLL(os.path.join(sllm_store.__path__[0], "libglog.so"))
  File "/root/miniconda3/envs/sllm-worker/lib/python3.10/ctypes/__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: /root/PROJECT/ServerlessLLM-NPU/sllm_store/sllm_store/libglog.so: cannot open shared object file: No such file or directory
[ERROR] 2025-06-09-23:13:26 (PID:2197687, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
```

Solution:

```shell
ln -s /root/PROJECT/ServerlessLLM-NPU/sllm_store/build/lib.linux-aarch64-cpython-310/sllm_store/libglog.so \
      /root/PROJECT/ServerlessLLM-NPU/sllm_store/sllm_store/libglog.so
```