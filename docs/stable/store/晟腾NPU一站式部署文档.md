# 晟腾 NPU 一站式部署文档

ServerlessLLM Store (`sllm-store`) 是一个 Python 库，支持将模型检查点从多层存储（如 DRAM、SSD、HDD）快速加载到 GPU/NPU 中。

ServerlessLLM Store 提供了一个模型管理器和两个关键功能：

- **`save_model`**：将 HuggingFace 模型转换为加载优化的格式并保存到本地路径。
- **`load_model`**：将模型加载到指定的 GPU 中。

---

本文档旨在部署 ServerlessLLM 中的 sllm\_store 加载加速库到昇腾 NPU 环境和 CANN 结合进行加载提速。


## 环境设置

- 查看 npu 状态：`npu-smi info`
- 实时监测 npu 状态： `watch -d -n 1 npu-smi info`
- 查看 CANN 版本：`ascend-dmi -c`

---

### 设置 CANN 环境

CANN 测试中使用版本为 `8.0.0`，使用其他版本可能导致BUG。

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
export ASCEND_SLOG_PRINT_TO_STDOUT=1 #日志打屏, 可选
export ASCEND_GLOBAL_LOG_LEVEL=1 #日志级别常用 1 INFO级别; 3 ERROR级别
```

### 下载 torch 和 torch\_npu

先下载 `torch` 和 `torch_npu` 以保证编译时候可以找到 `torch_npu` 的函数

```shell
pip install torch==2.4.0
pip install torch_npu==2.4.0.post2
```

### 设置 conda 环境

如没有 conda 内置，先下载 conda

```shell
conda create -n sllm-worker python=3.10 -y  
conda activate sllm-worker
conda install -c conda-forge gcc=13 gxx cmake -y
conda install -c conda-forge ninja

# Set USE_CANN environment variable to enable using CANN
export USE_CANN=1
```

---

### 从源码下载

1. 克隆仓库，进入 `store` 路径

<!-- end list -->

```shell
git clone https://gitee.com/openeuler/ServerlessLLM.git
```

2. 从源码下载 `store` 库

<!-- end list -->

```shell
rm -rf build  
pip install .
```

---

## 使用样例

1. 转换模型到 ServerlessLLM format 并保存到本地

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

2. 在一个单独终端 process 里启动 checkpoint store 服务器

<!-- end list -->

```shell
# 'mem_pool_size' is the maximum size of the memory pool in GB. It should be larger than the model size.  
export ASCEND_RT_VISIBLE_DEVICES="0" # 设置需要使用的 NPU Device "0,1,..."
sllm-store start --storage-path ./models --mem-pool-size 4GB
```

3. 加载模型并推理

<!-- end list -->

```python
import time
import torch
import torch_npu
from sllm_store.transformers import load_model

# warm up the GPU
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

## 内置测试用例

进入 `cd sllm_store/tests/python/`，有四个文件可以用来测试 CANN 和 NPU 的 sllm\_store

```
|
|-- test_cann_basic.py
|-- test_cann_load_model.py
|-- test_cann_load_vllm_model.py
|-- test_cann_save_model.py
```

-----

## NPU 上应用 vLLM 补丁

为了使用 vLLM，版本必须与以下对齐：

```shell
vllm-ascend==0.7.3.post1
torch-npu==2.5.1
torch-npu==2.5.1
```

已测试的 vLLM 版本：

[https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/installation.html](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/installation.html)

在 NPU 上安装 vLLM 时，一个存在的问题是 torch-npu 和 torch 版本与 NPU 上的 ServerlessLLM 不兼容，因此需要手动修复版本问题。

### 安装 vLLM 补丁

1. 使用源码下载 vLLM ascend，不能使用 pip install，要不是会出问题
2. 检查补丁状态（可选）：

<!-- end list -->

```
./sllm_store/vllm_patch/check_patch.sh
```

3. 应用补丁：

<!-- end list -->

```
./sllm_store/vllm_patch/patch.sh
patch -p1 < sllm_load_npu.patch
patch -p1 < vllm_ascend.patch
```

移除补丁（如果需要）：

```
./sllm_store/vllm_patch/remove_patch.sh
```

---

### 注意

> 补丁文件位于 ServerlessLLM 仓库的 `sllm_store/vllm_patch/sllm_load.patch`。

从 HuggingFace 下载模型并以 ServerlessLLM 格式保存：

```shell
python3 sllm_store/example/cann_save_vllm_model.py --model-name facebook/opt-1.3b --storage-path $PWD/models --tensor-parallel-size 1
```

也可以通过传递 `--local-model-path` 参数，从本地路径传输模型，而不是从网络下载。

下载模型后，启动 checkpoint store 服务器，并通过 `sllm` 加载格式在 vLLM 中加载模型。

在单独的进程中启动 checkpoint store 服务器：

```shell
# 'mem_pool_size' 是内存池的最大大小，单位为 GB。它应大于模型大小。
sllm-store start --storage-path $PWD/models --mem-pool-size 4GB
```

在 vLLM 中加载模型：

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

# 打印输出。
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### NNAL 问题

**当运行 vLLM 示例时，如果遇到与 NNAL 相关的问题：安装 NNAL 包（如果缺失）**

NNAL 提供 ATB 库，包括 libatb.so。下载并安装它（版本要与您的 CANN 匹配；使用 $(uname -i) 来获取架构，例如 aarch64 或 x86\_64）：

```shell
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-nnal_8.1.RC1_linux-"$(uname -i)".run

chmod +x ./Ascend-cann-nnal_8.1.RC1_linux-"$(uname -i)".run

./Ascend-cann-nnal_8.1.RC1_linux-"$(uname -i)".run --install
```

加载 NNAL/ATB 环境脚本以添加库（例如 libatb.so）的路径（在每个新的终端会话中，必须在运行脚本之前完成此操作）：

```shell
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## 故障排除

以下为编译和使用过程可能遇到的问题。

### Torch 版本问题

默认的 requirement.txt 设置为：

```
torch==2.5.1
torch-npu==2.5.1
```

如果遇到由于 torch 或 torch-npu 导致的编译错误，请切换到：

```shell
pip install torch==2.4.0
pip install torch-npu==2.4.0.post2. 
```

之后，如果在运行 sllm\_store, 时遇到与 torchvision 或 CANN 相关的问题，请切换回： 

```shell
pip install torch==2.5.1
pip install torch-npu==2.5.1
```

---

### 推理问题 -\> load\_model()

如果遇到与 torch\_npu 相关的问题：

1.  将 numpy 降级到 \<= 2.0 -\> `pip install numpy==1.26.4`
2.  pip 安装必要的包

启动后端服务器：

```shell
sllm-store start --storage-path /root/PROJECT/ServerlessLLM-NPU/sllm_store/tests/python/models --mem-pool-size 4GB
```

当运行 `test_cann_load_model` 时，如果遇到 gPRC 服务器无法连接的问题。在服务器上设置以下环境变量，因为代理会阻止连接。

```shell
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1" # 通常最好同时设置这两个以保证兼容性
```

如果遇到一下问题：

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

解决方法

```shell
ln -s /root/PROJECT/ServerlessLLM-NPU/sllm_store/build/lib.linux-aarch64-cpython-310/sllm_store/libglog.so \
      /root/PROJECT/ServerlessLLM-NPU/sllm_store/sllm_store/libglog.so
```