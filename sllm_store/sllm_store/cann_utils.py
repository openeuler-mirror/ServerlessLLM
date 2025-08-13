"""CANN utilities for ServerlessLLM"""
import torch

# Global variable to track torch_npu import
_torch_npu_imported = False

def is_cann_available():
    """Check if CANN/NPU is available"""
    global _torch_npu_imported
    try:
        # Check if torch_npu is available and import it
        import torch_npu
        _torch_npu_imported = True
        return torch.npu.is_available()
    except ImportError:
        return False

def ensure_torch_npu_imported():
    """Ensure torch_npu is imported before any tensor operations"""
    global _torch_npu_imported
    if not _torch_npu_imported and is_cann_available():
        try:
            import torch_npu
            _torch_npu_imported = True
            print("torch_npu imported and PrivateUse1HooksInterface registered")
        except ImportError:
            pass
    return _torch_npu_imported

def get_device_type():
    """Get the device type (cuda or npu)"""
    if is_cann_available():
        return "npu"
    else:
        return "cuda"

def get_memory_functions():
    """Get the appropriate memory allocation functions"""
    if is_cann_available():
        try:
            from sllm_store._C import allocate_cann_memory, get_cann_memory_handles
            return allocate_cann_memory, _get_cann_memory_handles_with_pid
        except ImportError:
            raise ImportError("CANN functions not available in compiled extension")
    else:
        from sllm_store._C import allocate_cuda_memory, get_cuda_memory_handles
        return allocate_cuda_memory, get_cuda_memory_handles

def _get_cann_memory_handles_with_pid(memory_info_map):
    """
    Wrapper for get_cann_memory_handles that automatically gets the server PID
    """
    from sllm_store._C import get_cann_memory_handles
    from sllm_store.client import SllmStoreClient

    try:
        # Get server PID via gRPC
        client = SllmStoreClient("127.0.0.1:8073")
        server_pid = client.get_pid()

        if server_pid is None:
            print("Warning: Could not get server PID via gRPC, using default (-1)")
            server_pid = -1
        else:
            print(f"Got server PID via gRPC: {server_pid}")

        # Call the C++ function with the server PID
        return get_cann_memory_handles(memory_info_map, server_pid)

    except Exception as e:
        print(f"Error getting server PID: {e}")
        print("Falling back to default PID (-1)")
        return get_cann_memory_handles(memory_info_map, -1)

def init_npu_backend():
    """Initialize NPU backend and register with PyTorch"""
    if not is_cann_available():
        return False

    try:
        import torch_npu

        # Register NPU as a custom backend for PrivateUse1
        # This is essential for PyTorch to recognize NPU devices
        try:
            # Try the newer API first (PyTorch 2.1+)
            if hasattr(torch.utils, 'rename_privateuse1_backend'):
                torch.utils.rename_privateuse1_backend("npu")
            elif hasattr(torch, 'rename_privateuse1_backend'):
                torch.rename_privateuse1_backend("npu")
            else:
                print("Warning: rename_privateuse1_backend not available, using fallback initialization")
        except Exception as e:
            print(f"Warning: Failed to register NPU backend name: {e}")

        # Initialize NPU context
        # This ensures NPU devices are properly initialized
        if torch_npu.npu.device_count() > 0:
            # Initialize NPU runtime
            try:
                # Try to register the NPU module with torch
                if hasattr(torch, '_register_device_module'):
                    torch._register_device_module('npu', torch_npu.npu)
            except Exception as e:
                print(f"Warning: Failed to register NPU module: {e}")

            # Warm up NPU devices to ensure proper initialization
            for i in range(torch_npu.npu.device_count()):
                try:
                    # Create a small tensor to initialize the device
                    x = torch.empty(1, device=f"npu:{i}")
                    del x  # Clean up
                    torch_npu.npu.synchronize()
                except Exception as e:
                    print(f"Warning: Failed to initialize NPU device {i}: {e}")
                    continue

            print(f"NPU backend initialized with {torch_npu.npu.device_count()} devices")
            return True
        else:
            print("No NPU devices found")
            return False

    except Exception as e:
        print(f"Failed to initialize NPU backend: {e}")
        import traceback
        traceback.print_exc()
        return False

# Initialize NPU backend when module is imported
_npu_initialized = False

def ensure_npu_initialized():
    """Ensure NPU backend is initialized"""
    global _npu_initialized
    if not _npu_initialized and is_cann_available():
        _npu_initialized = init_npu_backend()
    return _npu_initialized

# Automatically initialize NPU when module is imported
if is_cann_available():
    try:
        ensure_npu_initialized()
    except Exception as e:
        print(f"Warning: Failed to auto-initialize NPU backend: {e}")
        pass