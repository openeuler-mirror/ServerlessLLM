// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//              http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ----------------------------------------------------------------------------
#include "gpu_replica.h"

#ifdef USE_CANN
#include "acl/acl.h"
#include "../checkpoint/cann_ipc.h"
#else
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>

void GpuReplica::Clear() {
    // Iterate through all device IDs and their corresponding lists of device pointers.
    for (auto& [device_id, device_ptr] : device_ptrs_) {
#ifdef USE_CANN
        // Set the current NPU device context for the current thread.
        aclError ret = aclrtSetDevice(device_id);
        if (ret != ACL_ERROR_NONE) {
            LOG(ERROR) << "Failed to set device " << device_id << " error: " << ret;
        }

        // Iterate through each device pointer that was opened via IPC and close its handle.
        aclError close_ret = cannIpcCloseMemHandle(device_ptr);
        if (close_ret != ACL_ERROR_NONE) {
            LOG(ERROR) << "Failed to close CANN IPC memory handle for device " << device_id
                       << " pointer " << device_ptr << ", error: " << close_ret;
        } else {
            LOG(INFO) << "Successfully closed CANN IPC memory handle for device " << device_id
                      << " pointer " << device_ptr;
        }
#else
        // For CUDA, set the device and close IPC memory handles.
        cudaSetDevice(device_id);
        cudaError_t err = cudaIpcCloseMemHandle(device_ptr);
        if (err != cudaSuccess) {
            LOG(ERROR) << "Failed to close memory handle for device " << device_id
                       << " error: " << cudaGetErrorString(err);
        }
#endif
    }
    // Clear various internal data structures associated with the GPU replica.
    gpu_loading_queue_.clear(); // Clear the queue of tensors waiting to be loaded to GPU.
    tensor_offsets_.clear();    // Clear the map storing tensor offsets within the GPU memory.

    // Update the memory state to INTERRUPTED, indicating that the replica's memory is no longer valid.
    state_ = MemoryState::INTERRUPTED;

    // Notify all waiting threads that the state has changed. This is typically used in conjunction
    // with a condition variable (cv_) to unblock threads that might be waiting for the GPU replica
    // to be in a specific state (e.g., loaded or ready).
    cv_.notify_all();
}