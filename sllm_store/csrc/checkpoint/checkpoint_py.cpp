// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//
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
#include <torch/extension.h>

#include "checkpoint.h"

#include <iostream>
#include <unistd.h> // For getpid() and sleep()

namespace py = pybind11;

void start_profiling() {
    std::cout << "PID: " << getpid() << std::endl;
    sleep(30); // Pause for 30 seconds to allow perf recording
}

// define pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("start_profiling", &start_profiling, "Prints PID and pauses for profiling");

  m.def("save_tensors", &SaveTensors, "Save a state dict")
      .def("restore_tensors", &RestoreTensors, "Restore a state dict");

#ifdef USE_CANN
  // CANN-specific bindings
  m.def("allocate_cann_memory", &AllocateCannMemory, "Allocate CANN memory")
      .def(
          "get_cann_memory_handles",
          [](std::unordered_map<int, std::pair<void*, size_t>>& memory_info_map,
             int32_t target_process_id = -1) {
            std::unordered_map<int, std::string> memory_handles =
                GetCannMemoryHandles(memory_info_map, target_process_id);

            std::unordered_map<int, py::bytes> py_memory_handles;
            for (const auto& kv : memory_handles) {
              py_memory_handles[kv.first] = py::bytes(kv.second);
            }

            py::dict updated_ptrs;
            for (const auto& kv : memory_info_map) {
              updated_ptrs[py::int_(kv.first)] = py::int_(reinterpret_cast<uintptr_t>(kv.second.first));
            }

            return py::make_tuple(py_memory_handles, updated_ptrs);
          },
          py::arg("memory_info_map"), py::arg("target_process_id") = -1, "Get CANN memory handles")
      .def(
          "get_cann_memory_handles",
          [](std::unordered_map<int, std::vector<std::pair<void*, size_t>>>& memory_info_vectors,
             int32_t target_process_id = -1) {
            auto memory_handles = GetCannMemoryHandles(memory_info_vectors, target_process_id);

            std::unordered_map<int, std::vector<py::bytes>> py_memory_handles;
            for (const auto& kv : memory_handles) {
              std::vector<py::bytes> handles;
              for (const auto& handle : kv.second) {
                handles.push_back(py::bytes(handle));
              }
              py_memory_handles[kv.first] = handles;
            }

            py::dict updated_ptrs;
            for (const auto& kv : memory_info_vectors) {
              py::list ptr_list;
              for (const auto& pair : kv.second) {
                ptr_list.append(py::int_(reinterpret_cast<uintptr_t>(pair.first)));
              }
              updated_ptrs[py::int_(kv.first)] = ptr_list;
            }

            return py::make_tuple(py_memory_handles, updated_ptrs);
          },
          py::arg("memory_info_vectors"), py::arg("target_process_id") = -1, "Get CANN memory handles")
      .def("create_pointer_capsule", [](uintptr_t ptr) -> py::capsule {
          return py::capsule(reinterpret_cast<void*>(ptr));
      }, "Create a capsule from uintptr_t");
#else
  // CUDA-specific bindings
  m.def("allocate_cuda_memory", &AllocateCudaMemory, "Allocate cuda memory")
      .def(
          "get_cuda_memory_handles",
          [](const std::unordered_map<int, void*>& memory_ptrs) {
            std::unordered_map<int, std::string> memory_handles =
                GetCudaMemoryHandles(memory_ptrs);

            std::unordered_map<int, py::bytes> py_memory_handles;
            for (const auto& kv : memory_handles) {
              py_memory_handles[kv.first] = py::bytes(kv.second);
            }
            return py_memory_handles;
          },
          py::arg("memory_ptrs"), "Get cuda memory handles")
      .def(
          "get_cuda_memory_handles",
          [](const std::unordered_map<int, std::vector<void*>>& memory_ptrs) {
            auto memory_handles = GetCudaMemoryHandles(memory_ptrs);

            std::unordered_map<int, std::vector<py::bytes>> py_memory_handles;
            for (const auto& kv : memory_handles) {
              std::vector<py::bytes> handles;
              for (const auto& handle : kv.second) {
                handles.push_back(py::bytes(handle));
              }
              py_memory_handles[kv.first] = handles;
            }
            return py_memory_handles;
          },
          py::arg("memory_ptrs"), "Get cuda memory handles")
      .def(
          "get_cuda_memory_handles",
          [](const std::unordered_map<int, std::vector<uint64_t>>&
               memory_ptrs) {
            std::unordered_map<int, std::vector<void*>> memory_ptrs_void;
            for (const auto& kv : memory_ptrs) {
              std::vector<void*> ptrs;
              for (const auto& ptr : kv.second) {
                ptrs.push_back(reinterpret_cast<void*>(ptr));
              }
              memory_ptrs_void[kv.first] = ptrs;
            }
            auto memory_handles = GetCudaMemoryHandles(memory_ptrs_void);

            std::unordered_map<int, std::vector<py::bytes>> py_memory_handles;
            for (const auto& kv : memory_handles) {
              std::vector<py::bytes> handles;
              for (const auto& handle : kv.second) {
                handles.push_back(py::bytes(handle));
              }
              py_memory_handles[kv.first] = handles;
            }
            return py_memory_handles;
          },
          py::arg("memory_ptrs"), "Get cuda memory handles");
#endif

  m.def("get_device_uuid_map", &GetDeviceUuidMap, "Get device uuid map");
}