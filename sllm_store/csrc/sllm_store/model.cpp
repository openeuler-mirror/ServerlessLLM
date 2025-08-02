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
#include "model.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <malloc.h>  // for posix_memalign <- this fix O_DIRECT issue
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Third-party library headers
#ifdef USE_CANN
#include "acl/acl.h"
#else
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>

#ifdef USE_CANN
#include "cann_error_handling.h"
#include "../checkpoint/cann_ipc.h"
#else
#include "error_handling.h"
#endif

int Model::Initialize(const std::filesystem::path storage_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != MemoryState::UNINITIALIZED) {
        return 0;
    }
    model_size_ = 0;
    partition_sizes_.clear();
    partition_paths_.clear();
    // Attempt to read from 0 until the file is not found
    for (int partition_id = 0;; ++partition_id) {
        auto tensor_path = storage_path / model_path_ /
                           ("tensor.data_" + std::to_string(partition_id));
        if (access(tensor_path.c_str(), F_OK) == -1) {
            LOG(INFO) << "Tensor file " << tensor_path << " does not exist";
            break;
        }
        struct stat st;
        if (stat(tensor_path.c_str(), &st) != 0) {
            LOG(ERROR) << "Failed to get file size of " << tensor_path;
            return -1;
        }
        model_size_ += st.st_size;
        partition_sizes_.push_back(st.st_size);
        partition_paths_.push_back(tensor_path);
    }
    if (model_size_ == 0) {
        LOG(ERROR) << "Model " << model_path_ << " does not exist";
        return -1;
    }
    state_ = MemoryState::UNALLOCATED;

    return 0;
}

int Model::ToHost(int num_threads) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ != MemoryState::ALLOCATED) {
        if (state_ == MemoryState::LOADING || state_ == MemoryState::LOADED) {
            return 0;
        } else {
            LOG(ERROR) << "Model " << model_path_ << " is at state " << state_;
            return -1;
        }
    }

    LOG(INFO) << "Loading model " << model_path_ << " size " << model_size_
              << " to host";
    if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
        LOG(ERROR) << "CPU memory not allocated";
        return 1;
    }

    auto host_buffers = pinned_mem_->get();
    size_t num_chunks = pinned_mem_->num_chunks();
    size_t chunk_size = pinned_mem_->chunk_size();
    host_ptr_vector_ = std::make_shared<BatchVector>();
    host_ptr_vector_->init("queue_name", num_chunks);
    std::vector<std::future<int>> futures;
    size_t chunk_per_thread = (num_chunks + num_threads - 1) / num_threads;
    LOG(INFO) << "Loading model " << model_path_ << " to host with "
              << num_threads << " threads, " << num_chunks << " chunks, "
              << chunk_size << " chunk size, " << chunk_per_thread
              << " chunks per thread";

    state_ = MemoryState::LOADING;
    lock.unlock();

    for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        futures.emplace_back(std::async(std::launch::async, [&, thread_idx]() {
            LOG(INFO) << "=== Thread " << thread_idx << " START ===";

            // Each thread opens its own file descriptors
            std::vector<int> thread_fds;
            LOG(INFO) << "Thread " << thread_idx << ": Opening "
                      << partition_sizes_.size() << " partition files";

            for (int partition_id = 0; partition_id < partition_sizes_.size();
                 ++partition_id) {
                auto tensor_path = partition_paths_[partition_id];
                LOG(INFO) << "Thread " << thread_idx << ": Opening partition "
                          << partition_id << ": " << tensor_path;

                // Check file exists first
                if (access(tensor_path.c_str(), F_OK) == -1) {
                    LOG(ERROR) << "Thread " << thread_idx << ": File " << tensor_path
                               << " does not exist (access check failed)";
                    for (int cleanup_fd : thread_fds) close(cleanup_fd);
                    return -1;
                }

                // * If using O_DIRECT, will cause pread() FILE EXISTS error
                int fd = open(tensor_path.c_str(), O_DIRECT | O_RDONLY);
                if (fd < 0) {
                    int err_no = errno;
                    LOG(ERROR) << "Thread " << thread_idx << ": open() failed for "
                               << tensor_path << ", errno: " << err_no
                               << ", error: " << strerror(err_no);
                    for (int cleanup_fd : thread_fds) close(cleanup_fd);
                    return -1;
                }

                // // Get block size for alignment
                // struct stat st;
                // if (fstat(fd, &st) != 0) {
                //     int err_no = errno;
                //     LOG(ERROR) << "Thread " << thread_idx << ": fstat failed for "
                //                << tensor_path << ", errno: " << err_no
                //                << ", error: " << strerror(err_no);
                //     close(fd);
                //     for (int cleanup_fd : thread_fds) close(cleanup_fd);
                //     return -1;
                // }
                // size_t alignment = st.st_blksize;
                // if (alignment == 0) {
                //     alignment = 4096;  // Fallback to common page size
                //     LOG(WARNING) << "Thread " << thread_idx << ": fstat returned 0 blksize, using fallback 4096";
                // }
                // LOG(INFO) << "Thread " << thread_idx << ": Using alignment/block size=" << alignment << " for " << tensor_path;

                // // Test pread with aligned buffer and size
                // void* test_buf;
                // if (posix_memalign(&test_buf, alignment, alignment) != 0) {
                //     LOG(ERROR) << "Thread " << thread_idx << ": posix_memalign failed for test buffer";
                //     close(fd);
                //     for (int cleanup_fd : thread_fds) close(cleanup_fd);
                //     return -1;
                // }
                // ssize_t test_ret = pread(fd, test_buf, alignment, 0);
                // free(test_buf);
                // if (test_ret < 0) {
                //  int err_no = errno;
                //  LOG(ERROR) << "Thread " << thread_idx << ": immediate pread() failed for "
                //             << tensor_path << ", errno: " << err_no
                //             << ", error: " << strerror(err_no);
                //  close(fd);  // Clean up this fd before returning
                //  for (int cleanup_fd : thread_fds) close(cleanup_fd);
                //  return -1;
                // } else {
                //  LOG(INFO) << "Thread " << thread_idx << ": immediate pread() succeeded, read "
                //            << test_ret << " bytes";
                // }

                LOG(INFO) << "Thread " << thread_idx << ": Successfully opened "
                          << "partition " << partition_id << " with fd=" << fd
                          << ", file size=" << partition_sizes_[partition_id]
                          << " bytes";
                thread_fds.push_back(fd);
            }

            size_t partition_id = 0;
            size_t file_offset = thread_idx * chunk_per_thread * chunk_size;
            LOG(INFO) << "Thread " << thread_idx
                      << ": Initial file_offset=" << file_offset;

            // Calculate starting partition
            size_t original_offset = file_offset;
            while (partition_id < partition_sizes_.size() &&
                   file_offset >= partition_sizes_.at(partition_id)) {
                LOG(INFO) << "Thread " << thread_idx << ": Skipping partition "
                          << partition_id << " (size=" << partition_sizes_.at(partition_id)
                          << "), remaining offset=" << file_offset;
                file_offset -= partition_sizes_.at(partition_id);
                partition_id += 1;
            }

            if (partition_id >= partition_sizes_.size()) {
                LOG(INFO) << "Thread " << thread_idx << " early exits (offset "
                          << original_offset << " beyond all partitions)";
                for (int fd : thread_fds) close(fd);
                return 0;
            }

            LOG(INFO) << "Thread " << thread_idx << " starting from partition "
                      << partition_id << " offset " << file_offset
                      << " (adjusted from " << original_offset << ")";

            size_t total_bytes_read = 0;
            size_t chunks_processed = 0;

            for (size_t chunk_idx = thread_idx * chunk_per_thread;
                 chunk_idx < (thread_idx + 1) * chunk_per_thread &&
                 chunk_idx < num_chunks;
                 ++chunk_idx) {
                size_t size = std::min(chunk_size, model_size_ - chunk_idx * chunk_size);

                LOG(INFO) << "Thread " << thread_idx << ": Processing chunk " << chunk_idx
                          << "/" << num_chunks << ", size=" << size << " bytes";

                if (host_buffers[chunk_idx] == nullptr) {
                    LOG(ERROR) << "Thread " << thread_idx << ": Host buffer[" << chunk_idx
                               << "] is null!";
                    for (int fd : thread_fds) close(fd);
                    return -1;
                }

                if (state_ == MemoryState::CANCELLED) {
                    LOG(INFO) << "Thread " << thread_idx << ": Loading cancelled, processed "
                              << chunks_processed << " chunks";
                    for (int fd : thread_fds) close(fd);
                    return 0;
                }

                size_t remaining_bytes = size;
                size_t buffer_offset = 0;
                size_t chunk_bytes_read = 0;

                while (remaining_bytes > 0) {
                    if (partition_id >= partition_sizes_.size()) {
                        LOG(ERROR) << "Thread " << thread_idx
                                   << ": Unexpected end of partitions at partition "
                                   << partition_id << " while reading chunk " << chunk_idx;
                        for (int fd : thread_fds) close(fd);
                        return -1;
                    }

                    int fd = thread_fds[partition_id];

                    // Get alignment again (in case per-partition, but it's per-fd)
                    struct stat st;
                    if (fstat(fd, &st) != 0) {
                        int err_no = errno;
                        LOG(ERROR) << "Thread " << thread_idx << ": fstat failed during read";
                        for (int fd : thread_fds) close(fd);
                        return -1;
                    }
                    size_t alignment = st.st_blksize;
                    if (alignment == 0) alignment = 4096;

                    size_t bytes_left_in_partition = partition_sizes_.at(partition_id) - file_offset;
                    size_t read_size = std::min(remaining_bytes, bytes_left_in_partition);

                    LOG(INFO) << "Thread " << thread_idx << ": Reading " << read_size
                              << " bytes from partition " << partition_id << " at offset "
                              << file_offset << " (fd=" << fd << ")";

                    // O_DIRECT aligned read using temp buffer
                    size_t offset_misalign = file_offset % alignment;
                    size_t aligned_offset = file_offset - offset_misalign;
                    size_t aligned_read_size = ((read_size + offset_misalign + alignment - 1) / alignment) * alignment;

                    void* temp_buf;
                    if (posix_memalign(&temp_buf, alignment, aligned_read_size) != 0) {
                        LOG(ERROR) << "Thread " << thread_idx << ": posix_memalign failed for read buffer";
                        for (int fd : thread_fds) close(fd);
                        return -1;
                    }

                    ssize_t ret = pread(fd, temp_buf, aligned_read_size, aligned_offset);
                    if (ret < 0) {
                        int err_no = errno;
                        auto tensor_path = partition_paths_[partition_id];
                        LOG(ERROR) << "Thread " << thread_idx << ": pread() failed for file: "
                                   << tensor_path << ", errno: " << err_no
                                   << ", error: " << strerror(err_no) << ", fd=" << fd
                                   << ", offset=" << aligned_offset << ", size=" << aligned_read_size;
                        free(temp_buf);
                        for (int fd : thread_fds) close(fd);
                        return -1;
                    }
                    if (ret < (ssize_t)(read_size + offset_misalign)) {
                        LOG(ERROR) << "Thread " << thread_idx << ": Short read for file "
                                   << partition_paths_[partition_id] << ": expected at least "
                                   << (read_size + offset_misalign) << " bytes, got " << ret << " (fd=" << fd
                                   << ", offset=" << aligned_offset << ")";
                        free(temp_buf);
                        for (int fd : thread_fds) close(fd);
                        return -1;
                    }

                    // Copy the relevant part to the host buffer
                    memcpy((void*)(host_buffers[chunk_idx] + buffer_offset),
                           (char*)temp_buf + offset_misalign,
                           read_size);
                    free(temp_buf);

                    // Simulate ret as read_size for the rest of the code
                    ret = read_size;

                    // Sample first few bytes to verify data is non-zero
                    if (buffer_offset == 0 && chunk_idx % 10 == 0) {
                        unsigned char* data_ptr = (unsigned char*)(host_buffers[chunk_idx]);
                        LOG(INFO) << "Thread " << thread_idx << ": Chunk " << chunk_idx
                                  << " first 16 bytes: " << std::hex << std::setfill('0');
                        for (int i = 0; i < std::min(16, (int)ret); i++) {
                            std::cout << std::setw(2) << (int)data_ptr[i] << " ";
                        }
                        std::cout << std::dec << std::endl;
                    }

                    remaining_bytes -= ret;
                    buffer_offset += ret;
                    file_offset += ret;
                    chunk_bytes_read += ret;
                    total_bytes_read += ret;

                    // Move to next partition if current one is exhausted
                    if (file_offset >= partition_sizes_.at(partition_id)) {
                        LOG(INFO) << "Thread " << thread_idx << ": Finished partition "
                                  << partition_id << ", moving to partition "
                                  << (partition_id + 1);
                        partition_id += 1;
                        file_offset = 0;
                    }
                }

                LOG(INFO) << "Thread " << thread_idx << ": Chunk " << chunk_idx
                          << " completed, read " << chunk_bytes_read << " bytes";

                host_ptr_vector_->enqueue(chunk_idx, Batch{chunk_idx, size});
                chunks_processed++;
            }

            // Close thread-specific file descriptors
            LOG(INFO) << "Thread " << thread_idx << ": Closing "
                      << thread_fds.size() << " file descriptors";
            for (int i = 0; i < thread_fds.size(); i++) {
                close(thread_fds[i]);
                LOG(INFO) << "Thread " << thread_idx << ": Closed fd=" << thread_fds[i]
                          << " for partition " << i;
            }

            LOG(INFO) << "=== Thread " << thread_idx << " COMPLETE: processed "
                      << chunks_processed << " chunks, read " << total_bytes_read
                      << " bytes total ===";
            return 0;
        }));
    }

    bool error = false;
    for (auto& future : futures) {
        int ret = future.get();
        if (ret != 0) {
            LOG(ERROR) << "Error reading from disk, ret " << ret;
            error = true;
        }
    }

    lock.lock();
    if (error) {
        state_ = MemoryState::INTERRUPTED;
        // Deal with gpu replicas
        for (auto& [replica_uuid, gpu_replica] : gpu_replicas_) {
            if (gpu_replica->state_ == MemoryState::LOADING) {
                gpu_replica->state_ = MemoryState::CANCELLED;
                gpu_replica->cv_.notify_all();
            }
            // wait for gpu replicas to finish
            gpu_replica->cv_.wait(lock, [&gpu_replica] {
                return gpu_replica->state_ == MemoryState::LOADED ||
                       gpu_replica->state_ == MemoryState::INTERRUPTED;
            });
            // Note: gpu replicas will be handled by the caller
        }
        // release pinned memory
        pinned_mem_.reset();
        state_ = MemoryState::UNALLOCATED;

        return -1;
    }

    // === HOST MEMORY VALIDATION DEBUG ===
    LOG(INFO) << "=== HOST MEMORY VALIDATION START ===";
    LOG(INFO) << "Validating host memory after file loading completion";
    LOG(INFO) << "Total chunks: " << num_chunks << ", chunk size: " << chunk_size;

    size_t total_non_zero_bytes = 0;
    size_t total_zero_bytes = 0;

    // Sample multiple chunks across the entire model
    std::vector<size_t> sample_chunks;
    sample_chunks.push_back(0);            // First chunk
    sample_chunks.push_back(num_chunks / 4);   // 25% through
    sample_chunks.push_back(num_chunks / 2);   // 50% through
    sample_chunks.push_back(3 * num_chunks / 4); // 75% through
    sample_chunks.push_back(num_chunks - 1);   // Last chunk

    for (size_t chunk_idx : sample_chunks) {
        if (chunk_idx >= num_chunks) continue;

        LOG(INFO) << "Sampling chunk " << chunk_idx << "/" << num_chunks;
        unsigned char* chunk_data = (unsigned char*)host_buffers[chunk_idx];

        if (chunk_data == nullptr) {
            LOG(ERROR) << "Host buffer[" << chunk_idx << "] is null!";
            continue;
        }

        size_t actual_chunk_size = std::min(chunk_size, model_size_ - chunk_idx * chunk_size);
        LOG(INFO) << "Chunk " << chunk_idx << " size: " << actual_chunk_size << " bytes";

        // Count zero vs non-zero bytes in first 1KB
        size_t sample_size = std::min((size_t)1024, actual_chunk_size);
        size_t chunk_zero_bytes = 0;
        size_t chunk_non_zero_bytes = 0;

        for (size_t i = 0; i < sample_size; i++) {
            if (chunk_data[i] == 0) {
                chunk_zero_bytes++;
            } else {
                chunk_non_zero_bytes++;
            }
        }

        total_zero_bytes += chunk_zero_bytes;
        total_non_zero_bytes += chunk_non_zero_bytes;

        LOG(INFO) << "Chunk " << chunk_idx << " sample (first " << sample_size
                  << " bytes): " << chunk_non_zero_bytes << " non-zero, "
                  << chunk_zero_bytes << " zero bytes";

        // Show hex dump of first 64 bytes
        LOG(INFO) << "Chunk " << chunk_idx << " first 64 bytes (hex):";
        for (size_t i = 0; i < std::min((size_t)64, actual_chunk_size); i += 16) {
            std::cout << "  " << std::hex << std::setfill('0') << std::setw(8) << i << ": ";
            for (size_t j = 0; j < 16 && (i + j) < std::min((size_t)64, actual_chunk_size); j++) {
                std::cout << std::setw(2) << (int)chunk_data[i + j] << " ";
            }
            std::cout << std::dec << std::endl;
        }

        // Show as float16 values (first 32 bytes = 16 float16 values)
        if (actual_chunk_size >= 32) {
            LOG(INFO) << "Chunk " << chunk_idx << " first 16 float16 values:";
            uint16_t* float16_data = (uint16_t*)chunk_data;
            for (int i = 0; i < 16; i++) {
                // Convert float16 to float32 for display
                uint32_t f32_bits = ((uint32_t)(float16_data[i] & 0x8000)) << 16;  // Sign
                if ((float16_data[i] & 0x7c00) != 0) {  // Not zero/denormal
                    f32_bits |= (((uint32_t)(float16_data[i] & 0x7c00) + 0x1c000)) << 13;  // Exponent
                    f32_bits |= ((uint32_t)(float16_data[i] & 0x03ff)) << 13;  // Mantissa
                }
                float f32_val = *(float*)&f32_bits;
                std::cout << f32_val << " ";
            }
            std::cout << std::endl;
        }
    }

    LOG(INFO) << "HOST MEMORY SUMMARY:";
    LOG(INFO) << "  Total sampled bytes: " << (total_zero_bytes + total_non_zero_bytes);
    LOG(INFO) << "  Non-zero bytes: " << total_non_zero_bytes;
    LOG(INFO) << "  Zero bytes: " << total_zero_bytes;
    LOG(INFO) << "  Non-zero percentage: "
              << (100.0 * total_non_zero_bytes / (total_zero_bytes + total_non_zero_bytes))
              << "%";

    if (total_non_zero_bytes == 0) {
        LOG(ERROR) << "*** CRITICAL: ALL HOST MEMORY IS ZERO! File loading failed! ***";
    } else if (total_non_zero_bytes < (total_zero_bytes + total_non_zero_bytes) / 10) {
        LOG(WARNING) << "*** WARNING: Less than 10% non-zero data in host memory ***";
    } else {
        LOG(INFO) << "*** HOST MEMORY VALIDATION PASSED: Contains non-zero data ***";
    }

    LOG(INFO) << "=== HOST MEMORY VALIDATION END ===";

    state_ = MemoryState::LOADED;
    LOG(INFO) << "Finished loading model " << model_path_ << " from disk";

    return 0;
}

int Model::ToGpu(const std::string& replica_uuid,
                 const MemPtrListMap& device_ptrs,
                 const std::unordered_map<int, MemCopyChunkList>& mem_copy_chunks,
                 const std::unordered_map<int, MemCopyHandleList>& mem_copy_handles) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ == MemoryState::UNINITIALIZED) {
        LOG(ERROR) << "Model " << model_path_ << " is not initialized";
        return -1;
    }

    if (gpu_replicas_.find(replica_uuid) != gpu_replicas_.end()) {
        LOG(ERROR) << "Replica " << replica_uuid << " already exists";
        return -1;
    }
    LOG(INFO) << "Creating replica " << replica_uuid;
    gpu_replicas_.emplace(replica_uuid, std::make_shared<GpuReplica>());
    GpuReplicaPtr gpu_replica = gpu_replicas_.at(replica_uuid);
    for (const auto& [device_id, _] : device_ptrs) {
        LOG(INFO) << "Creating queue for device " << device_id;
        gpu_replica->gpu_loading_queue_.emplace(device_id,
                                                std::make_shared<BatchQueue>());
    }
    gpu_replica->device_ptrs_ = device_ptrs;
    gpu_replica->state_ = MemoryState::LOADING;
    LOG(INFO) << "Created replica " << replica_uuid;
    cv_.notify_all();
    lock.unlock();

    // Start a dispatcher first
    auto dispatch_future = std::async(
        std::launch::async,
        [this, gpu_replica, mem_copy_chunks, mem_copy_handles]() {
            return DispatchToGpu(gpu_replica, mem_copy_chunks, mem_copy_handles);
        });

    LOG(INFO) << "Dispatcher started for model " << model_path_;

    std::unordered_map<int, std::future<int>> futures;
    for (auto& [device_id, device_ptr_list] : device_ptrs) {
        futures.emplace(
            device_id, std::async(std::launch::async, [this, gpu_replica, device_id,
                                                         device_ptr_list]() {
                auto gpu_loading_queue =
                    gpu_replica->gpu_loading_queue_.at(device_id);
                if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
                    LOG(ERROR) << "CPU memory not allocated";
                    return 1;
                }

#ifdef USE_CANN
                aclError ret = aclrtSetDevice(device_id);
                if (ret != ACL_ERROR_NONE) {
                    LOG(ERROR) << "Error setting device " << ret;
                    return 1;
                }
#else
                cudaError_t err = cudaSetDevice(device_id);
                if (err != cudaSuccess) {
                    LOG(ERROR) << "Error setting device " << cudaGetErrorString(err);
                    return 1;
                }
#endif

                auto& host_buffers = pinned_mem_->get();

                size_t loaded_size = 0;
                size_t total_chunks_processed = 0;
                LOG(INFO) << "=== NPU LOADING DEBUG: Device " << device_id << " START ===";

                while (true) {
                    auto [chunk_id, chunk_offset, size, gpu_offset, handle_idx] =
                        gpu_loading_queue->dequeue();
                    if (size == 0) {
                        LOG(INFO) << "=== NPU LOADING DEBUG: Device " << device_id
                                  << " - End signal received, processed " << total_chunks_processed << " chunks ===";
                        break;
                    }
                    if (gpu_replica->state_ == MemoryState::CANCELLED) {
                        LOG(INFO) << "Loading from mem for model " << model_path_
                                  << " is cancelled,"
                                  << " chunk " << chunk_id << " offset "
                                  << " size " << size;
                        return 0;
                    }

#ifdef USE_CANN
                    aclrtSetDevice(device_id);

                    if (device_ptr_list[handle_idx] == nullptr) {
                        LOG(ERROR) << "Device pointer is null for handle " << handle_idx << " on Device " << device_id;
                        return 1;
                    }

                    // Debug: Sample host data before copying
                    if (total_chunks_processed % 50 == 0) {  // Sample every 50th chunk
                        LOG(INFO) << "NPU COPY DEBUG: Chunk " << chunk_id << ", offset " << chunk_offset
                                  << ", size " << size << ", gpu_offset " << gpu_offset;

                        unsigned char* host_data = (unsigned char*)(host_buffers[chunk_id] + chunk_offset);
                        LOG(INFO) << "Host data before copy (first 32 bytes):";
                        for (int i = 0; i < std::min(32, (int)size); i += 16) {
                            std::cout << "  " << std::hex << std::setfill('0') << std::setw(8) << i << ": ";
                            for (int j = 0; j < 16 && (i + j) < std::min(32, (int)size); j++) {
                                std::cout << std::setw(2) << (int)host_data[i + j] << " ";
                            }
                            std::cout << std::dec << std::endl;
                        }
                    }

                    aclError ret = aclrtMemcpy(
                        (void*)((char*)device_ptr_list[handle_idx] + gpu_offset),
                        size,
                        (const void*)(host_buffers[chunk_id] + chunk_offset),
                        size,
                        ACL_MEMCPY_HOST_TO_DEVICE);
                    if (ret != 0) {
                        LOG(ERROR) << "device_ptr_list[handle_idx]" << (void*)((char*)device_ptr_list[handle_idx] + gpu_offset)
                                   << "host buffer" << (const void*)(host_buffers[chunk_id] + chunk_offset) << "size" << size;

                        LOG(ERROR) << "Failed to copy memory from host to device "
                                   << device_id << " error: " << ret;
                        return 1;
                    }

                    // // Debug: Verify NPU data after copying
                    // if (total_chunks_processed % 50 == 0) {  // Sample every 50th chunk
                    //   LOG(INFO) << "NPU COPY SUCCESS: Host→NPU copy completed for chunk " << chunk_id;

                    //   // Sample NPU memory to verify the copy worked
                    //   char npu_sample[32];
                    //   aclError sample_ret = aclrtMemcpy(
                    //       npu_sample,
                    //       sizeof(npu_sample),
                    //       (void*)((char*)device_ptr_list[handle_idx] + gpu_offset),
                    //       std::min((size_t)32, size),
                    //       ACL_MEMCPY_DEVICE_TO_HOST);

                    //   if (sample_ret == ACL_ERROR_NONE) {
                    //     LOG(INFO) << "NPU data after copy (first 32 bytes):";
                    //     for (int i = 0; i < std::min(32, (int)size); i += 16) {
                    //       std::cout << "  " << std::hex << std::setfill('0') << std::setw(8) << i << ": ";
                    //       for (int j = 0; j < 16 && (i + j) < std::min(32, (int)size); j++) {
                    //         std::cout << std::setw(2) << (int)(unsigned char)npu_sample[i + j] << " ";
                    //       }
                    //       std::cout << std::dec << std::endl;
                    //     }

                    //     // Compare host vs NPU data
                    //     unsigned char* host_data = (unsigned char*)(host_buffers[chunk_id] + chunk_offset);
                    //     bool data_matches = true;
                    //     for (int i = 0; i < std::min(32, (int)size); i++) {
                    //       if (host_data[i] != (unsigned char)npu_sample[i]) {
                    //         data_matches = false;
                    //         break;
                    //       }
                    //     }

                    //     if (data_matches) {
                    //       LOG(INFO) << "✅ HOST → NPU DATA VERIFICATION PASSED for chunk " << chunk_id;
                    //     } else {
                    //       LOG(ERROR) << "❌ HOST → NPU DATA MISMATCH for chunk " << chunk_id;
                    //     }
                    //   } else {
                    //     LOG(ERROR) << "Failed to sample NPU memory for verification, error: " << sample_ret;
                    //   }
                    // }
#else
                    CUDA_CHECK(
                        cudaMemcpy(
                            (void*)((char*)device_ptr_list[handle_idx] + gpu_offset),
                            (void*)(host_buffers[chunk_id] + chunk_offset), size,
                            cudaMemcpyHostToDevice),
                        "cudaMemcpy Error");
#endif
                    loaded_size += size;
                    total_chunks_processed++;
                }

                LOG(INFO) << "=== NPU LOADING COMPLETE: Device " << device_id
                          << " - Loaded " << loaded_size << " bytes in " << total_chunks_processed << " chunks ===";
                LOG(INFO) << "Finished loading tensor from memory to device "
                          << device_id;

                return 0;
            }));
    }

    LOG(INFO) << "Waiting for model " << model_path_ << " num tasks "
              << futures.size() << " state " << gpu_replica->state_;
    dispatch_future.wait();
    bool error = false;
    for (auto& [device_id, future] : futures) {
        int ret = future.get();
        if (ret != 0) {
            LOG(ERROR) << "Error copying to device " << device_id;
            error = true;
        }
    }

    lock.lock();
    futures.clear();

    if (error) {
        LOG(ERROR) << "Failed to load model " << model_path_;
        gpu_replica->state_ = MemoryState::INTERRUPTED;
    } else {
        gpu_replica->state_ = MemoryState::LOADED;
    }
    gpu_replica->cv_.notify_all();

    // TODO: move to background thread
    for (auto& [device_id, device_ptr_list] : gpu_replica->device_ptrs_) {
#ifdef USE_CANN
        aclrtSetDevice(device_id);
        for (auto device_ptr : device_ptr_list) {
            // CRITICAL FIX: Use real CANN IPC cleanup now that we have proper IPC
            aclError ret = cannIpcCloseMemHandle(device_ptr);
            if (ret != ACL_ERROR_NONE) {
                LOG(ERROR) << "Failed to close CANN IPC handle for device " << device_id
                           << " pointer " << device_ptr << ", error: " << ret;
            } else {
                LOG(INFO) << "Successfully closed CANN IPC handle for device " << device_id
                          << " pointer " << device_ptr;
            }
        }
#else
        cudaSetDevice(device_id);
        for (auto device_ptr : device_ptr_list) {
            cudaError_t err = cudaIpcCloseMemHandle(device_ptr);
            if (err != cudaSuccess) {
                LOG(ERROR) << "Failed to close memory handle for device " << device_id
                           << " error: " << cudaGetErrorString(err);
            }
        }
#endif
    }

    if (gpu_replica->state_ == MemoryState::INTERRUPTED) {
        LOG(ERROR) << "Model " << model_path_ << " replica " << replica_uuid
                   << " is interrupted";
        return -1;
    }

    return 0;
}

int Model::WaitInHost() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ < MemoryState::LOADED) {
        cv_.wait(lock, [this] {
            return state_ == MemoryState::LOADED ||
                   state_ == MemoryState::INTERRUPTED;
        });
    }

    if (state_ >= MemoryState::INTERRUPTED) {
        LOG(INFO) << "Model " << model_path_ << " is interrupted";
        return 1;
    }

    return 0;
}

int Model::WaitInGpu(const std::string& replica_uuid) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (gpu_replicas_.find(replica_uuid) == gpu_replicas_.end()) {
        cv_.wait(lock, [this, replica_uuid] {
            return gpu_replicas_.find(replica_uuid) != gpu_replicas_.end();
        });
    }

    auto& gpu_replica = gpu_replicas_.at(replica_uuid);

    if (gpu_replica->state_ < MemoryState::LOADED) {
        gpu_replica->cv_.wait(lock, [&gpu_replica] {
            return gpu_replica->state_ == MemoryState::LOADED ||
                   gpu_replica->state_ == MemoryState::INTERRUPTED;
        });
    }

    if (gpu_replica->state_ >= MemoryState::INTERRUPTED) {
        LOG(INFO) << "Model " << model_path_ << " is interrupted";
        return 1;
    }

    return 0;
}

int Model::FreeGpu(const std::string& replica_uuid) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (gpu_replicas_.find(replica_uuid) == gpu_replicas_.end()) {
        LOG(ERROR) << "Model " << model_path_ << " replica " << replica_uuid
                   << " is not registered";
        return -1;
    }

    auto& gpu_replica = gpu_replicas_.at(replica_uuid);
    if (gpu_replica->state_ == MemoryState::UNINITIALIZED) {
        LOG(WARNING) << "Model " << model_path_ << " replica " << replica_uuid
                     << " is not initialized";
        gpu_replicas_.erase(replica_uuid);
        return 0;
    }

    if (gpu_replica->state_ == MemoryState::LOADING) {
        LOG(INFO) << "Waiting for model " << model_path_ << " replica "
                  << replica_uuid << " to be loaded";
        gpu_replica->cv_.wait(lock, [&gpu_replica] {
            return gpu_replica->state_ == MemoryState::LOADED ||
                   gpu_replica->state_ == MemoryState::INTERRUPTED;
        });
    }

    gpu_replicas_.erase(replica_uuid);
    return 0;
}

int Model::FreeHost() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ == MemoryState::UNINITIALIZED) {
        LOG(WARNING) << "Model " << model_path_ << " is not initialized";
        return 1;
    }

    if (state_ == MemoryState::UNALLOCATED) {
        LOG(WARNING) << "Model " << model_path_ << " is not allocated";
        return 1;
    }

    if (state_ == MemoryState::LOADING) {
        LOG(INFO) << "Waiting for model " << model_path_ << " to be loaded";
        cv_.wait(lock, [this] {
            return state_ == MemoryState::LOADED ||
                   state_ == MemoryState::INTERRUPTED;
        });
    }

    // make sure no gpu replicas are loading
    for (auto& [replica_uuid, gpu_replica] : gpu_replicas_) {
        if (gpu_replica->state_ == MemoryState::LOADING) {
            LOG(INFO) << "Waiting for replica " << replica_uuid << " to be loaded";
            gpu_replica->cv_.wait(lock, [&gpu_replica] {
                return gpu_replica->state_ == MemoryState::LOADED ||
                       gpu_replica->state_ == MemoryState::INTERRUPTED;
            });
        }
    }

    // free pinned memory
    int freed_chunks = pinned_mem_->num_chunks();
    pinned_mem_.reset();
    state_ = MemoryState::UNALLOCATED;

    return 0;
}

int Model::TryFreeHost() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ == MemoryState::UNINITIALIZED) {
        LOG(WARNING) << "Model " << model_path_ << " is not initialized";
        return 0;
    }

    if (state_ == MemoryState::UNALLOCATED) {
        LOG(WARNING) << "Model " << model_path_ << " is not allocated";
        return 0;
    }

    if (state_ == MemoryState::LOADING) {
        return -1;
    }

    // make sure no gpu replicas are loading
    for (auto& [replica_uuid, gpu_replica] : gpu_replicas_) {
        if (gpu_replica->state_ == MemoryState::LOADING) {
            return -1;
        }
    }

    // free pinned memory
    int freed_chunks = pinned_mem_->num_chunks();
    pinned_mem_.reset();
    state_ = MemoryState::UNALLOCATED;

    return freed_chunks;
}

int Model::DispatchToGpu(
    const std::shared_ptr<GpuReplica>& gpu_replica,
    const std::unordered_map<int, MemCopyChunkList>& mem_copy_chunks,
    const std::unordered_map<int, MemCopyHandleList>& mem_copy_handles) {
    // device_id, chunk_offset, size, gpu_offset

    size_t num_chunks = pinned_mem_->num_chunks();
    std::vector<std::vector<GpuChunk>> chunk_id_to_gpu_chunks(num_chunks);
    for (const auto& [device_id, mem_copy_chunk_list] : mem_copy_chunks) {
        const auto& device_handles = mem_copy_handles.at(device_id);
        std::vector<size_t> handle_offsets(device_handles.size(), 0);

        for (auto [host_offset, size, gpu_offset, handle_idx] :
             mem_copy_chunk_list) {
            handle_offsets[handle_idx] = gpu_offset;

            std::vector<std::tuple<int, size_t, size_t>> chunks =
                MapDataToChunks(host_offset, size, pinned_mem_->chunk_size());
            for (const auto& [chunk_id, chunk_offset, size] : chunks) {
                chunk_id_to_gpu_chunks[chunk_id].push_back(
                    std::make_tuple(device_id, chunk_offset, size,
                                    handle_offsets[handle_idx], handle_idx));
                handle_offsets[handle_idx] += size;
            }
        }
    }

    for (int i = 0; i < host_ptr_vector_->capacity(); i++) {
        auto data_chunk = host_ptr_vector_->dequeue(i);
        auto chunk_id = data_chunk.chunk_id_;
        auto& gpu_chunks = chunk_id_to_gpu_chunks[chunk_id];
        for (const auto& [device_id, chunk_offset, size, gpu_offset, handle_idx] :
             gpu_chunks) {
            auto& gpu_loading_queue = gpu_replica->gpu_loading_queue_.at(device_id);
            LOG(INFO) << "Enqueueing chunk " << chunk_id << " offset " << chunk_offset
                      << " size " << size << " to device " << device_id;
            gpu_loading_queue->enqueue(
                GpuBatch{chunk_id, chunk_offset, size, gpu_offset, handle_idx});
        }
    }

    // notify end of loading
    for (auto& [device_id, gpu_loading_queue] : gpu_replica->gpu_loading_queue_) {
        gpu_loading_queue->enqueue(GpuBatch{});
    }

    return 0;
}

std::vector<std::tuple<int, size_t, size_t>> Model::MapDataToChunks(
    size_t offset, size_t size, size_t chunk_size) {
    int start_chunk = offset / chunk_size;
    size_t offset_in_start_chunk = offset % chunk_size;
    size_t remaining_data = size;
    std::vector<std::tuple<int, size_t, size_t>> output;

    for (int chunk_id = start_chunk; remaining_data > 0; ++chunk_id) {
        const size_t chunk_data_size =
            (chunk_id == start_chunk)
                ? std::min(chunk_size - offset_in_start_chunk, remaining_data)
                : std::min(chunk_size, remaining_data);
        output.emplace_back(chunk_id,
                            chunk_id == start_chunk ? offset_in_start_chunk : 0,
                            chunk_data_size);
        remaining_data -= chunk_data_size;
    }

    return output;
}

int Model::AllocatePinnedMemory(std::shared_ptr<PinnedMemoryPool> pool) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == MemoryState::UNINITIALIZED) {
        LOG(ERROR) << "Model " << model_path_ << " is not initialized";
        return -1;
    }
    if (state_ != MemoryState::UNALLOCATED) {
        return 0;
    }
    pinned_mem_ = std::make_shared<PinnedMemory>();
    int ret = pinned_mem_->Allocate(model_size_, pool);
    if (ret < 0) {
        LOG(ERROR) << "Error allocating CPU memory for model " << model_path_;
        return ret;
    } else if (ret > 0) {
        LOG(WARNING) << "Not enough memory for model " << model_path_;
        return ret;
    } else if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
        LOG(ERROR) << "CPU memory not allocated";
        return -1;
    }

    state_ = MemoryState::ALLOCATED;
    return 0;
}