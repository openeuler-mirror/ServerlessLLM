// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//  Licensed under the Apache License, Version 2.0
// ----------------------------------------------------------------------------
#ifdef USE_CANN
#include "cann_ipc.h" // Includes the declarations of helper functions and CannIpcManager

#include <errno.h>    // For errno
#include <fcntl.h>    // For open, O_RDWR, O_CREAT etc.
#include <string.h>
#include <fstream>    // For std::ifstream, std::ofstream
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// System includes for FIFO operations - MUST be in the .cpp where functions are defined
#include <sys/stat.h> // For mkfifo
#include <unistd.h>   // For unlink (optional, for cleanup), though not used in helper fn, often used with FIFOs

// * FIFO mechanism is only used for testing purpose, in actual development the pid is using proto to transfer
// Helper function to open a FIFO robustly
int open_fifo_fd(const std::string& fifo_path, int flags)
{
  // Create the FIFO if it doesn't exist. Permissions 0666 allow read/write for all.
  if (mkfifo(fifo_path.c_str(), 0666) == -1 && errno != EEXIST)
  {
    std::cerr << "[ERROR][IPC_UTIL] Failed to create FIFO: " << fifo_path
              << ", errno: " << errno << " : " << strerror(errno) << std::endl;
    return -1;
  }

  // We open as file descriptor first to handle blocking correctly.
  // A simple std::fstream will block.
  int fd = open(fifo_path.c_str(), flags, 0666);
  if (fd == -1)
  {
    std::cerr << "[ERROR][IPC_UTIL] Failed to open FIFO: " << fifo_path
              << ", errno: " << errno << " : " << strerror(errno) << std::endl;
    // Consider retrying for transient errors like EINTR or EAGAIN (if O_NONBLOCK were used)
    return -1;
  }
  return fd;
}

// Helper functions for inter-process communication (using named pipes)
// These are the definitions.

void send_int32(int32_t value, const std::string& fifo_path)
{
  int fd = open_fifo_fd(fifo_path, O_WRONLY | O_SYNC); // Open for writing, with sync
  if (fd == -1)
  {
    return;
  }

  if (write(fd, &value, sizeof(value)) == -1)
  {
    std::cerr << "[ERROR][IPC_UTIL] Failed to write int32_t to FIFO: " << fifo_path
              << ", errno: " << errno << " : " << strerror(errno) << std::endl;
  }
  close(fd);
  std::cout << "[INFO][IPC_UTIL] Sent int32_t value " << value << " to "
            << fifo_path << std::endl;
}

int32_t receive_int32(const std::string& fifo_path)
{
  int fd = open_fifo_fd(fifo_path, O_RDONLY | O_SYNC); // Open for reading, with sync
  if (fd == -1)
  {
    return -1;
  }

  int32_t received_value = -1;
  ssize_t bytes_read = read(fd, &received_value, sizeof(received_value));
  if (bytes_read == -1)
  {
    std::cerr << "[ERROR][IPC_UTIL] Failed to read from FIFO: " << fifo_path
              << ", errno: " << errno << " : " << strerror(errno) << std::endl;
    received_value = -1;
  }
  else if (bytes_read == 0)
  {
    std::cerr << "[WARNING][IPC_UTIL] Reached EOF on FIFO: " << fifo_path
              << " (no data or writer closed)" << std::endl;
    received_value = -1; // Or handle as specific empty state
  }
  close(fd);
  std::cout << "[INFO][IPC_UTIL] Received int32_t value " << received_value
            << " from " << fifo_path << std::endl;
  return received_value;
}

void send_handle_info(uint64_t handle_id, size_t size,
                      const std::string& fifo_path)
{
  int fd = open_fifo_fd(fifo_path, O_WRONLY | O_SYNC);
  if (fd == -1)
  {
    return;
  }

  if (write(fd, &handle_id, sizeof(handle_id)) == -1 ||
      write(fd, &size, sizeof(size)) == -1)
  {
    std::cerr << "[ERROR][IPC_UTIL] Failed to write handle info to FIFO: "
              << fifo_path << ", errno: " << errno << " : " << strerror(errno)
              << std::endl;
  }
  close(fd);
  std::cout << "[INFO][IPC_UTIL] Sent handle_id " << std::hex << handle_id
            << ", size " << std::dec << size << " to " << fifo_path
            << std::endl;
}

std::pair<uint64_t, size_t> receive_handle_info(
    const std::string& fifo_path)
{
  int fd = open_fifo_fd(fifo_path, O_RDONLY | O_SYNC);
  if (fd == -1)
  {
    return {0, 0};
  }

  uint64_t received_handle_id = 0;
  size_t received_size = 0;
  ssize_t bytes_read_handle =
      read(fd, &received_handle_id, sizeof(received_handle_id));
  ssize_t bytes_read_size = read(fd, &received_size, sizeof(received_size));

  if (bytes_read_handle == -1 || bytes_read_size == -1)
  {
    std::cerr << "[ERROR][IPC_UTIL] Failed to read handle info from FIFO: "
              << fifo_path << ", errno: " << errno << " : " << strerror(errno)
              << std::endl;
    received_handle_id = 0;
    received_size = 0;
  }
  else if (bytes_read_handle == 0 || bytes_read_size == 0)
  {
    std::cerr << "[WARNING][IPC_UTIL] Reached EOF on FIFO for handle info: "
              << fifo_path << " (incomplete data or writer closed)"
              << std::endl;
    received_handle_id = 0;
    received_size = 0;
  }
  close(fd);
  std::cout << "[INFO][IPC_UTIL] Received handle_id " << std::hex
            << received_handle_id << ", size " << std::dec << received_size
            << " from " << fifo_path << std::endl;
  return {received_handle_id, received_size};
}

// Real CANN IPC implementation using native CANN APIs

aclError CannIpcManager::createIpcHandle(void** device_ptr, size_t size,
                                         int device_id, uint64_t* handle_id,
                                         int32_t process_b_pid)
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (device_ptr == nullptr || *device_ptr == nullptr)
  {
    std::cerr << "[ERROR] Device pointer is null, cannot create IPC handle"
              << std::endl;
    return ACL_ERROR_INVALID_PARAM;
  }

  void* original_ptr = *device_ptr; // Store the original pointer

  // Set device context
  aclError ret = aclrtSetDevice(device_id);
  if (ret != ACL_ERROR_NONE)
  {
    std::cerr << "[ERROR] Failed to set device " << device_id << ": " << ret
              << std::endl;
    return ret;
  }

  // Align size to 2 MB (huge page size)
  const size_t granularity = 2 * 1024 * 1024; // 2 MB
  size_t aligned_size = ((size + granularity - 1) / granularity) * granularity;

  // First reserve virtual address space
  void* vir_ptr = nullptr;
  ret = aclrtReserveMemAddress(&vir_ptr, aligned_size, 0, nullptr, 0);
  if (ret != ACL_ERROR_NONE)
  {
    std::cerr << "[ERROR] Failed to reserve memory address: " << ret
              << std::endl;
    return ret;
  }

  // Configure physical memory properties properly
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE; // Use 2MB pages
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<unsigned>(device_id);
  prop.reserve = 0;

  // Allocate physical memory for IPC with proper properties
  aclrtDrvMemHandle physical_handle;
  ret = aclrtMallocPhysical(&physical_handle, aligned_size, &prop, 0);
  if (ret != ACL_ERROR_NONE)
  {
    std::cerr << "[ERROR] Failed to allocate physical memory for IPC: " << ret
              << std::endl;
    aclrtReleaseMemAddress(vir_ptr);
    return ret;
  }

  // Map the physical memory to the reserved virtual address
  ret = aclrtMapMem(vir_ptr, aligned_size, 0, physical_handle, 0);
  if (ret != ACL_ERROR_NONE)
  {
    std::cerr << "[ERROR] Failed to map physical memory: " << ret << std::endl;
    aclrtFreePhysical(physical_handle);
    aclrtReleaseMemAddress(vir_ptr);
    return ret;
  }

  // Export physical memory handle for sharing
  uint64_t shareable_handle;
  ret = aclrtMemExportToShareableHandle(physical_handle, ACL_MEM_HANDLE_TYPE_NONE,
                                        0, &shareable_handle);
  if (ret != ACL_ERROR_NONE)
  {
    std::cerr << "[ERROR] Failed to export shareable handle: " << ret
              << std::endl;
    aclrtUnmapMem(vir_ptr);
    aclrtFreePhysical(physical_handle);
    aclrtReleaseMemAddress(vir_ptr);
    return ret;
  }

  // --- IPC LOGIC FOR PID EXCHANGE (Process A - Sender of Handle, Receiver of PID) ---

  // // 1. Send shareable_handle and its aligned_size to Process B
  // std::cout << "[INFO] Process A sending handle info to Process B..." << std::endl;
  // send_handle_info(shareable_handle, aligned_size, FIFO_PATH_A_TO_B);

  // // 2. Wait to receive Process B's PID
  // std::cout << "[INFO] Process A waiting for Process B's PID on " << FIFO_PATH_B_TO_A << "..." << std::endl;
  // int32_t process_b_pid = receive_int32(FIFO_PATH_B_TO_A);

  if (process_b_pid == -1)
  {
    std::cerr << "[ERROR] Failed to receive Process B's PID. Aborting IPC setup."
              << std::endl;
    // Perform cleanup for the created resources before returning
    aclrtUnmapMem(vir_ptr);
    aclrtFreePhysical(physical_handle);
    aclrtReleaseMemAddress(vir_ptr);
    return ACL_ERROR_INVALID_PARAM; // Or a more appropriate error code
  }

  std::cout << "[INFO] Process A received Process B's PID: " << process_b_pid
            << std::endl;

  // 3. Allow Process B (identified by process_b_pid) to import the handle
  // aclrtMemSetPidToShareableHandle expects an array of PIDs.
  int32_t pids_to_whitelist[] = {process_b_pid};
  ret = aclrtMemSetPidToShareableHandle(shareable_handle, pids_to_whitelist, 1);
  if (ret != ACL_ERROR_NONE)
  {
    std::cerr << "[ERROR] Failed to set PID whitelist for shareable handle: "
              << ret << std::endl;
    // Cleanup on failure
    aclrtUnmapMem(vir_ptr);
    aclrtFreePhysical(physical_handle);
    aclrtReleaseMemAddress(vir_ptr);
    return ret;
  }

  std::cout << "[INFO] Successfully set pid to whitelist: " << process_b_pid
            << std::endl;

  // Use shareable handle as the handle ID
  *handle_id = shareable_handle;

  // Store memory info for cleanup (store the mapped pointer, not original)
  MemoryInfo mem_info;
  mem_info.ptr = vir_ptr; // Store original device_ptr for lookup
  mem_info.size = aligned_size;
  mem_info.device_id = device_id;
  mem_info.physical_handle = physical_handle;
  mem_info.shareable_handle = shareable_handle;
  mem_info.vir_ptr = vir_ptr;     // Store virtual address for release
  mem_info.ipc_device_ptr = vir_ptr; // Store the mapped IPC pointer (vir_ptr)

  handle_to_memory_[*handle_id] = mem_info;
  // ptr_to_handle_[original_ptr] = *handle_id; // Keep original mapping for lookup
  ptr_to_handle_[vir_ptr] = *handle_id;       // Also map the shared VA for cleanup

  // Return the shared virtual address to the caller
  *device_ptr = vir_ptr;

  std::cout << "[INFO] Created CANN IPC handle " << std::hex << *handle_id
            << " for device ptr " << original_ptr << " -> shared VA " << vir_ptr << " on device " << device_id
            << ", size: " << aligned_size << std::endl;

  return ACL_ERROR_NONE;
}

aclError CannIpcManager::openIpcHandle(uint64_t handle_id, int device_id,
                                       void** device_ptr)
{
  std::lock_guard<std::mutex> lock(mutex_);

  std::cout << "[INFO] Opening CANN IPC handle " << std::hex << handle_id
            << " on device " << device_id << std::endl;

  // Get size from pending imports (this map is populated by stringToHandle when Process B receives the handle string)
  auto pending_it = pending_imports_.find(handle_id);
  if (pending_it == pending_imports_.end())
  {
    std::cerr << "[ERROR] Size information not found for handle " << std::hex
              << handle_id << ". Make sure stringToHandle was called."
              << std::endl;
    return ACL_ERROR_INVALID_PARAM;
  }
  size_t aligned_size = pending_it->second; // Use aligned_size as stored by stringToHandle
  pending_imports_.erase(pending_it);       // Clear after use

  // Set device context
  aclError ret = aclrtSetDevice(device_id);
  if (ret != ACL_ERROR_NONE)
  {
    std::cerr << "[ERROR] Failed to set device " << device_id << ": " << ret
              << std::endl;
    return ret;
  }

  // Reserve virtual address space first
  void* vir_ptr = nullptr;
  ret = aclrtReserveMemAddress(&vir_ptr, aligned_size, 0, nullptr, 0);
  if (ret != ACL_ERROR_NONE)
  {
    std::cerr << "[ERROR] Failed to reserve memory address: " << ret
              << std::endl;
    return ret;
  }

  // // Get current process id (Process B's PID)
  // int32_t pid_value;
  // ret = aclrtDeviceGetBareTgid(&pid_value);
  // if (ret != ACL_ERROR_NONE) {
  //   std::cerr << "[ERROR] Failed to get current process id: " << ret << std::endl;
  //   aclrtReleaseMemAddress(vir_ptr);
  //   return ret;
  // }

  // --- IPC LOGIC FOR PID EXCHANGE (Process B - Receiver of Handle, Sender of PID) ---

  // ! do this in grpc server, beforehand
  // // Send Process B's PID back to Process A
  // std::cout << "[INFO] Process B sending its PID (" << pid_value << ") to Process A on " << FIFO_PATH_B_TO_A << "..." << std::endl;
  // send_int32(pid_value, FIFO_PATH_B_TO_A);
  // Process A should now have received this PID and called aclrtMemSetPidToShareableHandle.

  // Import the shareable handle
  aclrtDrvMemHandle imported_handle;
  ret = aclrtMemImportFromShareableHandle(handle_id, device_id, &imported_handle);
  if (ret != ACL_ERROR_NONE)
  {
    std::cerr << "[ERROR] Failed to import shareable handle " << std::hex
              << handle_id << ": " << ret << std::endl;
    aclrtReleaseMemAddress(vir_ptr);
    return ret;
  }

  // Map the imported physical memory to reserved virtual address
  ret = aclrtMapMem(vir_ptr, aligned_size, 0, imported_handle, 0);
  if (ret != ACL_ERROR_NONE)
  {
    std::cerr << "[ERROR] Failed to map imported memory: " << ret << std::endl;
    aclrtFreePhysical(imported_handle);
    aclrtReleaseMemAddress(vir_ptr);
    return ret;
  }

  // MOD: Sync after map to ensure data visibility on client
  ret = aclrtSynchronizeDevice();
  if (ret != ACL_ERROR_NONE) {
    std::cerr << "[ERROR] Failed to synchronize device after IPC map: " << ret << std::endl;
    // Cleanup
    aclrtUnmapMem(vir_ptr);
    aclrtFreePhysical(imported_handle);
    aclrtReleaseMemAddress(vir_ptr);
    return ret;
  }

  // Return the mapped device pointer
  *device_ptr = vir_ptr;

  // Store the mapping for cleanup
  MemoryInfo mem_info;
  mem_info.ptr = vir_ptr; // Store the mapped IPC pointer
  mem_info.size = aligned_size; // Store the aligned size for consistency
  mem_info.device_id = device_id;
  mem_info.physical_handle = imported_handle;
  mem_info.shareable_handle = handle_id;
  mem_info.vir_ptr = vir_ptr;
  mem_info.ipc_device_ptr = vir_ptr; // This is the actual mapped pointer

  handle_to_memory_[handle_id] = mem_info;
  ptr_to_handle_[vir_ptr] = handle_id; // Map the *imported* pointer to its handle

  std::cout << "[INFO] Successfully opened CANN IPC handle " << std::hex
            << handle_id << " -> device ptr " << *device_ptr << " on device "
            << device_id << std::endl;

  return ACL_ERROR_NONE;
}

aclError CannIpcManager::closeIpcHandle(void* device_ptr)
{
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = ptr_to_handle_.find(device_ptr);
  if (it == ptr_to_handle_.end())
  {
    std::cerr << "[WARNING] Device pointer " << device_ptr
              << " not found in IPC registry" << std::endl;
    return ACL_ERROR_INVALID_PARAM;
  }

  uint64_t handle_id = it->second;
  auto mem_it = handle_to_memory_.find(handle_id);
  if (mem_it == handle_to_memory_.end())
  {
    std::cerr << "[WARNING] Handle " << std::hex << handle_id
              << " not found in memory map" << std::endl;
    return ACL_ERROR_INVALID_PARAM;
  }

  const MemoryInfo& mem_info = mem_it->second;

  // Unmap the memory first
  if (mem_info.vir_ptr != nullptr)
  { // Use vir_ptr for unmapping
    aclrtUnmapMem(mem_info.vir_ptr);
  }

  // Free the physical memory handle
  if (mem_info.physical_handle)
  {
    aclrtFreePhysical(mem_info.physical_handle);
  }

  // Release the virtual address space
  if (mem_info.vir_ptr != nullptr)
  {
    aclrtReleaseMemAddress(mem_info.vir_ptr);
  }

  // std::cout << "[INFO] Closed CANN IPC handle " << std::hex << handle_id
  //           << " for device ptr " << device_ptr << std::endl;

  handle_to_memory_.erase(handle_id);
  ptr_to_handle_.erase(it);

  return ACL_ERROR_NONE;
}

std::string CannIpcManager::handleToString(uint64_t handle_id)
{
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = handle_to_memory_.find(handle_id);
  if (it == handle_to_memory_.end())
  {
    std::cerr << "[ERROR] Invalid handle ID: " << std::hex << handle_id
              << std::endl;
    return "";
  }

  const MemoryInfo& mem_info = it->second;
  std::ostringstream oss;
  // Store the shareable handle and the *aligned* size for consistent import
  oss << std::hex << mem_info.shareable_handle << ":" << std::dec
      << mem_info.size;
  return oss.str();
}

uint64_t CannIpcManager::stringToHandle(const std::string& handle_str)
{
  size_t colon_pos = handle_str.find(':');
  if (colon_pos == std::string::npos)
  {
    std::cerr << "[ERROR] Invalid handle format - missing size: " << handle_str
              << std::endl;
    return 0;
  }

  try
  {
    // Parse shareable handle (hex)
    uint64_t shareable_handle =
        std::stoull(handle_str.substr(0, colon_pos), nullptr, 16);

    // Parse buffer size (decimal) - this should be the ALIGNED size from the sender
    size_t size = std::stoul(handle_str.substr(colon_pos + 1), nullptr, 10);

    // Store size for import operation
    // This is crucial for openIpcHandle to know the correct aligned_size for aclrtReserveMemAddress
    pending_imports_[shareable_handle] = size;

    return shareable_handle;
  }
  catch (const std::exception& e)
  {
    std::cerr << "[ERROR] Failed to parse handle string '" << handle_str
              << "': " << e.what() << std::endl;
    return 0;
  }
}

// Convenience functions that mimic CUDA IPC API exactly
aclError cannIpcGetMemHandle(std::string* handle_str, void** device_ptr,
                             size_t size, int device_id,
                             int32_t target_process_id)
{
  CannIpcManager& manager = CannIpcManager::getInstance();
  uint64_t handle_id;

  aclError ret = manager.createIpcHandle(device_ptr, size, device_id,
                                         &handle_id, target_process_id);
  if (ret != ACL_ERROR_NONE)
  {
    return ret;
  }

  *handle_str = manager.handleToString(handle_id);
  return ACL_ERROR_NONE;
}

aclError cannIpcOpenMemHandle(void** device_ptr, const std::string& handle_str,
                              int device_id)
{
  CannIpcManager& manager = CannIpcManager::getInstance();
  uint64_t handle_id = manager.stringToHandle(handle_str);

  if (handle_id == 0)
  {
    std::cerr << "[ERROR] Invalid handle string: " << handle_str << std::endl;
    return ACL_ERROR_INVALID_PARAM;
  }

  return manager.openIpcHandle(handle_id, device_id, device_ptr);
}

aclError cannIpcCloseMemHandle(void* device_ptr)
{
  CannIpcManager& manager = CannIpcManager::getInstance();
  return manager.closeIpcHandle(device_ptr);
}

#endif // USE_CANN