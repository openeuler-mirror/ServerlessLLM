// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//  Licensed under the Apache License, Version 2.0
// ----------------------------------------------------------------------------
#pragma once

#ifdef USE_CANN
#include "acl/acl.h"
#include <string>
#include <unordered_map>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <cstdint> // For int32_t, uint64_t
#include <utility> // For std::pair

// Required for named pipes (FIFOs) - needed for declarations
// We only include <sys/stat.h> and <unistd.h> if we specifically declare functions that use them.
// For the declarations, these might not be strictly necessary if the types are fundamental.
// However, including them ensures errno and other system-level types are known if helper functions
// were inline or templated (which they are not here). For external functions, it's generally fine.
// <fstream> and <iostream> for usage within the helper function definitions.

// Define FIFO paths for inter-process communication
#define FIFO_PATH_A_TO_B "/tmp/cann_ipc_a_to_b" // Process A sends handle info to Process B
#define FIFO_PATH_B_TO_A "/tmp/cann_ipc_b_to_a" // Process B sends its PID to Process A


// Helper functions for inter-process communication (using named pipes)
// These are declarations ONLY. Implementations go into a .cpp file.

/**
 * @brief Sends an int32_t (e.g., a PID) through a named pipe.
 * @param value The int32_t value to send.
 * @param fifo_path The path to the named pipe.
 */
void send_int32(int32_t value, const std::string& fifo_path);

/**
 * @brief Receives an int32_t (e.g., a PID) from a named pipe.
 * @param fifo_path The path to the named pipe.
 * @return The received int32_t value, or -1 on error.
 */
int32_t receive_int32(const std::string& fifo_path);

/**
 * @brief Sends a uint64_t handle ID and size_t through a named pipe.
 * @param handle_id The uint64_t handle ID to send.
 * @param size The size_t value to send.
 * @param fifo_path The path to the named pipe.
 */
void send_handle_info(uint64_t handle_id, size_t size, const std::string& fifo_path);

/**
 * @brief Receives a uint64_t handle ID and size_t from a named pipe.
 * @param fifo_path The path to the named pipe.
 * @return A pair containing the received handle ID and size, or {0, 0} on error.
 */
std::pair<uint64_t, size_t> receive_handle_info(const std::string& fifo_path);


// Real CANN IPC Manager using native CANN APIs
// Uses aclrtMemExportToShareableHandle and aclrtMemImportFromShareableHandle
// Based on CANN documentation: https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/apiref/appdevgapi/aclcppdevg_03_0117.html
class CannIpcManager {
public:
    static CannIpcManager& getInstance() {
        static CannIpcManager instance;
        return instance;
    }

    struct MemoryInfo {
        void* ptr;                          // Original or imported device pointer
        size_t size;                        // Memory size
        int device_id;                      // Device ID
        aclrtDrvMemHandle physical_handle;  // Physical memory handle for CANN IPC
        void* ipc_device_ptr;               // Mapped IPC device pointer (can be removed if not used)
        uint64_t shareable_handle;          // CANN shareable handle for inter-process sharing
        void* vir_ptr;                      // Virtual address pointer for cleanup
    };

    // Create IPC handle - mimics cudaIpcGetMemHandle
    aclError createIpcHandle(void** device_ptr, size_t size, int device_id, uint64_t* handle_id, int32_t process_b_pid);

    // Open IPC handle - mimics cudaIpcOpenMemHandle
    aclError openIpcHandle(uint64_t handle_id, int device_id, void** device_ptr);

    // Close IPC handle - mimics cudaIpcCloseMemHandle
    aclError closeIpcHandle(void* device_ptr);

    // Convert handle to/from string for compatibility (enhanced with size info)
    std::string handleToString(uint64_t handle_id);
    uint64_t stringToHandle(const std::string& handle_str);

private:
    CannIpcManager() : next_handle_id_(1) {}

    std::mutex mutex_;
    uint64_t next_handle_id_;
    std::unordered_map<uint64_t, MemoryInfo> handle_to_memory_;
    std::unordered_map<void*, uint64_t> ptr_to_handle_;
    std::unordered_map<uint64_t, size_t> pending_imports_; // Store size info for imports
};

// Convenience functions that mimic CUDA IPC API exactly
aclError cannIpcGetMemHandle(std::string* handle_str, void** device_ptr, size_t size, int device_id, int32_t target_process_id);
aclError cannIpcOpenMemHandle(void** device_ptr, const std::string& handle_str, int device_id);
aclError cannIpcCloseMemHandle(void* device_ptr);

#endif // USE_CANN
