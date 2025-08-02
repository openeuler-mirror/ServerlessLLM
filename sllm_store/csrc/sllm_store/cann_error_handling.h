// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
// ----------------------------------------------------------------------------
#pragma once

#ifdef USE_CANN
#include "acl/acl.h"
#endif
#include <errno.h>
#include <glog/logging.h>
#include <string.h>

#ifdef USE_CANN
// Macro to check CANN API call results. If an error occurs, it logs the error and returns -1.
#define CANN_CHECK(x, msg) { aclError ret = (x); if (ret != ACL_ERROR_NONE) { LOG(ERROR) << msg << " CANN error: " << ret << std::endl; return -1; } }
#else
// Placeholder macro when CANN is not enabled. It logs a message indicating CANN is unavailable.
#define CANN_CHECK(x, msg) { LOG(ERROR) << "CANN not available: " << msg << std::endl; return -1; }
#endif

// Macro to check POSIX system call results. If an error occurs (return value < 0),
// it logs the error message, errno, and strerror, then returns -1.
#define CHECK_POSIX(x, msg) { if ((x) < 0) { LOG(ERROR) << msg << " errno: " << errno << "msg: " << strerror(errno); return -1; } }

// Macro to wait for a vector of futures. It iterates through each future,
// gets its result, and if any result is non-zero, it logs an error and returns that result.
#define WAIT_FUTURES(futures, msg) { for (auto& future : futures) { int ret = future.get(); if (ret != 0) { LOG(ERROR) << msg; return ret; } } }

// Macro to check a return value. If the value is non-zero, it logs an error and returns -1.
#define CHECK_RETURN(x, msg) { if ((x) != 0) { LOG(ERROR) << msg; return -1; } }
