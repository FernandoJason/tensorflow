/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_CACHE_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_CACHE_H_

#include <utility>
#include <vector>

#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/tsl/util/tied_ref.h"

namespace xla::gpu {

// Owns multicast and symmetrical memories used by executable.
// This cache is needed to prevent destruction of the multicast objects and
// unregistering the symmetrical memories before the executable is done using
// them.
class CollectiveMemoryCache {
 public:
  void AddMulticastMemory(
      tsl::TiedRef<stream_executor::gpu::MulticastMemory> multicast_memory) {
    CleanupExpired();
    multicast_memories_.push_back(std::move(multicast_memory));
  }

 private:
  void CleanupExpired() {
    std::erase_if(
        multicast_memories_,
        [](tsl::TiedRef<stream_executor::gpu::MulticastMemory>&
               multicast_memory) { return multicast_memory.Expired(); });
  }

  // Prevents destruction of the collective memory on a given device until the
  // clique is destroyed.
  std::vector<tsl::TiedRef<stream_executor::gpu::MulticastMemory>>
      multicast_memories_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_CACHE_H_
