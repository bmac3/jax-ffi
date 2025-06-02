#ifndef _CUDA_EXAMPLES_KERNELS_H_
#define _CUDA_EXAMPLES_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace cuda_examples {
struct FooFwdDescriptor {
  size_t n;
};

struct FooBwdDescriptor {
  size_t n;
};

void gpu_foo_fwd(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_foo_bwd(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

}  // namespace cuda_examples

#endif