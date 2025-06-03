/* Copyright 2024 The JAX Authors.

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

#include "kernels.h"
#include "kernel_helpers.h"

namespace cuda_examples {

namespace {

__global__ void FooFwdKernel(const float *a, const float *b, float *c,
                             float *b_plus_1,  // intermediate output b+1
                             size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride) {
    b_plus_1[i] = b[i] + 1.0f;
    c[i] = a[i] * b_plus_1[i];
  }
}


__global__ void FooBwdKernel(const float *c_grad,    // incoming gradient wrt c
                             const float *a,         // original input a
                             const float *b_plus_1,  // intermediate output b+1
                             float *a_grad,          // outgoing gradient wrt a
                             float *b_grad,          // outgoing gradient wrt b
                             size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride) {
    // In practice on GPUs b_plus_1 can be recomputed for practically free
    // instead of storing it out and reusing, so the reuse here is a bit
    // contrived. We do it to demonstrate residual/intermediate output passing
    // between the forward and the backward pass which becomes useful when
    // recomputation is more expensive than reuse.
    a_grad[i] = c_grad[i] * b_plus_1[i];
    b_grad[i] = c_grad[i] * a[i];
  }
}


}  // namespace


void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}


void gpu_foo_fwd(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {
  const FooFwdDescriptor &d = *UnpackDescriptor<FooFwdDescriptor>(opaque, opaque_len);

  const int block_dim = 128;
  const int grid_dim = 1;

  const float *a = reinterpret_cast<const float *>(buffers[0]);
  const float *b = reinterpret_cast<const float *>(buffers[1]);
  float *c = reinterpret_cast<float *>(buffers[2]);
  float *b_plus_1 = reinterpret_cast<float *>(buffers[3]);

  FooFwdKernel<<<grid_dim, block_dim, 0, stream>>>(a, b, c, b_plus_1, d.n);
  ThrowIfError(cudaGetLastError());
}



void gpu_foo_bwd(cudaStream_t stream, void **buffers, const char *opaque,
                          std::size_t opaque_len) {
  const FooBwdDescriptor &d = *UnpackDescriptor<FooBwdDescriptor>(opaque, opaque_len);

  const int block_dim = 128;
  const int grid_dim = 1;

  const float *c_grad = reinterpret_cast<const float *>(buffers[0]);
  const float *a = reinterpret_cast<const float *>(buffers[1]);
  const float *b_plus_1 = reinterpret_cast<const float *>(buffers[2]);
  float *a_grad = reinterpret_cast<float *>(buffers[3]);
  float *b_grad = reinterpret_cast<float *>(buffers[4]);

  FooBwdKernel<<<grid_dim, block_dim, 0, stream>>>(c_grad, a, b_plus_1, a_grad, b_grad, d.n);
  ThrowIfError(cudaGetLastError());
}

}  // namespace cuda_examples