// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actually implementation of the
// custom call can be found in kernels.cc.cu.

#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace cuda_examples;

namespace {

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["gpu_foo_fwd"] = EncapsulateFunction(gpu_foo_fwd);
  dict["gpu_foo_bwd"] = EncapsulateFunction(gpu_foo_bwd);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("registrations", &Registrations);
  m.def("build_foo_fwd_descriptor",
        [](size_t n) { return PackDescriptor(FooFwdDescriptor{n}); });
  m.def("build_foo_bwd_descriptor",
        [](size_t n) { return PackDescriptor(FooBwdDescriptor{n}); });
}

}  // namespace
