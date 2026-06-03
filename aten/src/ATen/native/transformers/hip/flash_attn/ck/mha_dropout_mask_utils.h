#pragma once

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>

#include <limits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace pytorch_flash {

inline uint8_t ck_dropout_keep_threshold(float p_dropout) {
    return static_cast<uint8_t>(
        (1.0f - p_dropout) * static_cast<float>(std::numeric_limits<uint8_t>::max()));
}

template <typename scalar_t>
__global__ void ck_encode_dropout_randval_kernel(
    const uint8_t* randval,
    scalar_t* encoded,
    int64_t numel,
    uint8_t keep_threshold) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        encoded[idx] = randval[idx] <= keep_threshold ? scalar_t(0.5f) : scalar_t(-0.5f);
    }
}

inline at::Tensor ck_encode_dropout_randval(
    const at::Tensor& randval,
    float p_dropout,
    const at::TensorOptions& opts) {
    auto encoded = at::empty(randval.sizes(), opts);
    const auto numel = randval.numel();
    if (numel == 0) {
        return encoded;
    }

    constexpr int threads = 256;
    const dim3 blocks((numel + threads - 1) / threads);
    const auto keep_threshold = ck_dropout_keep_threshold(p_dropout);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        encoded.scalar_type(),
        "ck_encode_dropout_randval",
        [&] {
            ck_encode_dropout_randval_kernel<scalar_t>
                <<<blocks, threads, 0, stream>>>(
                    randval.data_ptr<uint8_t>(),
                    encoded.data_ptr<scalar_t>(),
                    numel,
                    keep_threshold);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return encoded;
}

} // namespace pytorch_flash
