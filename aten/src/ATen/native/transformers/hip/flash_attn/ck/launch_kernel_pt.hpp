// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/host/kernel_launch.hpp>
#include <c10/macros/Macros.h>

namespace ck_tile {
template <typename KernelImpl>
struct guarded_kernel_pt {
    static constexpr int kBlockSize      = KernelImpl::kBlockSize;
    static constexpr index_t kBlockPerCu = KernelImpl::kBlockPerCu;

    template <typename... Args>
    CK_TILE_DEVICE void operator()(Args... args) const
    {
#if (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__))
        KernelImpl{}(args...);
#else
        CUDA_KERNEL_ASSERT(false && "Fatal! Attempting to call a CK SDPA kernel on unsupported hardware");
#endif
    }
};

// PyTorch specific wrapper that keeps CK's kernel selection logic but adds the
// runtime guard above so we fail loudly on unsupported hardware.
template <int MinBlockPerCu = CK_TILE_MIN_BLOCK_PER_CU,
          typename Arch      = void,
          typename KernelImpl,
          typename... Args>
CK_TILE_HOST auto
make_kernel_pt(KernelImpl /*f*/, dim3 grid_dim, dim3 block_dim, std::size_t lds_byte, Args... args)
{
    using GuardedKernel = guarded_kernel_pt<KernelImpl>;
    return make_kernel<MinBlockPerCu, Arch>(GuardedKernel{}, grid_dim, block_dim, lds_byte, args...);
}
} // namespace ck_tile
