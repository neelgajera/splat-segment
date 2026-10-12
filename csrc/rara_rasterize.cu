// RaRa Clipper rasterizer for (Beta / Gaussian) splats.
//
// Implements the clipping of Li et al., "RaRa Clipper: A Clipper for Gaussian Splatting Based on Ray Tracer
// and Rasterizer" (arXiv:2506.20202): primitives cut by the clipping plane are still rasterized, but each
// pixel's alpha is scaled by the fraction of the camera ray's chord through the primitive that lies on the
// visible side of the plane (Eqs. 11-12). Fully visible primitives render unchanged and fully clipped ones are
// dropped on the Python side, so the only extra work is a ray/ellipsoid test for the few primitives the
// plane actually cuts.
//
// The tile loop is adapted from gsplat's rasterize_to_pixels_fwd kernel (Apache-2.0, Copyright the gsplat
// authors; Beta-kernel variant from Deformable Beta Splatting, Apache-2.0). Forward pass only: this renderer
// is for viewing and evaluating a trained model, not for training.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

namespace cg = cooperative_groups;

struct RaRaParams {
    // clipping plane n.x + d = 0; the half-space n.x + d > 0 is kept
    float nx, ny, nz, d;
    // camera: origin, camera-to-world rotation (row-major), pinhole intrinsics
    float ex, ey, ez;
    float r00, r01, r02, r10, r11, r12, r20, r21, r22;
    float fx, fy, cx, cy;
};

// Fraction of the ray's chord through the unit-support ellipsoid that lies on the visible side of the plane.
// minv maps world offsets into the primitive's unit-sphere frame (M^-1 with M = R S).
__device__ inline float rara_weight(const float *minv, const float *mu, const RaRaParams &p,
                                    float dx, float dy, float dz) {
    const float ox = p.ex - mu[0], oy = p.ey - mu[1], oz = p.ez - mu[2];
    // o' = M^-1 (e - mu), v' = M^-1 d
    const float o0 = minv[0] * ox + minv[1] * oy + minv[2] * oz;
    const float o1 = minv[3] * ox + minv[4] * oy + minv[5] * oz;
    const float o2 = minv[6] * ox + minv[7] * oy + minv[8] * oz;
    const float v0 = minv[0] * dx + minv[1] * dy + minv[2] * dz;
    const float v1 = minv[3] * dx + minv[4] * dy + minv[5] * dz;
    const float v2 = minv[6] * dx + minv[7] * dy + minv[8] * dz;
    const float A = v0 * v0 + v1 * v1 + v2 * v2;
    const float B = 2.f * (o0 * v0 + o1 * v1 + o2 * v2);
    const float C = o0 * o0 + o1 * o1 + o2 * o2 - 1.f;
    const float disc = B * B - 4.f * A * C;
    const float s0 = p.nx * p.ex + p.ny * p.ey + p.nz * p.ez + p.d;   // plane value at the camera
    const float s1 = p.nx * dx + p.ny * dy + p.nz * dz;                // its rate along the ray
    if (disc <= 0.f || A <= 0.f) {
        // The ray misses the support ellipsoid but the pixel is still inside the projected footprint (the 2D
        // footprint is a dilated, affine approximation). The paper uses w = 1 here; with compact Beta kernels
        // that lights up the rims of mostly-clipped primitives as speckles. Instead, decide by the side of the
        // plane at the ray's closest approach to the primitive.
        const float tc = A > 0.f ? -B / (2.f * A) : 0.f;
        return (s0 + s1 * tc) > 0.f ? 1.f : 0.f;
    }
    const float sq = sqrtf(disc);
    const float t1 = (-B - sq) / (2.f * A);
    const float t2 = (-B + sq) / (2.f * A);
    float vis;
    if (fabsf(s1) < 1e-12f) {
        vis = s0 > 0.f ? (t2 - t1) : 0.f;
    } else {
        const float tp = -s0 / s1;
        vis = s1 > 0.f ? fmaxf(0.f, t2 - fmaxf(t1, tp))    // visible for t > tp
                       : fmaxf(0.f, fminf(t2, tp) - t1);   // visible for t < tp
    }
    return fminf(1.f, vis / (t2 - t1));
}

template <uint32_t CDIM>
__global__ void rara_rasterize_fwd_kernel(
    const uint32_t n_isects,
    const float2 *__restrict__ means2d,   // [N, 2]
    const float3 *__restrict__ conics,    // [N, 3]
    const float *__restrict__ colors,     // [N, CDIM]
    const float *__restrict__ opacities,  // [N]
    const float *__restrict__ betas,      // [N]
    const float *__restrict__ background, // [CDIM] or null
    const uint8_t *__restrict__ cut,      // [N]  1 = primitive intersects the clipping plane
    const float *__restrict__ minv,       // [N, 9]
    const float *__restrict__ mu,         // [N, 3]
    const RaRaParams params,
    const uint32_t image_width, const uint32_t image_height,
    const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets,  // [tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,   // [n_isects]
    float *__restrict__ render_colors,         // [H, W, CDIM]
    float *__restrict__ render_alphas          // [H, W]
) {
    auto block = cg::this_thread_block();
    const int32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    const float px = (float)j + 0.5f, py = (float)i + 0.5f;
    const int32_t pix_id = i * image_width + j;
    const bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // camera ray for this pixel (only needed for primitives the plane cuts)
    const float cdx = (px - params.cx) / params.fx, cdy = (py - params.cy) / params.fy;
    const float dx = params.r00 * cdx + params.r01 * cdy + params.r02;
    const float dy = params.r10 * cdx + params.r11 * cdy + params.r12;
    const float dz = params.r20 * cdx + params.r21 * cdy + params.r22;

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end = (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    float4 *xy_opacity_batch = reinterpret_cast<float4 *>(&id_batch[block_size]);
    float3 *conic_batch = reinterpret_cast<float3 *>(&xy_opacity_batch[block_size]);

    float T = 1.f;
    float pix_out[CDIM] = {0.f};
    const uint32_t tr = block.thread_rank();

    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) {
            break;
        }
        const uint32_t batch_start = range_start + block_size * b;
        const uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            xy_opacity_batch[tr] = make_float4(xy.x, xy.y, opacities[g], betas[g]);
            conic_batch[tr] = conics[g];
        }
        block.sync();

        const uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float4 xy_opac = xy_opacity_batch[t];
            const float ddx = xy_opac.x - px, ddy = xy_opac.y - py;
            const float sigma = conic.x * ddx * ddx + conic.z * ddy * ddy + 2.f * conic.y * ddx * ddy;
            if (sigma < 0.f || sigma >= 1.f) {
                continue;
            }
            float alpha = fminf(0.999f, xy_opac.z * __powf(1.f - sigma, xy_opac.w));
            const int32_t g = id_batch[t];
            if (cut[g]) {
                alpha *= rara_weight(minv + 9 * g, mu + 3 * g, params, dx, dy, dz);   // Eq. 12
                if (alpha < 1.f / 255.f) {
                    continue;
                }
            }
            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }
            const float vis = alpha * T;
            const float *c_ptr = colors + g * CDIM;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }
            T = next_T;
        }
    }

    if (inside) {
        render_alphas[pix_id] = 1.f - T;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] = background == nullptr ? pix_out[k] : pix_out[k] + T * background[k];
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> rara_rasterize_fwd(
    const torch::Tensor &means2d, const torch::Tensor &conics, const torch::Tensor &colors,
    const torch::Tensor &opacities, const torch::Tensor &betas, const at::optional<torch::Tensor> &background,
    const torch::Tensor &cut, const torch::Tensor &minv, const torch::Tensor &mu, const std::vector<double> &plane_cam,
    const int64_t image_width, const int64_t image_height, const int64_t tile_size,
    const torch::Tensor &tile_offsets, const torch::Tensor &flatten_ids) {
    const at::cuda::OptionalCUDAGuard guard(device_of(means2d));
    TORCH_CHECK(colors.size(-1) == 3, "rara_rasterize_fwd supports RGB colors only");
    TORCH_CHECK(plane_cam.size() == 20, "plane_cam must hold plane(4) + origin(3) + c2w rotation(9) + fx,fy,cx,cy");
    for (const auto *t : {&means2d, &conics, &colors, &opacities, &betas, &cut, &minv, &mu, &tile_offsets, &flatten_ids}) {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous(), "all inputs must be contiguous CUDA tensors");
    }
    RaRaParams p;
    float *pf = reinterpret_cast<float *>(&p);
    for (int k = 0; k < 20; ++k) {
        pf[k] = static_cast<float>(plane_cam[k]);
    }
    const uint32_t th = tile_offsets.size(-2), tw = tile_offsets.size(-1);
    auto opts = means2d.options().dtype(torch::kFloat32);
    torch::Tensor renders = torch::empty({image_height, image_width, 3}, opts);
    torch::Tensor alphas = torch::empty({image_height, image_width}, opts);
    const dim3 threads = {(uint32_t)tile_size, (uint32_t)tile_size, 1};
    const dim3 blocks = {1, th, tw};
    const uint32_t shared_mem = tile_size * tile_size * (sizeof(int32_t) + sizeof(float4) + sizeof(float3));
    rara_rasterize_fwd_kernel<3><<<blocks, threads, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
        flatten_ids.size(0),
        reinterpret_cast<const float2 *>(means2d.data_ptr<float>()),
        reinterpret_cast<const float3 *>(conics.data_ptr<float>()),
        colors.data_ptr<float>(), opacities.data_ptr<float>(), betas.data_ptr<float>(),
        background.has_value() ? background.value().data_ptr<float>() : nullptr,
        cut.data_ptr<uint8_t>(), minv.data_ptr<float>(), mu.data_ptr<float>(), p,
        image_width, image_height, tile_size, tw, th,
        tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
        renders.data_ptr<float>(), alphas.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(renders, alphas);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rara_rasterize_fwd", &rara_rasterize_fwd, "RaRa Clipper forward rasterization (single camera)");
}
