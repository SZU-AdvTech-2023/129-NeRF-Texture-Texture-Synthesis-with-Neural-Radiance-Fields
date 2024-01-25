import math
from turtle import bgcolor
from unittest import TestLoader
import trimesh
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import raymarching
from .utils import custom_meshgrid

def sample_pdf(bins, weights, n_samples, det=False):
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def plot_pointcloud(pc, color=None):
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()

class NeRFRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1,
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 cal_dist_loss=False,
                 regularization=False,
                 optimize_camera=False,
                 camera_num=None,
                 optimize_gamma=False,
                 ):
        super().__init__()
        self.field_name = 'ngp'
        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius  # radius of the background sphere.
        self.cal_dist_loss = cal_dist_loss
        self.regularization = regularization
        self.visual_modes = []
        self.light_visual_modes = []
        self.optimize_camera = optimize_camera
        self.camera_num = camera_num
        self.optimize_gamma = optimize_gamma

        if self.optimize_camera:
            self.dRs = nn.Parameter(torch.zeros([camera_num, 3]), requires_grad=True)
            self.dts = nn.Parameter(torch.zeros([camera_num, 3]), requires_grad=True)
            self.dfs = nn.Parameter(torch.zeros([camera_num, 2]), requires_grad=True)
        if self.optimize_gamma:
            self.gammas = nn.Parameter(2.4 * torch.ones([camera_num]), requires_grad=True)

        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3])  # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8,
                                           dtype=torch.uint8)  # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32)  # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0

    def get_results(self, get_results_func):
        assert self.optimize_camera, 'Only valid when optimizing camera parameters.'
        results = get_results_func(self.dRs, self.dts, self.dfs)
        return results

    def camera_regularization(self):
        loss = self.dfs.norm(dim=-1).mean() + self.dRs.norm(dim=-1).mean() + self.dts.norm(dim=-1).mean() * 1e-4
        return loss

    def regular_loss(self, step):
        loss = 0.
        if self.optimize_camera:
            loss_cam = self.camera_regularization()
            weight = 1e2 if step > 2000 else 1e4
            loss = loss + weight * loss_cam
        return loss

    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def switch_light_mode(self, mode_id=None):
        if mode_id is None:
            mode_id = (self.light_visual_mode_id + 1) % len(self.light_visual_modes)
        self.light_visual_mode_id = mode_id
        self.light_visual_mode = self.light_visual_modes[self.light_visual_mode_id]
        return self.light_visual_mode

    def switch_visual_mode(self, mode_id=None):
        if mode_id is None:
            mode_id = (self.visual_mode_id + 1) % len(self.visual_modes)
        self.visual_mode_id = mode_id
        self.visual_mode = self.visual_modes[self.visual_mode_id]
        return self.visual_mode

    def reset_extra_state(self):
        if not self.cuda_ray:
            return
            # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def initialize_states(self, fp16):
        with torch.cuda.amp.autocast(enabled=fp16):
            # Clear density grids
            self.update_extra_state(decay=0., force_full_update=True, force_full_grid=True)
            # Random sampling for times to estimate grid density
            for _ in range(30):
                self.update_extra_state(decay=1., force_full_update=True, force_full_grid=True)

    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, **kwargs):
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            # z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1))  # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1])  # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps,
                                        det=not self.training).detach()  # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(
                    -1)  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:])  # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            # new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1)  # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1)  # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1))  # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T+t]

        distortion_loss = None
        # if self.cal_dist_loss:
        #     distortion_loss = eff_distloss(weights, z_vals, deltas)

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4  # hard coded
        rgbs = self.color(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
        rgbs = rgbs.view(N, -1, 3)  # [N, T+t, 3]

        # print(xyzs.shape, 'valid_rgb:', mask.sum().item())

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)  # [N]

        # calculate depth
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)  # [N, 3], in [0, 1]

        # mix background color
        if self.bg_radius > 0:
            polar = raymarching.polar_from_ray(rays_o, rays_d, self.bg_radius)  # [N, 2] in [-1, 1]
            bg_color = self.background(polar, rays_d.reshape(-1, 3))  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        mask = (weights_sum > .95).view(*prefix)
        return {
            'depth': depth,
            'image': image,
            'mask': mask,
            'normal_error': None,
            'distortion_loss': distortion_loss,
        }

    def sample(self, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024,
               **kwargs):
        prefix = rays_o.shape[:-1]
        counter = self.step_counter[self.local_step % 16]
        counter.zero_()  # set to 0  # torch.zeros([], device=rays_o.device, dtype=torch.int32)
        self.local_step += 1
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d,
                                                     self.aabb_train if self.training else self.aabb_infer,
                                                     self.min_near)
        if self.optimize_camera:
            xyzs, dirs, deltas, rays = raymarching.march_rays_train_differentiable(rays_o, rays_d, self.bound,
                                                                                   self.density_bitfield, self.cascade,
                                                                                   self.grid_size, nears.detach(),
                                                                                   fars.detach(), counter,
                                                                                   self.mean_count, perturb, 128,
                                                                                   force_all_rays, dt_gamma, max_steps)
        else:
            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield,
                                                                    self.cascade, self.grid_size, nears, fars, counter,
                                                                    self.mean_count, perturb, 128, force_all_rays,
                                                                    dt_gamma, max_steps)
        return xyzs, dirs, deltas, rays, prefix

    def run_cuda(self, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024,
                 **kwargs):
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d,
                                                     self.aabb_train if self.training else self.aabb_infer,
                                                     self.min_near)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            polar = raymarching.polar_from_ray(rays_o, rays_d, self.bg_radius)  # [N, 2] in [-1, 1]
            bg_color = self.background(polar, rays_d)  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        distortion_loss = None
        render_data_dict = {}

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_()  # set to 0
            self.local_step += 1

            if self.optimize_camera:
                xyzs, dirs, deltas, rays = raymarching.march_rays_train_differentiable(rays_o, rays_d, self.bound,
                                                                                       self.density_bitfield,
                                                                                       self.cascade, self.grid_size,
                                                                                       nears.detach(), fars.detach(),
                                                                                       counter, self.mean_count,
                                                                                       perturb, 128, force_all_rays,
                                                                                       dt_gamma, max_steps)
            else:
                xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound,
                                                                        self.density_bitfield, self.cascade,
                                                                        self.grid_size, nears, fars, counter,
                                                                        self.mean_count, perturb, 128, force_all_rays,
                                                                        dt_gamma, max_steps)
            frame_index = kwargs['index'] if 'index' in kwargs.keys() else None
            sigmas, rgbs, data_dict = self(xyzs, dirs, frame_index=frame_index)
            sigmas = self.density_scale * sigmas
            if len(sigmas.shape) == 2:
                K = sigmas.shape[0]
                depths = []
                images = []
                for k in range(K):
                    weights_sum, depth, image = raymarching.composite_rays_train(sigmas[k], rgbs[k], deltas, rays)
                    image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                    depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                    images.append(image.view(*prefix, 3))
                    depths.append(depth.view(*prefix))
                depth = torch.stack(depths, axis=0)  # [K, B, N]
                image = torch.stack(images, axis=0)  # [K, B, N, 3]

            else:
                weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays)
                image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                image = image.view(*prefix, 3)
                depth = depth.view(*prefix)
                for key in data_dict.keys():
                    if data_dict[key] is not None and 'normal' in key:
                        weights_sum_data, _, data_composed = raymarching.composite_rays_train(sigmas.detach(),
                                                                                              data_dict[key], deltas,
                                                                                              rays)
                        data_composed = data_composed + (1 - weights_sum_data).unsqueeze(-1) * torch.zeros_like(
                            bg_color)
                        data_composed = data_composed.view(*prefix, 3)
                        render_data_dict[key] = data_composed
                    else:
                        render_data_dict[key] = data_dict[key]
        else:
            if 'euler' not in kwargs.keys():
                kwargs['euler'] = None
            if 'is_gui_mode' not in kwargs.keys():
                kwargs['is_gui_mode'] = True
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)

            n_alive = N
            alive_counter = torch.zeros([1], dtype=torch.int32, device=device)

            rays_alive = torch.zeros(2, n_alive, dtype=torch.int32, device=device)  # 2 is used to loop old/new
            rays_t = torch.zeros(2, n_alive, dtype=dtype, device=device)

            step = 0
            i = 0
            while step < max_steps:

                # count alive rays
                if step == 0:
                    # init rays at first step.
                    torch.arange(n_alive, out=rays_alive[0])
                    rays_t[0] = nears
                else:
                    alive_counter.zero_()
                    raymarching.compact_rays(n_alive, rays_alive[i % 2], rays_alive[(i + 1) % 2], rays_t[i % 2],
                                             rays_t[(i + 1) % 2], alive_counter)
                    n_alive = alive_counter.item()  # must invoke D2H copy here

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)
                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], rays_o,
                                                            rays_d, self.bound, self.density_bitfield, self.cascade,
                                                            self.grid_size, nears, fars, 128, perturb, dt_gamma,
                                                            max_steps)
                sigmas, rgbs, data_dict = self(xyzs, dirs, euler=kwargs['euler'], is_gui_mode=kwargs['is_gui_mode'])
                sigmas = self.density_scale * sigmas

                raymarching.composite_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], sigmas, rgbs, deltas,
                                           weights_sum, depth, image)
                step += n_step
                i += 1

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
        mask = (weights_sum > .95).view(*prefix)

        return {
            'depth': depth,
            'image': image,
            'mask': mask,
            'distortion_loss': distortion_loss,
            **render_data_dict,
        }

    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):
        if not self.cuda_ray:
            return

        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]

        fx, fy, cx, cy = intrinsic

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)

        count = torch.zeros_like(self.density_grid)
        poses = poses.to(count.device)

        for xs in X:
            for ys in Y:
                for zs in Z:

                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                       dim=-1)  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0)  # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3]  # [S, N, 3]

                            mask_z = cam_xyzs[:, :, 2] > 0  # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1)  # [N]
                            count[cas, indices] += mask
                            head += S
        self.density_grid[count == 0] = -1

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128, force_full_update=False, force_full_grid=False):

        if not self.cuda_ray:
            return
        tmp_grid = - torch.ones_like(self.density_grid)

        # full update.
        if self.iter_density < 16 or force_full_update:
            # if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:

                        # construct points
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                           dim=-1)  # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long()  # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            # scale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # query density
                            sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                            sigmas *= self.density_scale
                            # assign
                            tmp_grid[cas, indices] = sigmas

        else:
            N = self.grid_size ** 3 // 4
            for cas in range(self.cascade):
                # random sample some positions
                coords = torch.randint(0, self.grid_size, (N, 3),
                                       device=self.density_grid.device)  # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long()  # [N]
                # random sample occupied positions
                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1)  # [Nz]
                if occ_indices.shape[0] > 0:
                    rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long,
                                              device=self.density_grid.device)
                    occ_indices = occ_indices[rand_mask]  # [Nz] --> [N], allow for duplication
                    occ_coords = raymarching.morton3D_invert(occ_indices)  # [N, 3]
                    # concat
                    indices = torch.cat([indices, occ_indices], dim=0)
                    coords = torch.cat([coords, occ_coords], dim=0)
                # same below
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                # scale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                sigmas *= self.density_scale
                tmp_grid[cas, indices] = sigmas

        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        if force_full_grid:
            valid_mask = torch.ones_like(valid_mask)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(
            self.density_grid.clamp(min=0)).item()  # -1 non-training regions are viewed as 0 density.
        self.iter_density += 1

        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, force_staged=False, **kwargs):
        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device
        if force_staged or (staged and not self.cuda_ray):
            depth = torch.empty((B, N), device=device)
            mask = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b + 1, head:tail], rays_d[b:b + 1, head:tail], **kwargs)
                    depth[b:b + 1, head:tail] = results_['depth']
                    image[b:b + 1, head:tail] = results_['image']
                    mask[b:b + 1, head:tail] = results_['mask']
                    head += max_ray_batch

            results = {}
            results['depth'] = depth
            results['image'] = image
            results['mask'] = mask
        else:
            results = _run(rays_o, rays_d, **kwargs)

        return results