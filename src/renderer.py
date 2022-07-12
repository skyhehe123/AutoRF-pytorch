import torch


class NeRFRenderer(torch.nn.Module):
    

    def __init__(
        self,
        n_coarse=64,
        noise_std=0.0,
        white_bkgd=False,
    ):
        super().__init__()
        self.n_coarse = n_coarse
        self.noise_std = noise_std
        self.white_bkgd = white_bkgd
      
    def sample_from_ray(self, rays):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

        step = 1.0 / self.n_coarse
        B = rays.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step
        return near * (1 - z_steps) + far * z_steps  # (B, Kf)
       

    def nerf_predict(self, model, rays, z_samp, latent=None):
        # the points on the non-intersect ray has z_vals = -1
        
        B = latent.shape[0]
        NB, K = z_samp.shape
       
        # (B, K, 3)
        points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
        
        
        viewdirs = rays[:, None, 3:6].expand(-1, K, -1).contiguous()  # (B, K, 3)
        
        split_points = points.view(-1, B, K, 3).permute(1, 0, 2, 3).reshape(B, -1, 3)
        split_viewdirs = viewdirs.view(-1, B, K, 3).permute(1, 0, 2, 3).reshape(B, -1, 3)
        
        out = model(split_points, viewdirs=split_viewdirs, latent=latent)
        C = out.shape[-1]
        out = out.view(B, -1, K,C).permute(1, 0, 2, 3).reshape(NB, K, C)
        
        
        out = out.view(NB, K, -1)  # (B, K, 4 or 5)
        rgbs = out[..., :3]  # (B, K, 3)
        sigmas = out[..., 3]  # (B, K)
        
        if self.training and self.noise_std > 0.0:
            sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std
        
        return rgbs, sigmas

    def volume_render(self, rgbs, sigmas, z_samp):
        
        deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
        # delta_inf = rays[:, -1:] - z_samp[:, -1:]
        delta_inf = torch.full_like(z_samp[..., :1], 0)
        deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)

        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (B, K)
        
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
        )  # (B, K+1) = [1, a1, a2, ...]
        
        T = torch.cumprod(alphas_shifted, -1)  # (B)
        weights = alphas * T[:, :-1]  # (B, K)
        
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3)
        depth_final = torch.sum(weights * z_samp, -1)  # (B)
        
        if self.white_bkgd:
            # White background
            pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
            rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)
        
        
        return (
            
            rgb_final,
            depth_final,
            weights,
        )

    def forward(
        self, model, rays, latent=None, 
    ):
        
        assert len(rays.shape) == 3
        N, B, _ = rays.shape
        
        rays = rays.view(-1, 8)  # (N * B, 8)
        z_coarse = self.sample_from_ray(rays)  # (B, Kc)
        
        rgbs, sigmas = self.nerf_predict(model, rays, z_coarse, latent)
        
        rgb, depth, weights = self.volume_render(
            rgbs, sigmas, z_coarse
        )

        outputs = {
            'rgb': rgb.view(N, B, -1),
            'depth': depth.view(N, B),
            'weights': weights.view(N, B, -1),
            'intersect': (z_coarse[:, 0] != -1).view(N, B)
        }
        
        return outputs

   