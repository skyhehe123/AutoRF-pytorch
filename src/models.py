import torch
import torch.nn as nn
import torch.nn.functional as F

from math import pi

from torchvision.models import resnet34

import matplotlib.pyplot as plt

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(True)

    def forward(self, x):
        # Extract feature pyramid from image. See Section 4.1., Section B.1 in the
        # Supplementary Materials, and: https://github.com/sxyu/pixel-nerf/blob/master/src/model/encoder.py.

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        feats1 = self.resnet.relu(x)

        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))
        feats3 = self.resnet.layer2(feats2)
        feats4 = self.resnet.layer3(feats3)

        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i], latent_sz, mode="bilinear", align_corners=True
            )

        latents = torch.cat(latents, dim=1)
        return F.max_pool2d(latents, kernel_size=latents.size()[2:])[:, :, 0, 0]

class Decoder(nn.Module):
   
    def __init__(self, 
                 hidden_size=128, 
                 n_blocks=8, 
                 n_blocks_view=1,
                 skips=[4], 
                 n_freq_posenc=10, 
                 n_freq_posenc_views=4, 
                 z_dim=128, 
                 rgb_out_dim=3
    ):
        super().__init__()
 
        self.n_freq_posenc = n_freq_posenc
        self.n_freq_posenc_views = n_freq_posenc_views
        self.skips = skips
        self.z_dim = z_dim

        self.n_blocks = n_blocks
        self.n_blocks_view = n_blocks_view

       
        dim_embed = 3 * self.n_freq_posenc * 2
        dim_embed_view = 3 * self.n_freq_posenc_views * 2

        # Density Prediction Layers
        self.fc_in = nn.Linear(dim_embed, hidden_size)
        
        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)
        ])
        n_skips = sum([i in skips for i in range(n_blocks - 1)])
        
        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList(
                [nn.Linear(z_dim, hidden_size) for i in range(n_skips)]
            )
            self.fc_p_skips = nn.ModuleList([
                nn.Linear(dim_embed, hidden_size) for i in range(n_skips)
            ])
        
        self.sigma_out = nn.Linear(hidden_size, 1)

        # Feature Prediction Layers
        self.fc_z_view = nn.Linear(z_dim, hidden_size)
        self.feat_view = nn.Linear(hidden_size, hidden_size)
        self.fc_view = nn.Linear(dim_embed_view, hidden_size)
        self.feat_out = nn.Linear(hidden_size, rgb_out_dim)
    
        self.blocks_view = nn.ModuleList(
            [nn.Linear(dim_embed_view + hidden_size, hidden_size) for _ in range(n_blocks_view - 1)]
        )

        self.fc_shape = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        
        self.fc_app = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        
        
    def transform_points(self, p, views=False):
        L = self.n_freq_posenc_views if views else self.n_freq_posenc
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * pi * p),
             torch.cos((2 ** i) * pi * p)],
            dim=-1) for i in range(L)], dim=-1)
        return p_transformed

    def forward(self, p_in, ray_d, latent=None):
        
        z_shape = self.fc_shape(latent)
        z_app = self.fc_app(latent)

        B, N, _ = p_in.shape
        
        z_shape = z_shape[:, None, :].repeat(1, N, 1)
        z_app = z_app[:, None, :].repeat(1, N, 1)

        p = self.transform_points(p_in)
        net = self.fc_in(p)
        
        if z_shape is not None:
            net = net + self.fc_z(z_shape)

        net = F.relu(net)

        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            net = F.relu(layer(net))
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                net = net + self.fc_z_skips[skip_idx](z_shape)
                net = net + self.fc_p_skips[skip_idx](p)
                skip_idx += 1
        sigma_out = self.sigma_out(net)

        net = self.feat_view(net)
        net = net + self.fc_z_view(z_app)
        
      
        ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
        ray_d = self.transform_points(ray_d, views=True)
        net = net + self.fc_view(ray_d)
        net = F.relu(net)
        if self.n_blocks_view > 1:
            for layer in self.blocks_view:
                net = F.relu(layer(net))

        feat_out = self.feat_out(net)

    
        return feat_out, sigma_out


class PixelNeRFNet(torch.nn.Module):
    def __init__(self,):
        
        super().__init__()
      
        self.encoder = ImageEncoder()
        self.decoder = Decoder()
       
    def encode(self, images):
        # self.encoder.eval()
        # with torch.no_grad():
        return self.encoder(images)

    def forward(self, xyz, viewdirs=None, latent=None):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
       
        rgb, sigma = self.decoder(xyz, viewdirs, latent)
       
        output_list = [torch.sigmoid(rgb), F.softplus(sigma)]
        output = torch.cat(output_list, dim=-1)
        
        return output

    






