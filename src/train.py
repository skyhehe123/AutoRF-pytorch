import os

import kitti_util
import numpy as np
from models import PixelNeRFNet
import torch.nn.functional as F
from renderer import NeRFRenderer

import torch
import random

from kitti import *

from functools import partial

import argparse
import imageio
import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ray_batch_size", type=int, default=2048)
    parser.add_argument("--print_interval", type=int, default=5)
    parser.add_argument("--vis_interval", type=int, default=100)
    parser.add_argument("--ckpt_interval", default=5, help='checkpoint interval (in epochs)')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=float, default=1000000)
    parser.add_argument("--save_path", type=str, default='output')
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()

def make_canvas(patches):
    image = patches.pop(0)
    banner = list()
    hmax = max([p.shape[0] for p in patches]) + 10
    for p in patches:
        H, W, _ = p.shape
        a = (hmax - H ) // 2
        b = hmax - H - a
        pp = np.pad(p, ((a, b), (0, 0), (0, 0)))
        banner.append(pp)
    banner = np.concatenate(banner, 1)
    imW, bnW = image.shape[1], banner.shape[1]
    a = (bnW - imW) // 2
    b = bnW - imW - a
    image = np.pad(image, ((0, 0), (a, b), (0, 0)))
    canvas = np.concatenate([image, banner], 0)
    return canvas
    
class PixelNeRFTrainer():
    def __init__(self, args, net, renderer, train_dataset, test_dataset, device):
        super().__init__()
        self.args = args
        self.device = device
        self.net = net
        self.renderer = renderer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            collate_fn = partial(collate_lambda_train, ray_batch_size=args.ray_batch_size)
        )
        
        os.makedirs(self.args.save_path, exist_ok = True)

        
        self.num_epochs = args.epochs

        self.optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optim, milestones=[100, 150], gamma=0.1
        )
       
       
    
    def train_step(self, data, is_train=True):
    
        src_images, all_rays, all_rgb_gt, all_mask_gt = data

        src_images = src_images.to(self.device)
        all_rays = all_rays.to(self.device)
        all_rgb_gt = all_rgb_gt.to(self.device)
        all_mask_gt = all_mask_gt.to(self.device)

        latent = self.net.encode(src_images)
        
        render_dict = self.renderer(self.net, all_rays, latent)
        
        render_rgb = render_dict['rgb']
    
        intersect = render_dict['intersect']
        
       
        render_rgb = render_rgb[intersect, ...]
        all_rgb_gt = all_rgb_gt[intersect, ...]
        all_mask_gt = all_mask_gt[intersect, ...]
        
        loss = F.mse_loss(render_rgb, all_rgb_gt * all_mask_gt, reduction='mean') 
        #loss = loss.sum() / all_mask_gt.sum()

        if is_train:
            loss.backward()
        
        return loss


    def vis_step(self, data):
        src_images, all_rays = data
        
       
        all_rays = all_rays.to(device)
        src_images = src_images.to(device)

        self.net.eval()
        pred_rgb = list()
        with torch.no_grad():
            latent = self.net.encode(src_images)
            for batch_rays in torch.split(all_rays, self.args.batch_size):
                pred_rgb.append( self.renderer(self.net, batch_rays.flatten(0, 1), latent)['rgb'] )
        
        self.net.train()
        
        pred_rgb = torch.cat(pred_rgb, 0).view(-1, 3)
       
        return pred_rgb


    def vis_scene(self, idx, manipulation):
        image, all_rays, rois, intersect, objs = self.test_dataset.__getscene__(idx, manipulation)
     
        H, W, _ = image.shape
        
        self.net.eval()
    
        all_rays = all_rays.to(device)
        src_images = rois.to(device)
        intersect = intersect.to(device)
        objs = objs.to(device)

        all_rays = all_rays.view(H*W, -1, 8)
        valid_rays = all_rays[intersect, ...]

        _, Nb, _ = valid_rays.shape
        Nk = self.renderer.n_coarse
       
        with torch.no_grad():
            latents = self.net.encode(src_images)
            
            rgb_map = list()
            for batch_rays in tqdm.tqdm(torch.split(valid_rays, self.args.batch_size)):
                
                rays = batch_rays.view(-1, 8)  # (N * B, 8)
                z_coarse = self.renderer.sample_from_ray(rays)
                empty_space = z_coarse == -1

                rgbs, sigmas = self.renderer.nerf_predict(self.net, rays, z_coarse, latents)

                pts_o = rays[:, None, :3] + z_coarse[:, :, None] * rays[:, None, 3:6]
                pts_o = pts_o.view(-1, Nb, Nk, 3).permute(1, 0, 2, 3).contiguous()
                
                pts_w = kitti_util.object2world(pts_o.view(Nb, -1, 3), objs)
                pts_w = pts_w.view(Nb, -1, Nk, 3).permute(1, 0, 2, 3).contiguous()

                z_world = torch.norm(pts_w, p=2, dim=-1).view_as(z_coarse)
                z_world[empty_space] = -1
                
                z_world = z_world.view(-1, Nb*Nk)

                z_sort = torch.sort(z_world, 1).values
                z_args = torch.searchsorted(z_sort, z_world)
            
                rgbs[empty_space, ...] = 0
                sigmas[empty_space] = 0

                rgbs = rgbs.view(-1, Nb * Nk, 3)
                sigmas = sigmas.view(-1, Nb * Nk)

                rgbs_sort = torch.zeros_like(rgbs).scatter_(1, z_args[:, :, None].repeat(1, 1, 3), rgbs)
                sigmas_sort = torch.zeros_like(sigmas).scatter_(1, z_args, sigmas)

                rgb, depth, weights = self.renderer.volume_render(rgbs_sort, sigmas_sort, z_sort)
                
                # rgb = self.renderer(self.net, batch_rays, latents)['rgb'][:, 0, :]

                rgb_map.append(rgb)

            rgb_map = torch.cat(rgb_map, 0)  

            canvas = torch.zeros(H*W, 3).type_as(all_rays)
            canvas[intersect, :] = rgb_map
            canvas = (canvas.view(H, W, 3).cpu().numpy() * 255).astype(np.uint8) 

            return canvas

    def train(self):
       
        for epoch in range(self.num_epochs):
        
            batch = 0
            for data in self.train_data_loader:
                losses = self.train_step(data)

                self.optim.step()
                self.optim.zero_grad()

                if batch % self.args.print_interval == 0:
                    print("E", epoch,"B",batch, "loss", losses.item(),"lr", self.optim.param_groups[0]["lr"])
                
                if batch % self.args.vis_interval == 0:  
                    idx = random.choice(range(len(self.test_dataset)))
                
                    img, test_data, out_shape = self.test_dataset.__getviews__(idx)
                    
                    patches = [img]
                    for d, hw in zip(test_data, out_shape):
                        vis = self.vis_step(d)
                        h, w = hw
                        vis = vis[:h*w, :].reshape(h, w, 3).cpu()
                        patches.append((vis.numpy() * 255).astype(np.uint8))

                    canvas = make_canvas(patches)
                    
                    imageio.imwrite( 
                        os.path.join(
                            self.args.save_path,"test.png",#"{:04}_{:04}_vis.png".format(epoch, batch),
                        ), canvas)
               
                batch += 1
            
            if (epoch + 1) % self.args.ckpt_interval == 0:
                torch.save(
                    self.net.state_dict(), 
                    os.path.join(
                        self.args.save_path,"epoch_%d.ckpt" % (epoch + 1),
                    )
                )
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

   
if __name__ == "__main__":

    args = get_args()

    device = torch.device("cuda:0") 
    
    net = PixelNeRFNet().to(device=device)
    
    renderer = NeRFRenderer().to(device=device)

    trainer = PixelNeRFTrainer(
        args, net, renderer, 
        KITTI(),
        KITTI(),
        device
    )
    
    
    if args.demo:
        trainer.net.load_state_dict(torch.load(os.path.join(args.save_path,"200.ckpt")))

        with imageio.get_writer(os.path.join(args.save_path, 'scene.gif'), mode='I', duration=0.5) as writer:
            for z in [10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6]:
                canvas = trainer.vis_scene(8, [0, 0, z])
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                writer.append_data(canvas)
        writer.close()
        exit(0)

    trainer.train()
   