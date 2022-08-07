import os

import torch
import cv2

import torchvision.transforms as T

import numpy as np

import kitti_util

img_transform = T.Compose([T.Resize((128, 128)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def manipulate(objs, txyz):
    objs[:, 0] += txyz[0]
    objs[:, 1] += txyz[1]
    objs[:, 2] += txyz[2]
    
    # objs[:, :3] = kitti_util.rotate_yaw(objs[:, :3], np.pi/15)
    # objs[:, 6] += np.pi/15
    
    # corners = np.stack(get_corners(obj) for obj in objs)

    # kitti_util.visualize_offscreen(np.zeros([1,3]), corners, save_path='boxes.png')
    return objs
    
class KITTI(torch.utils.data.Dataset):

    def __init__(self, ):
        super().__init__()

        self.filelist = [f[:-10] for f in os.listdir("/data0/billyhe/KITTI/training/nerf") if "label" in f ]
        self.filelist.sort()

        self.cam_pos = torch.eye(4)[None, :, :]
        self.cam_pos[:, 2, 2] = -1
        self.cam_pos[:, 1, 1] = -1
  
    def __getitem__(self, idx):
        # idx=2
        id = self.filelist[idx]
       
        img = cv2.imread('/data0/billyhe/KITTI/training/nerf/%s_patch.png' % id)
        msk = cv2.imread('/data0/billyhe/KITTI/training/nerf/%s_mask.png' % id)

        with open('/data0/billyhe/KITTI/training/nerf/%s_label.txt' % id , 'r') as f:
            obj = f.readlines()[0].split()

        sid = id[:6]


        calib = kitti_util.Calibration('/data0/billyhe/KITTI/training/calib/%s.txt' % sid)
        imshape = cv2.imread('/data0/billyhe/KITTI/training/image_2/%s.png' % sid).shape
        
        render_rays = kitti_util.gen_rays(
            self.cam_pos, imshape[1], imshape[0], 
            torch.tensor([calib.f_u, calib.f_v]), 0, np.inf, 
            torch.tensor([calib.c_u, calib.c_v])
        )[0].numpy()

        xmin, xmax, ymin, ymax, tx, ty, tz, dx, dy, dz, ry = [float(a) for a in obj]

        cam_rays = render_rays[int(ymin):int(ymax), int(xmin):int(xmax), :].reshape(-1, 8)
        
        objs = np.array([tx, ty, tz, dx, dy, dz, ry]).reshape(1, 7)
       
        
        ray_o = kitti_util.world2object(np.zeros((len(cam_rays), 3)), objs)
        ray_d = kitti_util.world2object(cam_rays[:, 3:6], objs, use_dir=True)

        z_in, z_out, intersect = kitti_util.ray_box_intersection(ray_o, ray_d)
        
        bounds =  np.ones((*ray_o.shape[:-1], 2)) * -1
        bounds [intersect, 0] = z_in
        bounds [intersect, 1] = z_out

        cam_rays = np.concatenate([ray_o, ray_d, bounds], -1)


        return img, msk, cam_rays

    def __len__(self):
        return len(self.filelist)

    def __getviews__(self, idx, 
                    ry_list = [0, np.pi/2, np.pi, 1.75*np.pi],
                    txyz=[0., 1.75, 12]):
                        
        id = self.filelist[idx]
       
        img = cv2.imread('/data0/billyhe/KITTI/training/nerf/%s_patch.png' % id)
        
        with open('/data0/billyhe/KITTI/training/nerf/%s_label.txt' % id , 'r') as f:
            obj = f.readlines()[0].split()

        sid = id[:6]

        calib = kitti_util.Calibration('/data0/billyhe/KITTI/training/calib/%s.txt' % sid)
        canvas = cv2.imread('/data0/billyhe/KITTI/training/image_2/%s.png' % sid)
        
        render_rays = kitti_util.gen_rays(
            self.cam_pos, canvas.shape[1], canvas.shape[0], 
            torch.tensor([calib.f_u, calib.f_v]), 0, np.inf, 
            torch.tensor([calib.c_u, calib.c_v])
        )[0].numpy()

       
        test_data = list()
        out_shape = list()
        for ry in ry_list:
            _,_,_,_,_,_,_, l, h, w, _ = [float(a) for a in obj]
            xmin, ymin, xmax, ymax = box3d_to_image_roi(txyz + [l, h, w, ry], calib.P, canvas.shape)

            cam_rays = render_rays[int(ymin):int(ymax), int(xmin):int(xmax), :].reshape(-1, 8)

            objs = np.array(txyz + [l, h, w, ry]).reshape(1, 7)
            
            ray_o = kitti_util.world2object(np.zeros((len(cam_rays), 3)), objs)
            ray_d = kitti_util.world2object(cam_rays[:, 3:6], objs, use_dir=True)

            z_in, z_out, intersect = kitti_util.ray_box_intersection(ray_o, ray_d)

            bounds =  np.ones((*ray_o.shape[:-1], 2)) * -1
            bounds [intersect, 0] = z_in
            bounds [intersect, 1] = z_out

            cam_rays = np.concatenate([ray_o, ray_d, bounds], -1)

            out_shape.append( [int(ymax)-int(ymin), int(xmax)-int(xmin) ])

            test_data.append( collate_lambda_test(img, cam_rays) )

        return img, test_data, out_shape

    def __getscene__(self, sid, manipulation=None):
        calib = kitti_util.Calibration('/data0/billyhe/KITTI/training/calib/%06d.txt' % sid)
        canvas = cv2.imread('/data0/billyhe/KITTI/training/image_2/%06d.png' % sid)
        
        render_rays = kitti_util.gen_rays(
            self.cam_pos, canvas.shape[1], canvas.shape[0], 
            torch.tensor([calib.f_u, calib.f_v]), 0, np.inf, 
            torch.tensor([calib.c_u, calib.c_v])
        )[0].flatten(0,1).numpy()

        objs = kitti_util.read_label('/data0/billyhe/KITTI/training/label_2/%06d.txt' % sid)

        objs_pose = np.array([obj.t for obj in objs if obj.type == 'Car']).reshape(-1, 3)
        objs_dim = np.array([obj.dim for obj in objs if obj.type == 'Car']).reshape(-1, 3)
        objs_yaw = np.array([obj.ry for obj in objs if obj.type == 'Car']).reshape(-1, 1)
        # objs_box = np.stack([obj.box2d for obj in objs if obj.type == 'Car']).reshape(-1, 4)
        
        objs = np.concatenate([objs_pose, objs_dim, objs_yaw], -1)
       
        #####################
        rois = list()
        for obj in objs:
           
            xmin, ymin, xmax, ymax = box3d_to_image_roi(obj, calib.P, canvas.shape)
            
            roi = canvas[int(ymin):int(ymax), int(xmin):int(xmax), :]
            roi = T.ToTensor()(roi)
            roi = img_transform(roi)
            rois.append(roi)

        rois = torch.stack(rois)
     
        # manipulate 3d boxes 
        if manipulation is not None:
            objs = manipulate(objs, manipulation)

        # get rays from 3d boxes
        ray_o = kitti_util.world2object(np.zeros((len(render_rays), 3)), objs)
        ray_d = kitti_util.world2object(render_rays[:, 3:6], objs, use_dir=True)
       
        z_in, z_out, intersect = kitti_util.ray_box_intersection(ray_o, ray_d)
        
        bounds =  np.ones((*ray_o.shape[:-1], 2)) * -1
        bounds [intersect, 0] = z_in
        bounds [intersect, 1] = z_out

        scene_render_rays = np.concatenate([ray_o, ray_d, bounds], -1)
        _, nb, nc = scene_render_rays.shape
        scene_render_rays = scene_render_rays.reshape(canvas.shape[0], canvas.shape[1], nb, nc)
        
        return canvas, \
               torch.FloatTensor(scene_render_rays), \
               rois, \
               torch.from_numpy( np.any(intersect, 1) ),\
               torch.FloatTensor(objs)






def collate_lambda_train(batch, ray_batch_size=1024):
    imgs = list()
    msks = list()
    rays = list()
    rgbs = list()

    for el in batch:
        im, msk, cam_rays = el 
        im = T.ToTensor()(im)
        msk = T.ToTensor()(msk)
        cam_rays = torch.FloatTensor(cam_rays)

        _, H, W = im.shape
        
        pix_inds = torch.randint(0,  H * W, (ray_batch_size,))
       
        rgb_gt = im.permute(1,2,0).flatten(0,1)[pix_inds,...] 
        msk_gt = msk.permute(1,2,0).flatten(0,1)[pix_inds,...]
        ray = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds]

        imgs.append(
            img_transform(im)
        )
        msks.append(msk_gt)  
        rays.append(ray)
        rgbs.append(rgb_gt)
    
    imgs = torch.stack(imgs)
    rgbs = torch.stack(rgbs, 1)  
    msks = torch.stack(msks, 1)
    rays = torch.stack(rays, 1)  
    
    return imgs, rays, rgbs, msks



def collate_lambda_test(im, cam_rays, ray_batch_size=1024):
    imgs = list()
    rays = list()
   
    im = T.ToTensor()(im)
    cam_rays = torch.FloatTensor(cam_rays)

    N = cam_rays.shape[0]
    
    for i in range(N// ray_batch_size + 1):
        
        pix_inds = np.arange(i*ray_batch_size, i*ray_batch_size + ray_batch_size)
        
        if i == N // ray_batch_size:
            pix_inds = np.clip(pix_inds, 0, N-1)

        ray = cam_rays[pix_inds]
        rays.append(ray)
       
    imgs = img_transform(im).unsqueeze(0)
    rays = torch.stack(rays)  
    
    return imgs, rays


def get_corners(obj):
    if isinstance(obj, list):
        tx, ty, tz, l, h, w, ry = obj
    else:
        tx, ty, tz, l, h, w, ry = obj.tolist()
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
   
    R = kitti_util.roty(ry)    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + tx
    corners_3d[1,:] = corners_3d[1,:] + ty
    corners_3d[2,:] = corners_3d[2,:] + tz
    return np.transpose(corners_3d)


def box3d_to_image_roi(obj, P, imshape=None):
    corners_3d = get_corners(obj)

    # project the 3d bounding box into the image plane
    corners_2d = kitti_util.project_to_image(corners_3d, P)
    xmin, ymin = np.min(corners_2d, axis=0)
    xmax, ymax = np.max(corners_2d, axis=0)

    if imshape is not None:
        xmin = np.clip(xmin, 0, imshape[1])
        xmax = np.clip(xmax, 0, imshape[1])
        ymin = np.clip(ymin, 0, imshape[0])
        ymax = np.clip(ymax, 0, imshape[0])

    return xmin, ymin, xmax, ymax

if __name__ == "__main__":
    ds = KITTI()
    ds.__getscene__(8) 
