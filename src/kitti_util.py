from dotmap import DotMap
from matplotlib import use
import numpy as np
import cv2
import os

import torch

class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        
        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.dim = (self.l, self.h, self.w)
        self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
            (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
            (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
            (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
            (self.t[0],self.t[1],self.t[2],self.ry))



class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2'] 
        self.P = np.reshape(self.P, [3,4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3,4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2]
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative 
        self.b_y = self.P[1,3]/(-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data
    
    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3,4))
        Tr_velo_to_cam[0:3,0:3] = np.reshape(velo2cam['R'], [3,3])
        Tr_velo_to_cam[:,3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom
 
    # =========================== 
    # ------- 3d to 3d ---------- 
    # =========================== 
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))
    
    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
 
    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        ''' 
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # =========================== 
    # ------- 3d to 2d ---------- 
    # =========================== 
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]
    
    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # =========================== 
    # ------- 2d to 3d ---------- 
    # =========================== 
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
        pts_3d_rect = np.zeros((n,3))
        pts_3d_rect[:,0] = x
        pts_3d_rect[:,1] = y
        pts_3d_rect[:,2] = uv_depth[:,2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

 
def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

def load_image(img_filename):
    return cv2.imread(img_filename)

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n,1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]


def compute_box_3d(obj, P=None):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
   
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)    

    # 3d bounding box dimensions
    l = obj.l;
    w = obj.w;
    h = obj.h;
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + obj.t[0];
    corners_3d[1,:] = corners_3d[1,:] + obj.t[1];
    corners_3d[2,:] = corners_3d[2,:] + obj.t[2];
    if P is None:
        return np.transpose(corners_3d)

    #print 'cornsers_3d: ', corners_3d 
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2,:]<0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)
    
    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P);
    #print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def compute_orientation_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    '''
    
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)
   
    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l],[0,0],[0,0]])
    
    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0,:] = orientation_3d[0,:] + obj.t[0]
    orientation_3d[1,:] = orientation_3d[1,:] + obj.t[1]
    orientation_3d[2,:] = orientation_3d[2,:] + obj.t[2]
    
    # vector behind image plane?
    if np.any(orientation_3d[2,:]<0.1):
      orientation_2d = None
      return orientation_2d, np.transpose(orientation_3d)
    
    # project orientation into the image plane
    orientation_2d = project_to_image(np.transpose(orientation_3d), P);
    return orientation_2d, np.transpose(orientation_3d)

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       # use LINE_AA for opencv3
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
    return image





def visualize_offscreen(pts, boxes=None, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(1000, 1000), save_path='test.png'):
    
    import numpy as np
    from pyvirtualdisplay import Display
   

    display = Display(visible=False, size=(1280, 1024))
    display.start()

    import mayavi.mlab as mlab
    mlab.options.offscreen = True
    
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)

    mlab.points3d(0, 0, 0, color=(1, 0, 0), mode='cube', scale_factor=0.5)
    
    if boxes is not None:
        num = len(boxes)
        for n in range(num):
            b = boxes[n]

            color=(1,0,0)
            

            for k in range(0,4):
                #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
                i,j=k,(k+1)%4
                mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=1, figure=fig)

                i,j=k+4,(k+1)%4 + 4
                mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=1, figure=fig)

                i,j=k,k+4
                mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=1, figure=fig)



    mlab.view(azimuth=-90, elevation=160, focalpoint=[0, 0, 5], distance=30.0, figure=fig)
    
    mlab.savefig(save_path)
    display.stop()
    return fig



def rotate_yaw(p, yaw):
    """Rotates p with yaw in the given coord frame with y being the relevant axis and pointing downwards
    Args:
        p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
        yaw: Rotation angle
    Returns:
        p: Rotated points [N_pts, N_frames, N_samples, 3]
    """
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)

    p_x = c_y * p[..., [0]] - s_y * p[..., [2]]
    p_y = p[..., [1]]
    p_z = s_y * p[..., [0]] + c_y * p[..., [2]]

    return np.concatenate([p_x, p_y, p_z], axis=-1)

def rotate_yaw_torch(p, yaw):
    """Rotates p with yaw in the given coord frame with y being the relevant axis and pointing downwards
    Args:
        p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
        yaw: Rotation angle
    Returns:
        p: Rotated points [N_pts, N_frames, N_samples, 3]
    """
    c_y = torch.cos(yaw)
    s_y = torch.sin(yaw)

    p_x = c_y * p[..., [0]] - s_y * p[..., [2]]
    p_y = p[..., [1]]
    p_z = s_y * p[..., [0]] + c_y * p[..., [2]]

    return torch.cat([p_x, p_y, p_z], dim=-1)

def world2object(pts, obj, use_dir=False):
    
    pose, dim, theta_y = obj[:, :3], obj[:, 3:6], obj[:, 6:]
    
    pose_w = pose.copy()
    pose_w[:, 1] -= dim[:, 1] / 2

    N_obj = pose.shape[0]
    N_pts = pts.shape[0]

    pose_w = np.repeat(pose_w[None, :, :], N_pts, axis=0)
    theta_y = np.repeat(theta_y[None, :, :], N_pts, axis=0)
    dim = np.repeat(dim[None, :, :], N_pts, axis=0)

    # Describes the origin of the world system w in the object system o
    t_w_o = rotate_yaw(-pose_w, theta_y)
    N_obj = theta_y.shape[1]

    pts_w = np.repeat(pts[:, np.newaxis, ...], N_obj, axis=1)
   
    if use_dir:
        # for dir, no need to shift
        pts_o = rotate_yaw(pts_w, theta_y)
    else:
        pts_o = rotate_yaw(pts_w, theta_y) + t_w_o
   
    # Scale rays_o_v and rays_d_v for box [[-1.,1], [-1.,1], [-1.,1]]
    pts_o = pts_o / (dim / 2 + 1e-9)
    
    if use_dir:
        pts_o = pts_o / np.linalg.norm(pts_o, axis=-1, keepdims=True)
   
    return pts_o


def object2world(pts, objs):
    # pts: Nb x Np x 3
    # objs: Nb x 7

    pose, dim, theta_y = objs[:, :3], objs[:, 3:6], objs[:, 6:]

    pose_w = pose.clone()
    pose_w[:, 1] -= dim[:, 1] / 2

    N_pts = pts.shape[1]

    pose_w = pose_w[:, None, :].repeat(1, N_pts, 1)
    theta_y = theta_y[:, None, :].repeat(1, N_pts, 1)
    dim = dim[:, None, :].repeat(1, N_pts, 1)

    pts_w = pts * (dim / 2 + 1e-9)
    t_w_o = rotate_yaw_torch(-pose_w, theta_y)
    pts_w = rotate_yaw_torch(pts_w - t_w_o, -theta_y) 

    return pts_w





def ray_box_intersection(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected
    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary
    Args:
        ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified
    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = np.ones_like(ray_o) * -1. # tf.constant([-1., -1., -1.])
    if aabb_max is None:
        aabb_max = np.ones_like(ray_o) # tf.constant([1., 1., 1.])

    inv_d = np.reciprocal(ray_d)

    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    t0 = np.minimum(t_min, t_max)
    t1 = np.maximum(t_min, t_max)

    t_near = np.maximum(np.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = np.minimum(np.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])

    # Check if rays are inside boxes
    intersection_map = t_far > t_near # np.where(t_far > t_near)[0]
    
    # Check that boxes are in front of the ray origin
    positive_far = (t_far * intersection_map) > 0
    intersection_map = np.logical_and(intersection_map, positive_far)

    if not intersection_map.shape[0] == 0:
        z_ray_in = t_near[intersection_map]
        z_ray_out = t_far[intersection_map]
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def unproj_map(width, height, f, c=None, device="cpu"):
    """
    Get camera unprojection map for given image size.
    [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    :param width image width
    :param height image height
    :param f focal length, either a number or tensor [fx, fy]
    :param c principal point, optional, either None or tensor [fx, fy]
    if not specified uses center of image
    :return unproj map (height, width, 3)
    """
    if c is None:
        c = [width * 0.5, height * 0.5]
    else:
        c = c.squeeze()
    if isinstance(f, float):
        f = [f, f]
    elif len(f.shape) == 0:
        f = f[None].expand(2)
    elif len(f.shape) == 1:
        f = f.expand(2)
    Y, X = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) - float(c[1]),
        torch.arange(width, dtype=torch.float32) - float(c[0]),
    )
    X = X.to(device=device) / float(f[0])
    Y = Y.to(device=device) / float(f[1])
    Z = torch.ones_like(X)
    unproj = torch.stack((X, -Y, -Z), dim=-1)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj

    
def gen_rays(poses, width, height, focal, z_near, z_far, c=None, ndc=False):
    """
    Generate camera rays
    :return (B, H, W, 8)
    """
    num_images = poses.shape[0]
    device = poses.device
    cam_unproj_map = (
        unproj_map(width, height, focal.squeeze(), c=c, device=device)
        .unsqueeze(0)
        .repeat(num_images, 1, 1, 1)
    )
    cam_centers = poses[:, None, None, :3, 3].expand(-1, height, width, -1)
    cam_raydir = torch.matmul(
        poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1)
    )[:, :, :, :, 0]
    

    cam_nears = (
        torch.tensor(z_near, device=device)
        .view(1, 1, 1, 1)
        .expand(num_images, height, width, -1)
    )
    cam_fars = (
        torch.tensor(z_far, device=device)
        .view(1, 1, 1, 1)
        .expand(num_images, height, width, -1)
    )
    return torch.cat(
        (cam_centers, cam_raydir, cam_nears, cam_fars), dim=-1
    )  # (B, H, W, 8)

    
    