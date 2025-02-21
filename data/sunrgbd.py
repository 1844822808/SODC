import cv2
import os.path
import glob
import numpy as np
from PIL import Image

def convertPNG(pngfile, outdir):
    # READ THE DEPTH
    im_depth = cv2.imread(pngfile)
    # apply colormap on deoth image(image must be converted to 8-bit per pixel first)
    im_color = cv2.applyColorMap(cv2.convertScaleAbs(im_depth, alpha=1), 2)
    # convert to mat png
    im = Image.fromarray(im_color)
    # save image
    im.save(outdir)


class SUNRGBD_Calibration(object):
  
    def __init__(self, calib_filepath):
        lines = [line.rstrip() for line in open(calib_filepath)]
        Rtilt = np.array([float(x) for x in lines[0].split(' ')])
        self.Rtilt = np.reshape(Rtilt, (3, 3), order='F')
        K = np.array([float(x) for x in lines[1].split(' ')])
        self.K = np.reshape(K, (3, 3), order='F')
        self.f_u = self.K[0, 0]
        self.f_v = self.K[1, 1]
        self.c_u = self.K[0, 2]
        self.c_v = self.K[1, 2]

    def project_upright_depth_to_camera(self, pc):
        ''' project point cloud from depth coord to camera coordinate
            Input: (N,3) Output: (N,3)
        '''
        # Project upright depth to depth coordinate
        pc2 = np.dot(np.transpose(self.Rtilt), np.transpose(pc[:, 0:3]))  # (3,n)
        return flip_axis_to_camera(np.transpose(pc2))

    def project_upright_depth_to_image(self, pc):
        ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
        pc2 = self.project_upright_depth_to_camera(pc)
        uv = np.dot(pc2, np.transpose(self.K))  # (n,3)
        uv[:, 0] /= uv[:, 2]
        uv[:, 1] /= uv[:, 2]
        return uv[:, 0:2], pc2[:, 2]

    def project_upright_depth_to_upright_camera(self, pc):
        return flip_axis_to_camera(pc)

    def project_upright_camera_to_upright_depth(self, pc):
        return flip_axis_to_depth(pc)

    def project_image_to_camera(self, uv_depth):
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v
        pts_3d_camera = np.zeros((n, 3))
        pts_3d_camera[:, 0] = x
        pts_3d_camera[:, 1] = y
        pts_3d_camera[:, 2] = uv_depth[:, 2]
        return pts_3d_camera

    def project_image_to_upright_camerea(self, uv_depth):
        pts_3d_camera = self.project_image_to_camera(uv_depth)
        pts_3d_depth = flip_axis_to_depth(pts_3d_camera)
        pts_3d_upright_depth = np.transpose(np.dot(self.Rtilt, np.transpose(pts_3d_depth)))
        return self.project_upright_depth_to_upright_camera(pts_3d_upright_depth)
def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[:,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[:,2] *= -1
    return pc2

def load_image(img_filename):
    return cv2.imread(img_filename)

def load_depth_points(depth_filename):
    depth = np.loadtxt(depth_filename)
    return depth

def load_depth_points_mat(depth_filename):
    depth = sio.loadmat(depth_filename)['instance']
    return depth

class sunrgbd_object(object):
    ''' Load and parse object data '''
    def __init__(self, root_dir, split='training', use_v1=False):
        self.root_dir = root_dir
        self.split = split
        assert(self.split=='training')
        self.split_dir = os.path.join(root_dir)

        if split == 'training':
            self.num_samples = 10335
        elif split == 'testing':
            self.num_samples = 2860
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.depth_dir = os.path.join(self.split_dir, 'depth')
        if use_v1:
            self.label_dir = os.path.join(self.split_dir, 'label_v1')
        else:
            self.label_dir = os.path.join(self.split_dir, 'label')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, '%06d.jpg'%(idx))
        return load_image(img_filename)

    def get_depth(self, idx):
        depth_filename = os.path.join(self.depth_dir, '%06d.mat'%(idx))
        return load_depth_points_mat(depth_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return SUNRGBD_Calibration(calib_filename)

if __name__ == '__main__':
    import scipy.io as sio
    import matplotlib.pyplot as plt
    pc = sio.loadmat()['instance']
    calib= SUNRGBD_Calibration()
    uv, d = calib.project_upright_depth_to_image(pc[:, 0:3])
    depth_map = np.zeros((530, 730), dtype=np.float32)
    for i in range(uv.shape[0]):
        u = int(np.round(uv[i, 0]))
        v = int(np.round(uv[i, 1]))
        depth = d[i]
        depth_map[v-1, u-1] = depth
