import time

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags, lil_matrix
import cv2
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import open3d as o3d
import logging
from datetime import datetime
import argparse
from zoedepth.trainers.loss import SILogLoss
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from model_zoo.FFDGNet import *
from data.nyu_dataloader import *
from data.sunrgbd_dataset import *
from torchvision import transforms
from tqdm import tqdm
from zoedepth.utils.misc import *

def depth_to_point_cloud(depth_image, camera_matrix):
    height, width = depth_image.shape[:2]
    points = []
    for v in range(height):
        for u in range(width):
            depth = depth_image[v, u]
            if depth > 0:
                x = (u - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
                y = (v - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
                z = depth
                points.append([x, y, z])
    return np.array(points)

def compute_normals_from_depth_five_point(D,camera_matrix):
    return normals
def create_data_matrix(size, mask):
def create_normal_matrix(surface_normals, xres, yres, occlusion_boundaries):
    return sp.diags(diagonals, offsets, shape=(size, size), format='csr')
def create_smoothness_matrix(xres, yres):
    return sp.diags(diagonals, offsets, shape=(size, size))
def create_depth_image(surface_normals, raw_depth, lambda_D=1.0, lambda_N=10.0, lambda_S=0.0):
    return depth_image



def create_zoenetwork():
    global global_zoe_network
    conf = get_config("zoedepth", "infer")
    model_zoe_n = build_model(conf)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    global_zoe_network = model_zoe_n.to(DEVICE)

def perform_inference(image):
    global global_zoe_network
    zoe = global_zoe_network
    zoe.hook_feats.clear()
    depth_pred = zoe.infer(image)
    hook_feats = zoe.get_hook_feats()
    return depth_pred,hook_feats.copy()

def release_network():
    global global_zoe_network
    global_zoe_network = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=4, help='scale factor')
    parser.add_argument("--num_feats", type=int, default=32, help="channel number of the middle hidden layer")
    parser.add_argument('--device', default="0", type=str, help='which gpu use')
    parser.add_argument("--root_dir", type=str, default="", help="root dir of dataset")
    parser.add_argument("--model_dir", type=str, default="", help="checkpoint")
    opt = parser.parse_args()
    dataset_name = opt.root_dir.split('\\')[-1]
    values=[80]
    net = FDG_network2(num_feats=opt.num_feats, kernel_size=3, scale=opt.scale).cuda()
    net.load_state_dict(torch.load(opt.model_dir, map_location='cuda:0'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    data_transform = transforms.Compose([transforms.ToTensor()])

    create_zoenetwork()
    for aaa in values:
        test_dataset = NYuV2Test(root_dir=opt.root_dir, scale=opt.scale, aaa=1.0, transform=data_transform, train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
        data_num = len(test_dataloader)

        with torch.no_grad():
            net.eval()
            metrics = RunningAverageDict()
            pixels_below_threshold_list=[]
            for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                image_raw, depth_raw, image_crop, depth_crop, box, rawhole_depth_crop, camera_matrix = \
                    data['image_raw'].cuda(), data['depth_raw'].cuda(), \
                    data['image_crop'].cuda(), data['depth_crop'].cuda(), \
                    data['box'].cuda(), data['rawhole_depth_crop'].cuda(), data['camera_matrix']
                camera_matrix = camera_matrix.squeeze().cpu().numpy()
                depth_pred2, _ = perform_inference(image_crop)
                depth_pred2_lr = nn.functional.interpolate(depth_pred2, (384 // opt.scale, 512 // opt.scale),
                                                           mode='bilinear',
                                                           align_corners=False)
                depth_pred2_lr = depth_pred2_lr / 10
                out = net((image_crop, depth_pred2_lr))
                out = out.cpu()
                out = out.squeeze().numpy()
                N = compute_normals_from_depth_five_point(out,camera_matrix)

                rawhole_depth_crop = rawhole_depth_crop.cpu()
                rawhole_depth_crop = rawhole_depth_crop.squeeze().numpy()
                depth_image = create_depth_image(N, rawhole_depth_crop * 10, lambda_D=1.0, lambda_N=20, lambda_S=0.001)

            def r(m): return round(m, 3)
            metrics = {k: r(v) for k, v in metrics.get_value().items()}
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    release_network()
