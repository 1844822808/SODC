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

# fx_d = 5.8262448167737955e+02;
# fy_d = 5.8269103270988637e+02;
# cx_d = 3.1304475870804731e+02;
# cy_d = 2.3844389626620386e+02;

def depth_to_point_cloud(depth_image, camera_matrix):
    height, width = depth_image.shape[:2]
    points = []
    for v in range(height):
        for u in range(width):
            depth = depth_image[v, u]
            if depth > 0:
                # 根据像素坐标和深度计算三维坐标

                x = (u - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
                y = (v - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
                z = depth
                # 保存三维坐标
                points.append([x, y, z])
    return np.array(points)

def compute_normals_from_depth_five_point(D,camera_matrix):
    point_cloud_xyz1 = depth_to_point_cloud(D, camera_matrix)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_xyz1)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    normals = np.asarray(pcd.normals)
    return normals
def create_data_matrix(size, mask):
    data = np.ones(mask.sum(), dtype=np.float64)
    row = np.arange(size)[mask]
    col = np.arange(size)[mask]
    return sp.csr_matrix((data, (row, col)), shape=(size, size))
def create_normal_matrix(surface_normals, xres, yres, occlusion_boundaries):
    size = xres * yres

    diag0 = np.zeros(size)
    diag1 = np.zeros(size-1)
    diag2 = np.zeros(size-xres)
    # for w in range(0,xres):
    #     for h in range(0,yres):
    for i in range(0, size):
        if (i+1) % xres != 0:
            if occlusion_boundaries == 1:
                u = i % xres
                v = i // xres
                # normals = (surface_normals[i]+surface_normals[i+1])/2
                normals = surface_normals[i]
                x = (u - camera_matrix[0, 2]) / camera_matrix[0, 0]
                y = (v - camera_matrix[1, 2]) / camera_matrix[1, 1]

                ai = np.dot(np.array([x, y, 1]), normals)
                ai1 = np.dot(np.array([x+1/camera_matrix[0, 0], y, 1]), normals)

                diag0[i] = diag0[i] + ai*ai
                diag0[i + 1] = diag0[i + 1] + ai1*ai1
                diag1[i] = diag1[i] - ai*ai1

        if i < (size-xres):
            if occlusion_boundaries == 1:

                u = i % xres
                v = i // xres
                # normals = (surface_normals[i]+surface_normals[i+xres])/2
                normals = surface_normals[i]
                x = (u - camera_matrix[0, 2]) / camera_matrix[0, 0]
                y = (v - camera_matrix[1, 2]) / camera_matrix[1, 1]

                ai = np.dot(np.array([x, y, 1]), normals)
                aiw = np.dot(np.array([x, y+1/camera_matrix[1, 1], 1]), normals)

                diag0[i] = diag0[i] + ai*ai
                diag0[i + xres] = diag0[i + xres] + aiw*aiw
                diag2[i] = diag2[i] - ai*aiw


    diagonals = [diag0, diag1, diag1, diag2, diag2]
    offsets = [0, -1, 1, -xres, xres]

    return sp.diags(diagonals, offsets, shape=(size, size), format='csr')
def create_smoothness_matrix(xres, yres):
    size = xres * yres
    diagonals = []
    offsets = []

    # Main diagonal
    diagonals.append(4 * np.ones(size))
    offsets.append(0)

    # Neighbor diagonals
    for offset in [-1, 1, -xres, xres]:
        diagonals.append(-1 * np.ones(size))
        offsets.append(offset)

    # Boundary conditions: adjust for non-square images
    for i in range(yres):
        if i > 0:
            diagonals[-2][i * xres] = 0  # Left boundary
        if i < yres - 1:
            diagonals[-1][(i + 1) * xres - 1] = 0  # Right boundary

    return sp.diags(diagonals, offsets, shape=(size, size))
def create_depth_image(surface_normals, raw_depth,lambda_D=1.0, lambda_N=10.0, lambda_S=1.0):
    height, width = raw_depth.shape
    xres = width
    yres = height

    n = xres * yres

    # Flatten the images
    surface_normals = surface_normals.reshape(-1, 3)


    raw_depth = raw_depth.flatten()
    mask = raw_depth != 0
    # Create sparse matrix for smoothness term


    # Assemble the total matrix A
    A_D = create_data_matrix(n,mask)
    A_S= create_smoothness_matrix(xres, yres)
    A_N = create_normal_matrix(surface_normals, xres, yres, occlusion_boundaries=1)
    A = lambda_D * A_D + lambda_N * A_N+A_S* lambda_S

    # Create the right-hand side vector b
    b = lambda_D *raw_depth

    # Solve the system using sparse Cholesky decomposition
    cho_factor = spla.splu(A.tocsc())
    depth_solution = cho_factor.solve(b)

    # Reshape the solution back to the image shape
    depth_image = depth_solution.reshape(height, width)

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
    parser.add_argument("--root_dir", type=str, default=r"C:\zj\ffc-HRDE\data\NYU_v2", help="root dir of dataset")
    parser.add_argument("--cftl", type=bool, default=False, help="CFTL")
    parser.add_argument("--model_dir", type=str,
                        default=r"C:\zj\localfusion\ffd\trainlog\1\20240923211246-lr_0.0001-s_4-NYU_v2-b_1\best_model.pth", help="checkpoint")
    opt = parser.parse_args()

    dataset_name = opt.root_dir.split('\\')[-1]
    # print(opt)

    values=[80]
    #[0.01,0.1,0,1.0,10.0][0.001,0.0001,0.00001]
    #[2.0,20.0,40.0,0.5,0.05,0.025]10.0[80.0,100.0,200.0,400.0]
    # [0.00255, 0.01, 0.1, 0.99, 0.3, 0.5, 0.7, 0.9, 1.0, 0.00722]
    net = FFDG_network2(num_feats=opt.num_feats, kernel_size=3, scale=opt.scale).cuda()
    net.load_state_dict(torch.load(opt.model_dir, map_location='cuda:0'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)


    data_transform = transforms.Compose([transforms.ToTensor()])



    create_zoenetwork()
    for aaa in values:
        test_dataset = NYuV2Test(root_dir=opt.root_dir, scale=opt.scale, aaa=1.0, transform=data_transform, train=False)

        # test_dataset = SUNRGBDPseudoDataset(transform=data_transform)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
        data_num = len(test_dataloader)

        with torch.no_grad():
            net.eval()
            metrics = RunningAverageDict()
            pixels_below_threshold_list=[]
            for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                # image_crop, depth_crop = data['image_crop'].cuda(), data['depth_crop'].cuda()
                image_raw, depth_raw, image_crop, depth_crop, box, rawhole_depth_crop, camera_matrix = \
                    data['image_raw'].cuda(), data['depth_raw'].cuda(), \
                    data['image_crop'].cuda(), data['depth_crop'].cuda(), \
                    data['box'].cuda(), data['rawhole_depth_crop'].cuda(), data['camera_matrix']
                camera_matrix = camera_matrix.squeeze().cpu().numpy()
                # depth_pred1, _ = perform_inference(image_raw)
                t0= time.time()
                depth_pred2, _ = perform_inference(image_crop)
                t1 = time.time()
                # depth_pred2_lr = nn.functional.interpolate(depth_pred2, (96,128), mode='bilinear', align_corners=False)
                # depth_pred2_lr = depth_pred2_lr / 10
                depth_pred2_lr = nn.functional.interpolate(depth_pred2, (384 // opt.scale, 512 // opt.scale),
                                                           mode='bilinear',
                                                           align_corners=False)
                depth_pred2_lr = depth_pred2_lr / 10

                # depth_pred1, hook_feats1 = perform_inference(image_raw)
                # depth_pred1 = depth_pred1 / 10
                # out = net((image_crop, depth_pred2_lr, depth_pred1, box))

                # out = net((image_crop, depth_pred2_lr)) # [0,1]
                # if i==1:
                #     for key in suft_hook_feats.keys():
                #
                #         img = suft_hook_feats[key][0, 1, :, :]
                #         img = img.cpu().numpy()
                #         img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-12)
                #         img = np.uint8(img * 255)
                #
                #         cv2.imwrite(key + '_0.png', img)
                #
                #         if key == 'suft_feat_out':
                #             img = suft_hook_feats[key][0, 0, :, :]
                #             img = img.cpu().numpy()
                #             img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-12)
                #             img = np.uint8(img * 255)
                #
                #             cv2.imwrite(key + '_1.png', img)
                t2 = time.time()
                out = net((image_crop, depth_pred2_lr))
                t3 = time.time()

                out = out.cpu()
                out = out.squeeze().numpy()
                # out = cv2.flip(out, 0)
                out=out[78:306, 104:408]

                N = compute_normals_from_depth_five_point(out,camera_matrix)

                # # hole *10*1000 raw *1000
                # depth_pred2 = depth_pred2.cpu()
                # depth_pred2 = depth_pred2.squeeze().numpy()
                # N = compute_normals_from_depth_five_point(depth_pred2/10,camera_matrix)

                rawhole_depth_crop = rawhole_depth_crop.cpu()
                rawhole_depth_crop = rawhole_depth_crop.squeeze().numpy()
                rawhole_depth_crop= rawhole_depth_crop[78:306, 104:408]
                t4=time.time()
                # rawhole_depth_crop = cv2.flip(rawhole_depth_crop, 0)

                # depth_image = create_depth_image(N, rawhole_depth_crop * 10, lambda_D=1.0, lambda_N=10.0, lambda_S=0.0)
                depth_image = create_depth_image(N, rawhole_depth_crop * 10, lambda_D=1.0, lambda_N=20, lambda_S=0.001)
                t5=time.time()
                # # depth_image = cv2.flip(depth_image, 0)
                # # rawhole_depth_crop = cv2.flip(rawhole_depth_crop, 0)
                # # out = cv2.flip(out, 0)
                # depth_image = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).cuda()
                #
                # pixels_below_threshold = np.sum(rawhole_depth_crop < 0.0001)
                # pixels_below_threshold_list.append(pixels_below_threshold)
                #
                # metrics.update(compute_metrics(depth_crop * 10, depth_image))# m
                #
                # path_output = r'C:\zj\localfusion\ffd\trainlog\outputnyu10'
                # os.makedirs(path_output, exist_ok=True)
                # path_save_pred = '{}/{:05d}.png'.format(path_output, i)
                # path_save_pseudo_depth = '{}/{:03d}.png'.format(path_output, i)
                # path_save_rgb = '{}/rgb{:03d}.png'.format(path_output, i)
                # path_save_raw_pseudo_depth = '{}/raw{:05d}.png'.format(path_output, i)
                # # Save results  (Save the output depth map)
                # # depth_image = depth_pred2*10
                # pred = depth_image.squeeze(dim=0).squeeze(dim=0).cpu().detach().numpy()
                # pseudo_depth = rawhole_depth_crop
                # # rawhole_depth_crop1000=rawhole_depth_crop*1000
                # # # cv2.imwrite(path_save_raw_pseudo_depth,  rawhole_depth_crop.astype(np.uint16))
                # # np.save(path_save_raw_pseudo_depth, rawhole_depth_crop1000.astype(np.uint16))
                # rgb = image_crop.squeeze(dim=0).permute(1, 2, 0).cpu().detach().numpy()
                # # pred = colorize(pred)
                # # pred = pred/1000
                # d_min = np.min(pred)
                # d_max = np.max(pred)
                #
                # depth_relative = (pred - d_min) / (d_max - d_min)
                # cmap = plt.cm.viridis
                #
                # im_color = 255 * cmap(depth_relative)[:, :, :3]
                # im = Image.fromarray(im_color.astype('uint8'))
                #
                # pseudo_depth = (pseudo_depth * 255).astype(np.uint8)
                # pseudo_depth = Image.fromarray(pseudo_depth)
                #
                # # pred = (pred * 25.5).astype(np.uint8)
                # # pred = Image.fromarray(pred)
                #
                # rgb = (rgb * 255).astype(np.uint8)
                # rgb = Image.fromarray(rgb)
                # rgb.save(path_save_rgb)
                # im.save(path_save_pred)
                # pseudo_depth.save(path_save_pseudo_depth)
                print(t1 - t0, t3 - t2, t5 - t4)
                # if i == 1:
                #     pred2 = depth_pred2
                #     pred2 = pred2.squeeze(dim=0).squeeze(dim=0).cpu().detach().numpy()
                #     pred2 = pred2
                #     pred2 = (pred2 * 25.5).astype(np.uint8)
                #     pred2 = Image.fromarray(pred2)
                #     pred2.save('pred2.png')
                #     pred0 = out * 10
                #     pred0 = (pred0 * 25.5).astype(np.uint8)
                #     pred0 = Image.fromarray(pred0)
                #     pred0.save('out.png')


            def r(m): return round(m, 3)


            metrics = {k: r(v) for k, v in metrics.get_value().items()}
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            average = sum(pixels_below_threshold_list) / len(pixels_below_threshold_list)
            # with open("textlog8rawwpseudo.txt", "a") as file:
            #     file.write(f"{current_time},{average}\n {metrics}\n")

        # print(f"{colors.fg.green}")
        # print(metrics)
        # print(f"{colors.reset}")

    release_network()
#
# {'a1': 0.996, 'a2': 0.999, 'a3': 0.999, 'abs_rel': 0.011, 'rmse': 0.126, 'log_10': 0.006, 'rmse_log': 0.12, 'silog': 12.023, 'sq_rel': 0.006}
#









# {'a1': 0.935, 'a2': 0.991, 'a3': 0.998, 'abs_rel': 0.083, 'rmse': 0.351, 'log_10': 0.038, 'rmse_log': 0.113, 'silog': 8.206, 'sq_rel': 0.043}
# {'a1': 0.945, 'a2': 0.993, 'a3': 0.999, 'abs_rel': 0.079, 'rmse': 0.33, 'log_10': 0.034, 'rmse_log': 0.103, 'silog': 8.183, 'sq_rel': 0.04}
# zoe-depth  0.946 0.994 0.999  0.082	0.294	0.035 log10

# 8{'a1': 0.952, 'a2': 0.994, 'a3': 0.999, 'abs_rel': 0.077, 'rmse': 0.322, 'log_10': 0.033, 'rmse_log': 0.101, 'silog': 8.011, 'sq_rel': 0.037}
# {'a1': 0.953, 'a2': 0.992, 'a3': 0.999, 'abs_rel': 0.076, 'rmse': 0.317, 'log_10': 0.033, 'rmse_log': 0.1, 'silog': 8.032, 'sq_rel': 0.037}

# 4{'a1': 0.949, 'a2': 0.993, 'a3': 0.999, 'abs_rel': 0.077, 'rmse': 0.318, 'log_10': 0.033, 'rmse_log': 0.101, 'silog': 8.055, 'sq_rel': 0.037}
# {'a1': 0.953, 'a2': 0.994, 'a3': 0.999, 'abs_rel': 0.077, 'rmse': 0.315, 'log_10': 0.033, 'rmse_log': 0.1, 'silog': 7.829, 'sq_rel': 0.038}
# fusion
# 4{'a1': 0.954, 'a2': 0.993, 'a3': 0.999, 'abs_rel': 0.075, 'rmse': 0.315, 'log_10': 0.032, 'rmse_log': 0.098, 'silog': 7.889, 'sq_rel': 0.037}
# 8{'a1': 0.952, 'a2': 0.993, 'a3': 0.998, 'abs_rel': 0.082, 'rmse': 0.328, 'log_10': 0.035, 'rmse_log': 0.105, 'silog': 8.533, 'sq_rel': 0.041}

#optimize-10
#raw{'a1': 0.999, 'a2': 1.0, 'a3': 1.0, 'abs_rel': 0.002, 'rmse': 0.038, 'log_10': 0.001, 'rmse_log': 0.014, 'silog': 1.38, 'sq_rel': 0.001}
#holeR{'a1': 0.5, 'a2': 0.5, 'a3': 0.5, 'abs_rel': 0.5, 'rmse': 1.307, 'log_10': 1.68, 'rmse_log': 3.873, 'silog': 18.14, 'sq_rel': 1.226}
#hole 0.33 -rmse0.067
# 0.7{'a1': 0.985, 'a2': 0.998, 'a3': 1.0, 'abs_rel': 0.025, 'rmse': 0.147, 'log_10': 0.01, 'rmse_log': 0.055, 'silog': 5.406, 'sq_rel': 0.01}
# 0.5{'a1': 0.994, 'a2': 0.999, 'a3': 1.0, 'abs_rel': 0.012, 'rmse': 0.096, 'log_10': 0.005, 'rmse_log': 0.033, 'silog': 3.333, 'sq_rel': 0.004}
# hole-0.5-500
#{'a1': 0.978, 'a2': 0.997, 'a3': 0.999, 'abs_rel': 0.047, 'rmse': 0.212, 'log_10': 0.02, 'rmse_log': 0.076, 'silog': 7.517, 'sq_rel': 0.017}
# hole-0.5-5000
#{'a1': 0.989, 'a2': 0.998, 'a3': 1.0, 'abs_rel': 0.023, 'rmse': 0.137, 'log_10': 0.01, 'rmse_log': 0.051, 'silog': 5.061, 'sq_rel': 0.008}

#optimize-449
#hole-0.5{'a1': 0.992, 'a2': 0.999, 'a3': 1.0, 'abs_rel': 0.016, 'rmse': 0.124, 'log_10': 0.007, 'rmse_log': 0.04, 'silog': 4.011, 'sq_rel': 0.006}
#hole-0.5-500{'a1': 0.953, 'a2': 0.993, 'a3': 0.998, 'abs_rel': 0.064, 'rmse': 0.285, 'log_10': 0.027, 'rmse_log': 0.095, 'silog': 9.356, 'sq_rel': 0.032}
#holeR{'a1': 0.491, 'a2': 0.491, 'a3': 0.491, 'abs_rel': 0.509, 'rmse': 1.621, 'log_10': 1.749, 'rmse_log': 4.033, 'silog': 17.724, 'sq_rel': 1.52}
#{'a1': 0.491, 'a2': 0.491, 'a3': 0.491, 'abs_rel': 0.509, 'rmse': 1.625, 'log_10': 1.749, 'rmse_log': 4.034, 'silog': 17.575, 'sq_rel': 1.525}
#0.33{'a1': 0.996, 'a2': 1.0, 'a3': 1.0, 'abs_rel': 0.008, 'rmse': 0.083, 'log_10': 0.003, 'rmse_log': 0.027, 'silog': 2.672, 'sq_rel': 0.003}
#0.33-500{'a1': 0.964, 'a2': 0.995, 'a3': 0.999, 'abs_rel': 0.053, 'rmse': 0.249, 'log_10': 0.022, 'rmse_log': 0.083, 'silog': 8.167, 'sq_rel': 0.024}

#all-4
#{'a1': 0.992, 'a2': 0.999, 'a3': 1.0, 'abs_rel': 0.014, 'rmse': 0.121, 'log_10': 0.006, 'rmse_log': 0.039, 'silog': 3.924, 'sq_rel': 0.006}
#500{'a1': 0.96, 'a2': 0.994, 'a3': 0.999, 'abs_rel': 0.056, 'rmse': 0.262, 'log_10': 0.023, 'rmse_log': 0.087, 'silog': 8.584, 'sq_rel': 0.027}
#holeR

#all-8
#{'a1': 0.992, 'a2': 0.999, 'a3': 1.0, 'abs_rel': 0.014, 'rmse': 0.12, 'log_10': 0.006, 'rmse_log': 0.039, 'silog': 3.889, 'sq_rel': 0.006}
#500{'a1': 0.96, 'a2': 0.994, 'a3': 0.999, 'abs_rel': 0.056, 'rmse': 0.262, 'log_10': 0.023, 'rmse_log': 0.087, 'silog': 8.616, 'sq_rel': 0.027}
#holeR{'a1': 0.493, 'a2': 0.493, 'a3': 0.493, 'abs_rel': 0.507, 'rmse': 1.631, 'log_10': 1.743, 'rmse_log': 4.019, 'silog': 17.826, 'sq_rel': 1.528}
# fusion-4
#0.33-500{'a1': 0.963, 'a2': 0.994, 'a3': 0.999, 'abs_rel': 0.054, 'rmse': 0.254, 'log_10': 0.022, 'rmse_log': 0.084, 'silog': 8.281, 'sq_rel': 0.025}
#{'a1': 0.992, 'a2': 0.999, 'a3': 1.0, 'abs_rel': 0.014, 'rmse': 0.122, 'log_10': 0.006, 'rmse_log': 0.04, 'silog': 3.977, 'sq_rel': 0.006}
# fusion-8
#0.33{'a1': 0.996, 'a2': 1.0, 'a3': 1.0, 'abs_rel': 0.008, 'rmse': 0.084, 'log_10': 0.003, 'rmse_log': 0.027, 'silog': 2.68, 'sq_rel': 0.003}
#0.33-500{'a1': 0.963, 'a2': 0.995, 'a3': 0.999, 'abs_rel': 0.053, 'rmse': 0.251, 'log_10': 0.022, 'rmse_log': 0.083, 'silog': 8.187, 'sq_rel': 0.025}
#0.5{'a1': 0.992, 'a2': 0.999, 'a3': 1.0, 'abs_rel': 0.014, 'rmse': 0.122, 'log_10': 0.006, 'rmse_log': 0.039, 'silog': 3.902, 'sq_rel': 0.006}