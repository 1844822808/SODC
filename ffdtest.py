import logging
from datetime import datetime
import argparse
from zoedepth.trainers.loss import SILogLoss
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import torch
from zoedepth.utils.misc import save_raw_16bit
from zoedepth.utils.misc import colorize
from PIL import Image
from model_zoo.FFDGNet import *
from data.nyu_dataloader import *
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from zoedepth.utils.misc import *
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
                        default=r"C:\zj\localfusion\ffd\trainlog\20240531094836-lr_0.0001-s_4-NYU_v2-b_1\nyu-1de-4.pth", help="checkpoint")
    opt = parser.parse_args()

    dataset_name = opt.root_dir.split('\\')[-1]
    # print(opt)




    net = FFDG_network(num_feats=opt.num_feats, kernel_size=3, scale=opt.scale).cuda()
    net.load_state_dict(torch.load(opt.model_dir, map_location='cuda:0'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)


    data_transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = NYuV2Test(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    data_num = len(test_dataloader)
    create_zoenetwork()
    with torch.no_grad():
        net.eval()
        metrics = RunningAverageDict()
        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

            image_crop, depth_crop = data['image_crop'].cuda(), data['depth_crop'].cuda()

            depth_pred2, _ = perform_inference(image_crop)
            depth_pred2_lr = nn.functional.interpolate(depth_pred2, (96,128), mode='bilinear', align_corners=False)
            depth_pred2_lr = depth_pred2_lr / 10
            out = net((image_crop, depth_pred2_lr))


            metrics.update(compute_metrics(depth_crop*10,depth_pred2))

        def r(m): return round(m, 3)
        metrics = {k: r(v) for k, v in metrics.get_value().items()}
        print(metrics['rmse'])
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    release_network()

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