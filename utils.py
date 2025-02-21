import torch
import pytorch_ssim
from torch.autograd import Variable

def calc_rmse(a, b, minmax):
    # minmax = torch.from_numpy(minmax).cuda()
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    
    a = a*(minmax[0]-minmax[1]) + minmax[1]
    b = b*(minmax[0]-minmax[1]) + minmax[1]
    a = a * 100
    b = b * 100
    
    return torch.sqrt(torch.mean(torch.pow(a-b,2)))
def calc_ssim(gt, out, minmax):
    # minmax = torch.from_numpy(minmax).cuda()
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]


    gt = Variable(gt.unsqueeze(0).unsqueeze(0)).float()
    out = Variable(out.unsqueeze(0).unsqueeze(0))

    # gt = (gt -minmax[1])/(minmax[0]-minmax[1])
    # out = out * (minmax[0] - minmax[1]) + minmax[1]

    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    ssim_value = ssim_loss(gt, out)

    return ssim_value


def rgbdd_calc_rmse(gt, out, minmax):
    # minmax = torch.from_numpy(minmax).cuda()
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]
    non_zero_mask = (gt != 0)
    out = torch.where(non_zero_mask, out, torch.tensor(0.0, device=out.device))
    # gt = gt*(minmax[0]-minmax[1]) + minmax[1]
    out = out*(minmax[0]-minmax[1]) + minmax[1]
    gt = gt / 10.0
    out = out / 10.0
    
    return torch.sqrt(torch.mean(torch.pow(gt-out,2)))

def midd_calc_rmse(gt, out):
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]
    gt = gt * 255.0
    out = out * 255.0

    return torch.sqrt(torch.mean(torch.pow(gt-out,2)))


def rgbdd_calc_ssim(gt, out, minmax):
    # minmax = torch.from_numpy(minmax).cuda()
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]
    non_zero_mask = (gt != 0)
    out = torch.where(non_zero_mask, out, torch.tensor(0.0, device=out.device))

    gt = Variable(gt.unsqueeze(0).unsqueeze(0)).float()
    out = Variable(out.unsqueeze(0).unsqueeze(0))

    gt = (gt -minmax[1])/(minmax[0]-minmax[1])
    # out = out * (minmax[0] - minmax[1]) + minmax[1]

    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    ssim_value = ssim_loss(gt, out)

    return ssim_value

def midd_calc_ssim(gt, out):

    gt = Variable(gt[6:-6, 6:-6].unsqueeze(0).unsqueeze(0)).float()
    out = Variable(out[6:-6, 6:-6].unsqueeze(0).unsqueeze(0))

    # ssim_loss = pytorch_ssim.SSIM()
    # ssim_out = -ssim_loss(gt, out)
    # ssim_value = - ssim_out.item()
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    ssim_value = ssim_loss(gt, out)

    return ssim_value