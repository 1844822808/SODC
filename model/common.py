import random

# def arugment(img, gt, hflip=True, vflip=True):
#     if hflip and random.random() < 0.5:
#         img = img[:, ::-1, :]
#         gt = gt[:, ::-1, :]
#     if vflip and random.random() < 0.5:
#         img = img[::-1, :, :]
#         gt = gt[::-1, :, :]
#
#     return img, gt
def arugment(img,gt, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5

    if hflip:
        img = img[:, ::-1, :].copy()
        gt = gt[:, ::-1, :].copy()
    if vflip:
        img = img[::-1, :, :].copy()
        gt = gt[::-1, :, :].copy()

    return img, gt

# def get_patch(img, gt, patch_size=16):
#     th, tw = img.shape[:2]
#
#     # 计算补丁的长宽
#     patch_height = patch_size
#     patch_width = int(patch_height / 4 * 3 )  # 4:3 的长宽比
#
#     # 随机选择补丁的左上角坐标
#     tx = random.randrange(0, (tw - patch_width))
#     ty = random.randrange(0, (th - patch_height))
#
#     # 切割图像和GT以获取补丁
#     patch_img = img[ty:ty + patch_height, tx:tx + patch_width, :]
#     patch_gt = gt[ty:ty + patch_height, tx:tx + patch_width, :]
#     box =(ty, ty + patch_height, tx, tx + patch_width)
#     return patch_img, patch_gt,box

def get_patch(img, gt, patch_size=16):
    th, tw = img.shape[:2]

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[ty:ty + tp, tx:tx + tp, :], gt[ty:ty + tp, tx:tx + tp, :]