import random
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

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

def get_patch(img, gt, patch_size=16):
    th, tw = img.shape[:2]
    patch_width = patch_size
    patch_height = int(patch_width / 4 * 3 )
    tx = random.randrange(0, (tw - patch_width))
    ty = random.randrange(0, (th - patch_height))
    patch_img = img[ty:ty + patch_height, tx:tx + patch_width, :]
    patch_gt = gt[ty:ty + patch_height, tx:tx + patch_width, :]
    box =(ty, ty + patch_height, tx, tx + patch_width)
    return patch_img, patch_gt,box
def get_patch2(img, gt, patch_size=16):
    th, tw = img.shape[:2]
    patch_width = patch_size
    patch_height = int(patch_width / 4 * 3) 
    resized_img = cv2.resize(img, (patch_width, patch_height))
    resized_gt = cv2.resize(gt, (patch_width, patch_height))
    box = (0, patch_height, 0, patch_width)
    return resized_img, resized_gt, box

def hole_image_jit_v2(image, width, height, nums):
    """Occlusion areas do not overlap"""
    image_height, image_width = image.shape[:2]
    rows, columns = (image_height - height + 1), (image_width - width + 1)
    optional_region = np.ones((rows * columns))
    masks = np.ones_like(image)

    for k in range(nums):
        # random select a region, (upper left corner)
        idx = np.random.choice(np.where(optional_region == 1)[0])

        # mask selected region
        x, y = idx // columns, idx % columns
        masks[x:x + width, y:y + height] = 0

        # set the nearby area unobstructed
        left_bound, right_bound = x - width + 1, x + width - 1
        upper_bound, bottom_bound = y - height + 1, y + height - 1
        for i in range(max(0, left_bound), min(right_bound, columns)):
            for j in range(max(0, upper_bound), min(bottom_bound, rows)):
                # x, y = j, i
                optional_region[j * columns + i] = 0
    return image * masks


class Spatter:
    def __init__(self,
                 threshold=True,
                 granularity=8,
                 percentile_void=0.05,#0.05
                 percentile_deform=0.02):
        self.threshold = threshold
        self.granularity = granularity
        self.percentile_deform = percentile_deform
        self.percentile_void = percentile_void

    def spatter(self, layer, mask, granularity=10, percentile=0.4):
        holes_mask = self.create_holes_mask(layer, granularity, percentile)

        res = layer.copy().squeeze()
        mask = mask.copy().squeeze()
        res[holes_mask] = 0
        mask[holes_mask] = 0

        return res, holes_mask
        # return res, mask

    def create_holes_mask(self, layer, granularity, percentile):
        gaussian_layer = np.random.uniform(size=layer.shape[1:])
        gaussian_layer = gaussian_filter(gaussian_layer, sigma=granularity)
        threshold = np.percentile(gaussian_layer.reshape([-1]), 100 * (1 - percentile))

        return gaussian_layer > threshold

    def __call__(self, sample):
        if self.threshold:
            pass

        raw_depth = sample['raw_depth'][np.newaxis, ...]
        _, mask = self.spatter(raw_depth,
                               raw_depth > 0,
                               granularity=self.granularity,
                               percentile=self.percentile_void)

        return mask

class MaskBlacks:
    def __init__(self, vmin=0, vmax=5):
        self.vmax = vmax
        self.vmin = vmin

    def __call__(self, sample):
        rgb = sample['rgb']

        masks = ((rgb >= self.vmin) & (rgb <= self.vmax/256)).sum(2)
        masks = masks == 3    # rgb channel

        return masks


class SegmentationHighLight:
    def __init__(self, T1=210):
        self.T1 = T1/256

    def calc_specular_mask(self, cE, cG, cB):
        h, w = cE.shape[:2]
        T1 = self.T1

        p95_cG = cG * 0.95
        p95_cE = cE * 0.95
        p95_cB = cB * 0.95

        rGE = p95_cG / (p95_cE + 1e-8)
        rBE = p95_cB / (p95_cE + 1e-8)
        img_new = np.zeros((h, w, 1), dtype=np.float32)

        mask1, mask2, mask3 = cG > (rGE * T1), cB > (rBE * T1), cE > T1
        mask = mask1 & mask2 & mask3
        img_new[mask] = 255

        return img_new

    def __call__(self, sample):
        rgb = sample['rgb']

        cR = rgb[:, :, 0]
        cG = rgb[:, :, 1]
        cB = rgb[:, :, 2]

        cE = 0.2989 * cR + 0.5870 * cG + 0.1140 * cB

        specular_mask = self.calc_specular_mask(cE, cG, cB)
        specular_mask = specular_mask.squeeze() / 255

        return specular_mask == 1


class CutOffBlackBorder:
    def __init__(self,
                 top_pixel_num=45,
                 bottom_pixel_num=15,
                 left_pixel_num=45,
                 right_pixel_num=40):
        self.top_pixel_num = top_pixel_num
        self.bottom_pixel_num = bottom_pixel_num
        self.left_pixel_num = left_pixel_num
        self.right_pixel_num = right_pixel_num

    def __call__(self, img):
        # np.ndarry
        croped_img = img[self.top_pixel_num:-self.bottom_pixel_num, self.left_pixel_num:-self.right_pixel_num]

        return croped_img
