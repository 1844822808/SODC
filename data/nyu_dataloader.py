import os
import torch
# import transformers.image_transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.common1 import *
import os
import random
from torchvision import transforms
ROOT = r"C:\zj\ffc-HRDE\data\NYU_v2"

def read_image_path(root=r"C:\Users\22476\Desktop\VOC_format_dataset\ImageSet\train.txt"):
    image = np.loadtxt(root, dtype=str)
    n =len(image)
    color, depth, rawdepth= [None]*n, [None]*n, [None] * n
    for i,fname in enumerate(image):
        color[i] = r"C:\zj\ffc-HRDE\data\NYU_v2\images\%s.jpg" % fname
        depth[i] = r"C:\zj\ffc-HRDE\data\NYU_v2\nyu_depths_not_max_normalized\%s.npy" % fname
        rawdepth[i] = r"C:\zj\ffc-HRDE\data\NYU_v2\nyu_rawDepths_not_max_normalized\%s.npy" % fname

    return color, depth, rawdepth

class NYuV2Test(Dataset):
    def __init__(
            self, root_dir=ROOT, scale=8, aaa = 1.0, train=True,transform=None,):
        self.transform = transform
        self.scale = scale
        self.train = train
        if train:
            split = "train"
        else:
            split = "test"
        self.aaa = aaa
        self.data_root = os.path.join(root_dir, "data")
        self.split_file = os.path.join(root_dir, split + ".txt")

        color_list, depth_list, rawdepth_list = read_image_path(root=self.split_file)
        self.color_name = color_list
        self.depth_name = depth_list
        self.rawdepth_name = rawdepth_list

    def __len__(self):
        return len(self.depth_name)

    def __getitem__(self, index):
        image = np.array(Image.open(self.color_name[index]).convert('RGB')).astype(np.float32)

        image_max = np.max(image)
        image_min = np.min(image)
        image_raw = (image - image_min) / (image_max - image_min)
        depth_raw = np.array(np.load(self.depth_name[index])).astype(np.float32)
        depth_raw = np.flip(depth_raw, axis=1)
        rawhole_depth = np.flip(np.array(np.load(self.rawdepth_name[index])).astype(np.float32), axis=1).copy()

        # crop_transform=transforms.Compose([
        #     transforms.center_crop((448,608))
        # ])
        image_raw =image_raw[12:-12,16:-16,:]
        rawhole_depth = rawhole_depth[12:-12,16:-16]
        depth_raw = depth_raw[12:-12,16:-16]
        th, tw = image.shape[:2]
        depth_raw[depth_raw > 10] = 10
        depth_raw = depth_raw/10

        rawhole_depth[rawhole_depth > 10] = 10
        rawhole_depth = rawhole_depth/10

        image, depth0, box = get_patch2(img=image_raw, gt=np.expand_dims(depth_raw, 2), patch_size=512)
        # if self.train:
        #     image, depth = arugment(img=image, gt=depth)

        crop_y1, crop_y2, crop_x1, crop_x2 = box
        height = 480
        width = 640
        bbox_roi = torch.tensor([crop_x1 / width * 512, crop_y1 / height * 384,
                                 crop_x2 / width * 512, crop_y2 / height * 384]).float()
        # rawhole_depth_crop = rawhole_depth[crop_y1:crop_y2, crop_x1:crop_x2]
        rawhole_depth_crop = cv2.resize(rawhole_depth, (512, 384))
        zero_positions = np.where(rawhole_depth_crop == 0)

        # -------------- perform pseudo operations -------------- #

        depth = depth0
        depth[zero_positions] = 0
        masks = []
        pseudo_sample = {'rgb': image, 'raw_depth': np.squeeze(depth)}
        prop = np.random.uniform(0.0, 1.0)
        if prop > 0.3:
            t = SegmentationHighLight()
            masks.append(t(sample=pseudo_sample))

        # prop = np.random.uniform(0.0, 1.0)
        # if prop > 0.3:
        t = Spatter()
        masks.append(t(sample=pseudo_sample))

        prop = np.random.uniform(0.0, 1.0)
        if prop > 0.3:
            t = MaskBlacks()
            masks.append(t(sample=pseudo_sample))

        # combine all pseudo masks
        pseudo_maks = np.zeros_like(np.squeeze(depth), dtype=bool)

        for m in masks:
            pseudo_maks |= m

        depth_incomplete = np.squeeze(depth).copy()
        depth_incomplete[pseudo_maks] = 0.0

        # Clip 25-30% of depth values farther than 8

        clip_percentage = random.uniform(0.25, 0.3)
        depth_incomplete = self.clip_depth(depth_incomplete, clip_percentage)
        # Add Gaussian noise to 5-10% of valid pixels
        # noise_percentage = random.uniform(0.05, 0.1)
        # depth_incomplete = self.add_gaussian_noise(depth_incomplete, noise_percentage)



        if self.transform:
            image_raw = self.transform(image_raw).float()
            depth_raw = np.expand_dims(depth_raw, 2)
            depth_raw = self.transform(depth_raw).float()
            #
            depth_incomplete = np.expand_dims(depth_incomplete, 2)
            depth_incomplete = self.transform(depth_incomplete).float()

            rawhole_depth_crop = np.expand_dims(rawhole_depth_crop, 2)
            rawhole_depth_crop = self.transform(rawhole_depth_crop).float()
            image = self.transform(image).float()
            depth = self.transform(depth0).float()
        depth_incomplete = self.get_sparse_depth(depth_incomplete, self.aaa)
        # nonzero_indices = torch.nonzero(depth_incomplete)
        # num_nonzero = nonzero_indices.size(0)
        camera_matrix = np.array([[5.8262448167737955e+02, 0.0, 3.1304475870804731e+02],
                                  [0.0, 5.8269103270988637e+02, 2.3844389626620386e+02],
                                  [0.0, 0.0, 1.0]])
        sample = {
            'image_raw': image_raw,
            'depth_raw': depth_raw,
            'image_crop': image,
            'depth_crop': depth,
            'box': bbox_roi,
            'rawhole_depth_crop': depth_incomplete,
            'camera_matrix': camera_matrix
        }#depth_incomplete   rawhole_depth_crop
        return sample

    def get_sparse_depth(self, depth,aaa):
        c, h, w = depth.shape

        assert c == 1
        num_sample = round(aaa*h*w)
        # Pytorchv1.2+
        idx_nzz = torch.nonzero(depth.view(-1) > 0.0001, as_tuple=False)

        # idx_nzz = torch.where(depth.view(-1) > 0.0001)[0].view(-1, 1)

        num_idx = len(idx_nzz)
        idx_sample = torch.randperm(num_idx)[:min(num_sample, num_idx)]

        idx_nzz = idx_nzz[idx_sample[:]]

        mask = torch.zeros((c * h * w))
        mask[idx_nzz] = 1.0
        mask = mask.view(c, h, w)

        depth_sp = depth * mask.type_as(depth)

        return depth_sp

    def add_gaussian_noise(self, depth_map, percentage=0.05):

        valid_pixels = depth_map > 0
        num_valid_pixels = np.sum(valid_pixels)
        num_pixels_to_modify = int(num_valid_pixels * percentage)

        # Randomly select pixels to modify
        flat_indices = np.argwhere(valid_pixels.ravel()).ravel()
        random_indices = np.random.choice(flat_indices, size=num_pixels_to_modify, replace=False)

        # Add Gaussian noise to selected pixels
        noise = np.random.normal(0, 0.1, size=num_pixels_to_modify)
        depth_map.flat[random_indices] += noise

        return depth_map

    def clip_depth(self, depth_map, clip_percentage=0.25, max_depth=0.95):
        """
        Randomly clip a percentage of depth values farther than a specified maximum depth.

        Args:
            depth_map (numpy.ndarray): Input depth map.
            clip_percentage (float, optional): Percentage of depth values to clip. Default is 0.25 (25%).
            max_depth (float, optional): Maximum depth value. Values farther than this will be clipped. Default is 8.0.

        Returns:
            numpy.ndarray: Depth map with depth values clipped.
        """
        valid_pixels = depth_map > 0
        num_valid_pixels = np.sum(valid_pixels)
        num_pixels_to_clip = int(num_valid_pixels * clip_percentage)

        # Flatten and get indices of pixels to clip
        depth_flat = depth_map.ravel()
        valid_flat = valid_pixels.ravel()
        far_pixels = (depth_flat > max_depth) & valid_flat
        flat_indices = np.argwhere(far_pixels).ravel()

        # Randomly select pixels to clip
        random_indices = np.random.choice(flat_indices, size=min(num_pixels_to_clip, len(flat_indices)), replace=False)

        # Clip selected pixels
        depth_map.flat[random_indices] = max_depth

        return depth_map
