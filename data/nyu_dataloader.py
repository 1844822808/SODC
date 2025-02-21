mport torch
import json
import cv2
from torchvision.transforms import transforms as T
import numpy as np
import os
import torchvision.transforms.functional as TF
from PIL import Image
import h5py
import random


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)


class NYUDataset(BaseDataset):
    def __init__(self, args, mode):
        super(NYUDataset, self).__init__()

        if mode != "train" and mode != "val" and mode != "test":
            raise NotImplementedError

        height, width = (240, 320)
        crop_size = (228, 304)
        # crop_size = (224, 224)

        self.height = height
        self.width = width
        self.crop_size = crop_size
        self.num_sample = args.num_sample
        self.data_dir = args.data_dir
        self.mode = mode
        self.augment = args.augment
        self.is_sparse = args.is_sparse

        self.K = torch.Tensor(
            [
                5.1885790117450188e02 / 2.0,
                5.1946961112127485e02 / 2.0,
                3.2558244941119034e02 / 2.0 - 8.0,
                2.5373616633400465e02 / 2.0 - 6.0,
            ]
        )

        with open(os.path.join(args.data_dir, "nyu.json")) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        path_file = os.path.join(self.data_dir, self.sample_list[idx]["filename"])

        f = h5py.File(path_file, "r")
        rgb_h5 = f["rgb"][:].transpose(1, 2, 0)
        dep_h5 = f["depth"][:]

        rgb = Image.fromarray(rgb_h5, mode="RGB")
        dep = Image.fromarray(dep_h5.astype("float32"), mode="F")

        if self.augment and self.mode == "train":
            _scale = np.random.uniform(1.0, 1.5)
            scale = int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)

            rgb = TF.rotate(rgb, angle=degree)
            dep = TF.rotate(dep, angle=degree)

            t_rgb = T.Compose(
                [
                    T.Resize(scale),
                    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    T.CenterCrop(self.crop_size),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

            t_dep = T.Compose(
                [
                    T.Resize(scale),
                    T.CenterCrop(self.crop_size),
                    self.ToNumpy(),
                    T.ToTensor(),
                ]
            )

            rgb = t_rgb(rgb)
            dep = t_dep(dep)
            dep = dep / _scale

        else:
            t_rgb = T.Compose(
                [
                    T.Resize(self.height),
                    T.CenterCrop(self.crop_size),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

            t_dep = T.Compose(
                [
                    T.Resize(self.height),
                    T.CenterCrop(self.crop_size),
                    self.ToNumpy(),
                    T.ToTensor(),
                ]
            )

            rgb = t_rgb(rgb)
            dep = t_dep(dep)
          
        depth_incomplete = self.get_sparse_depth(depth_incomplete, self.aaa)
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
        }
        return sample

      
        def get_sparse_depth(self, depth, num_sample):
        c, h, w = depth.shape

        assert c == 1

        # Pytorchv1.2+
        idx_nzz = torch.nonzero(depth.view(-1) > 0.0001, as_tuple=False)

        # idx_nzz = torch.where(depth.view(-1) > 0.0001)[0].view(-1, 1)

        num_idx = len(idx_nzz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nzz = idx_nzz[idx_sample[:]]

        mask = torch.zeros((c * h * w))
        mask[idx_nzz] = 1.0
        mask = mask.view(c, h, w)

        depth_sp = depth * mask.type_as(depth)

        return depth_sp
