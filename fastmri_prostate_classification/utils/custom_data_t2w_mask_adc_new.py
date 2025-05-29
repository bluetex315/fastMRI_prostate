# import modules
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset
from torch.autograd import Variable
from torchvision.ops import sigmoid_focal_loss
import random
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
import monai
from monai.transforms import (
    Compose, LoadImaged, Lambdad, EnsureChannelFirstd, Orientationd, RandAffined, RandFlipd, RandRotated, ScaleIntensityRanged, ScaleIntensityRangePercentilesd, CenterSpatialCropd, NormalizeIntensityd, ToTensord
)
from monai.data import Dataset, CacheDataset, DataLoader

from torchvision.utils import save_image as tv_save_image, make_grid
import nibabel as nib

from utils.utils import parse_3d_volumes, parse_syn_slices, split_dset_by_patient, get_patient_splits

# Define a custom wrapper to apply intensity scaling with probability
def probabilistic_intensity_scaling(image, probability=0.5):
    if random.random() < probability:
        # Define random lower and upper bounds for intensity scaling
        lower_bound = np.percentile(image, randrange(10))
        upper_bound = np.percentile(image, randrange(90,100))
        
        # Apply ScaleIntensityRange with the defined bounds
        # TODO: need to figure out what is b_min, b_max,
        transform = ScaleIntensityRange(
            a_min=lower_bound,
            a_max=upper_bound,
            b_min=0,
            b_max=255,
            clip=True
        )
        
        return transform(image)
    else:
        return image

# Adding the custom scaling function to the pipeline
class ProbabilisticScaleIntensity:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image):
        return probabilistic_intensity_scaling(image, self.probability)


def save_images_from_loader(loader, save_dir, split,
                            n_images=16, saveims_format=('png',)):
    """
    Save up to n_images from the first batch of loader into save_dir.
    save_formats can be:
      - 'png'         : a single 4x4 grid image
      - 'nifti','nii' : individual .nii.gz slices
      - or a list/tuple containing any combination, e.g. ['png','nifti']
    """
    os.makedirs(save_dir, exist_ok=True)
    try:
        batch = next(iter(loader))
    except StopIteration:
        print(f"[save_images] No data in {split} loader.")
        return

    images = batch['image']
    labels = batch['class_label']
    patient_ids = batch['patient_id']
    slice_ids = batch['slice_idx']

    images = images.detach().cpu()[:n_images]

    if images.ndim == 4 and images.size(1) > 1:
        images = images[:, :1, ...]  # keep only channel 0

    # normalize formats arg to list
    if isinstance(saveims_format, str):
        formats = [saveims_format]
    else:
        formats = list(saveims_format)

    for fmt in formats:
        if fmt == 'png':
            grid = make_grid(images, nrow=4, padding=2, normalize=True, scale_each=True)
            out_path = os.path.join(save_dir, f"{split}_grid.png")
            tv_save_image(grid, out_path)
            print(f"[save_images] Saved PNG grid → {out_path}")

        elif fmt in ('nifti','nii','nii.gz'):
            for img, pid, sid in zip(images, patient_ids, slice_ids):
                img_np = img.squeeze(0).numpy()
                fname = f"{split}_pid{pid}_slice{sid}.nii.gz"
                out_path = os.path.join(save_dir, fname)
                nii = nib.Nifti1Image(img_np, affine=np.eye(4))
                nib.save(nii, out_path)
                print(f"[save_images] Saved NIfTI → {out_path}")

        elif fmt == 'npz':
            for img, pid, sid in zip(images, patient_ids, slice_ids):
                arr = img.squeeze(0).numpy()
                fname = f"{split}_pid{pid}_slice{sid}.npz"
                out_path = os.path.join(save_dir, fname)
                np.savez_compressed(out_path, image=arr)
                print(f"[save_images] Saved NPZ → {out_path}")

        else:
            raise ValueError(f"Unsupported format '{fmt}' in saveims_format")


def load_data(config, datapath, labelpath, gland_maskpath, norm_type, augment, saveims, saveims_format, rundir, split, rank=0, world_size=1):
    """
    Load FastMRI prostate data and create DataLoader instances for training, validation, and testing.

    Parameters:
    - datapath (str): Path to the folder containing patient data subfolders.
    - labelpath (str): Path to the folder containing label files.
    - norm_type (int): Integer representing the chosen normalization scheme.
    - augment (bool): Flag indicating whether to apply data augmentation.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - valid_loader (DataLoader): DataLoader for the validation set.
    - test_loader (DataLoader): DataLoader for the test set.
    """
    
    print(f"\n\n-------------------------- START Loading {split} Dataset --------------------------")
    print(f"LOADING ADC        --> {config['concat_adc']}")
    print(f"LOADING gland_mask --> {config['concat_mask']}")
        
    # load data
    dset_dict = {}
    if config['data']['real_datapath'] is not None:
        img_paths = [
            os.path.join(root, file)
            for root, _, files in sorted(os.walk(config['data']['real_datapath']))
            for file in files if file.endswith("T2W.nii.gz")
        ]
        dset_dict["image"] = img_paths
    
    
    if config['data']['atlas_CGPZ_path'] is not None:
        seg_types = os.listdir(config['data']['atlas_CGPZ_path'])
        seg_paths = {
            seg_type: [
                os.path.join(root, file)
                for root, _, files in sorted(os.walk(os.path.join(config['data']['atlas_CGPZ_path'], seg_type)))
                for file in files if file.endswith(".nii.gz")
            ]
            for seg_type in seg_types
        }
        for seg_type in seg_types:
            print(f"[main] Using {seg_type} for boundary slice removal")
            seg_key = 'seg_' + seg_type
            dset_dict.update({seg_key: seg_paths[seg_type]})

    print(f"[main] {dset_dict.keys()}")

    # 1) get the exact same (as diffusion) patient‐level splits using real data path
    train_ids, val_ids, test_ids = get_patient_splits(
        datapath=config['data']['real_datapath'],
        test_size=0.3,
        val_size=0.5,
        seed=config['seed'],
        exclude_ids=["223"]   # As patient_223 does not have correspoding segmentation, so remove it
    )

    print(f"[load_data] splits seed {config['seed']}")
    print("[load_data] train ids\n", train_ids)
    print("[load_data] val ids\n", val_ids)
    print("[load_data] test ids\n", test_ids)
    print()
    print(f"[load_data] Train: {len(train_ids)} patients")
    print(f"[load_data] val:   {len(val_ids)} patients")
    print(f"[load_data] test:  {len(test_ids)} patients")
    print()

    # Create datasets
    dset_dict_train = split_dset_by_patient(dset_dict, train_ids)
    dset_dict_val = split_dset_by_patient(dset_dict, val_ids)
    dset_dict_test = split_dset_by_patient(dset_dict, test_ids)
    
    # for transforms
    tot_keys, norm_keys = ['image'], ['image']          # keep a tot_key and norm_key list to keep track of modalities

    # need to went through the diffusion eval transform to make sure the transform stays the same
    # for using syn data in val/test, need to go through diffusion_pre_transforms because loaded from the original file from parse_3d_volumes
    diffusion_pre_transforms = Compose([
        # Add a channel dimension to 'image' and 'segm'
        EnsureChannelFirstd(keys=tot_keys, channel_dim='no_channel'),
        Orientationd(keys=tot_keys, axcodes='LAS'),
        # center spatial crop
        CenterSpatialCropd(keys=tot_keys, roi_size=(128, 128)),
        # scale to [0, 1]
        ScaleIntensityRangePercentilesd(
            keys=norm_keys,
            lower=2.5, upper=97.5, 
            b_min=0.0, b_max=1.0, 
            clip=True, relative=False, 
            channel_wise=False
        ),
        # normalize
        NormalizeIntensityd(
            keys=norm_keys,
            subtrahend=0.5,
            divisor=0.5
        ),
    ])

    # training transform for the downstream classification
    cls_train_transforms = Compose([
        # EnsureChannelFirstd(keys=tot_keys, channel_dim='no_channel'),
        RandAffined(
            keys=tot_keys,
            prob=0.5,
            translate_range=(0, 16, 16)
        ),
        # ProbabilisticScaleIntensity(probability=0.5),
        RandFlipd(
            keys=tot_keys,
            prob=0.5, 
            spatial_axis=1
        ),
        RandRotated(
            keys=tot_keys,
            range_x=12,   # Degrees of rotation for the x-axis (between -12 and 12)
            range_y=12,   # Degrees of rotation for the y-axis (between -12 and 12)
            range_z=0.0,  # No rotation for the z-axis
            prob=0.5,     # Ensure that the rotation is always applied
            keep_size=True, # Keep the same image size after rotation (reshape=False equivalent)
            padding_mode="zeros"
        ),
        ToTensord(keys=tot_keys),
    ])



    syn_slices_train_transforms = Compose([
        # Add a channel dimension to 'image' and 'segm'
        EnsureChannelFirstd(keys=tot_keys, channel_dim='no_channel'),
        *cls_train_transforms.transforms,           # cls_train_transform (augmentation for cls) + ToTensor
    ])

    # if not using synthetic, then having diffusion transform, loading from original 3D data, need to match diffusion pipeline
    real_slices_train_transforms = Compose([
        *diffusion_pre_transforms.transforms,       # diffusion pipeline
        *cls_train_transforms.transforms,           # cls_train_transform (augmentation for cls) + ToTensor
    ])

    # eval transform for the downstream classification
    cls_eval_transforms = Compose([
        # EnsureChannelFirstd(keys=tot_keys, channel_dim='no_channel'),
        ToTensord(keys=tot_keys),
    ])

    # eval transform always the same regardless of whether using synthetic, because always loading from 3D data, need to match diffusion pipeline
    eval_transforms = Compose([
        *diffusion_pre_transforms.transforms,       # diffusion pipeline
        *cls_eval_transforms.transforms,            # cls_eval_transform + ToTensor
    ])


    if split == "train":
        if config['data']['use_synthetic_data']:
            syn_slices_dset_list_train = parse_syn_slices(train_ids, config['data']['syn_datapath'], split='train')
            syn_train_dataset = monai.data.Dataset(syn_slices_dset_list_train, transform=syn_slices_train_transforms)

            if config['data']['combine_real_data']:
                real_slices_dset_list_train = parse_3d_volumes(dset_dict_train, seg_type, label_csv_file=config['data']['label_csv_dir'])
                real_train_dataset = monai.data.Dataset(real_slices_dset_list_train, transform=real_slices_train_transforms)
                
                train_dataset = ConcatDataset([syn_train_dataset, real_train_dataset])

            else:
                train_dataset = syn_train_dataset
            
        else:       # using only real data extracted from 3D volume
            real_slices_dset_list_train = parse_3d_volumes(dset_dict_train, seg_type, label_csv_file=config['data']['label_csv_dir'])
            real_train_dataset = monai.data.Dataset(real_slices_dset_list_train, transform=real_slices_train_transforms)
            
            train_dataset = real_train_dataset
        
        # validation set will be all real images regardless
        slices_dset_list_val = parse_3d_volumes(dset_dict_val, seg_type, label_csv_file=config['data']['label_csv_dir'])
        val_dataset = monai.data.Dataset(slices_dset_list_val, transform=eval_transforms)

        train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, num_workers=0, shuffle=False)

        # optionally save some samples
        if saveims and rank == 0:
            base_dir = os.path.join(rundir, "sample_images")
            save_images_from_loader(train_loader, os.path.join(base_dir, "train"), "train", n_images=16, saveims_format=saveims_format)
            save_images_from_loader(val_loader, os.path.join(base_dir, "val"), "val", n_images=16, saveims_format=saveims_format)
        
        print(f"[load_data] train:{len(train_dataset)}, val:{len(val_dataset)}")
        return train_loader, val_loader, None

    elif split == 'eval':
        slices_dset_list_test = parse_3d_volumes(dset_dict_test, seg_type, label_csv_file=config['data']['label_csv_dir'])
        test_dataset = monai.data.Dataset(slices_dset_list_test, transform=eval_transforms)
        test_loader = DataLoader(test_dataset, batch_size=128, num_workers=0, shuffle=False)
        
        if saveims and rank == 0:
            base_dir = os.path.join(rundir, "sample_images")
            save_images_from_loader(test_loader, os.path.join(base_dir, "test"), "test", n_images=16, saveims_format=saveims_format)
        
        print(f"[load_data] test:{len(test_dataset)}")
        return None, None, test_loader
