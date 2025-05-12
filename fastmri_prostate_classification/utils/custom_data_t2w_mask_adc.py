# import modules
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
from torchvision.ops import sigmoid_focal_loss
import random
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, LoadImaged, Lambdad, EnsureChannelFirstd, RandAffined, RandFlipd, RandRotated, ScaleIntensityRanged, CenterSpatialCropd, NormalizeIntensityd, ResampleToMatchd, EnsureChannelFirstd
)
from monai.data import Dataset, CacheDataset, DataLoader
import re

from torchvision.utils import save_image as tv_save_image, make_grid
import nibabel as nib

from utils.utils import parse_3d_volumes, split_dset_by_patient, get_patient_splits


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


class FakeFastMRIDataset(data.Dataset):
    """
    Custom MONAI dictionary-based dataset for loading FastMRI prostate data at the slice level.

    Parameters:
    - datapath (str): Path to the folder containing patient data subfolders.
    - labelpath (str): Path to the folder containing label files.
    - norm_type (int): Integer representing the chosen normalization scheme.
    - augment (bool): Flag indicating whether to apply data augmentation.
    - split (str): One of 'train', 'val', or 'test' indicating the dataset split.
    """
    def __init__(self, config, datapath, labelpath, gland_maskpath, norm_type, augment, split):
        super().__init__()

        self.config = config
        self.datapath = datapath
        self.labelpath = labelpath
        self.gland_maskpath = gland_maskpath
        self.norm_type = norm_type
        self.augment = augment
        self.split = split
        
        print(f"\n\n-------------------------- START Loading {self.split} Dataset --------------------------")
        print(f"LOADING ADC        --> {self.config['concat_adc']}")
        print(f"LOADING gland_mask --> {self.config['concat_mask']}")
        
        # 1) get the exact same patient‐level splits
        self.train_ids, self.val_ids, self.test_ids = get_patient_splits(
            datapath=config['data']['real_datapath'],
            test_size=0.3,
            val_size=0.5,
            seed=config['seed'],
            exclude_ids=["223"]   # As patient_223 does not have correspoding segmentation, so remove it
        )

        print(f"[main] splits seed{config.get('seed', 42)}")
        print("[main] train ids\n", self.train_ids)
        print("[main] val ids\n", self.val_ids)
        print("[main] test ids\n", self.test_ids)
        print()
        print(f"Train: {len(self.train_ids)} patients")
        print(f"val:   {len(self.val_ids)} patients")
        print(f"test:  {len(self.test_ids)} patients")
        print()

        # # split off 70% train, 30% temp
        # train_ids, temp_ids = train_test_split(
        #     patient_ids,
        #     train_size=0.7,
        #     random_state=self.config['seed'],
        #     shuffle=True
        # )

        # # then split that 30% into equal val/test (0.15 each of original)
        # val_ids, test_ids = train_test_split(
        #     temp_ids,
        #     test_size=0.5,   # half of temp_ids → 0.15 of total
        #     random_state=self.config['seed'],
        #     shuffle=True
        # )

        if self.split == "train":
            patient_ids = self.train_ids
        elif self.split == "val":
            patient_ids = self.val_ids
        else:
            patient_ids = self.test_ids

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

        dset_dict_train = split_dset_by_patient(dset_dict, self.train_ids)
        dset_dict_val = split_dset_by_patient(dset_dict, self.val_ids)
        dset_dict_test = split_dset_by_patient(dset_dict, self.test_ids)

        if config['data']['use_synthetic_data']:
            pass
        else:
            slices_dset_list_train = parse_3d_volumes(dset_dict_train, seg_type, label_csv_file=config['data']['label_csv_dir'])
            slices_dset_list_val = parse_3d_volumes(dset_dict_val, seg_type, label_csv_file=config['data']['label_csv_dir'])
            slices_dset_list_test = parse_3d_volumes(dset_dict_test, seg_type, label_csv_file=config['data']['label_csv_dir'])

        # pattern = re.compile(
        #     r'(?P<epoch>\d+)_'           
        #     r'(?P<patient>\d+)_slice'    
        #     r'(?P<slice>\d+)_'           
        #     r'(?P<kind>class|fakeclass)'
        #     r'(?P<label>\d+)_'           
        #     r'(?P<suffix>orig|syn)\.nii\.gz'
        # )     # naming pattern of fake images
        
        # rows = []
        # NUM_CLASSES = 5

        # # 2) for each patient, list *only* the real (orig) class files first
        # for pt_id in sorted(patient_ids):
        #     pt_folder = os.path.join(datapath, f"patient_{pt_id}")
        #     if not os.path.isdir(pt_folder):
        #         continue

        # patient_dirs = [d for d in os.listdir(self.datapath) if d.startswith("patient_")]         # filter down to only the patient directories

        # Extract the numeric ID from each name
        patient_ids = []
        for d in patient_dirs:
            match = re.match(r"patient_(\d+)", d)
            if match:
                patient_ids.append(match.group(1))

        for pt_id in patient_ids:       
            pt_folder = os.path.join(datapath, "patient_"+pt_id)
            if not os.path.isdir(pt_folder):
                raise FileNotFoundError(f"Patient folder not found: {pt_folder}")
            
            grouped = {}
            for path in os.listdir(pt_folder):      # iterate through patient folder, e.g `patient_078`
                m = pattern.match(path)
                if not m:
                    continue

                slice_idx  = int(m.group('slice'))   # e.g. 1
                kind       = m.group('kind')         # "class" or "fakeclass"
                label      = int(m.group('label'))   # the numeric class
                suffix     = m.group('suffix')       # "orig" or "syn"
                fullpath   = os.path.join(pt_folder, path)
                
                # group key: one entry per (patient, slice)
                key = (pt_id, slice_idx)
                if key not in grouped:
                    grouped[key] = {
                        "patient_id":   pt_id,
                        "slice":        slice_idx,
                        "orig":         None,
                        "fake_classes": {},
                    }

                entry = grouped[key]
                if kind == "class" and suffix == "orig":
                    entry["orig"]  = fullpath
                    entry["label"] = label
                elif kind == "fakeclass" and suffix == "syn":
                    entry.setdefault("fake_classes", {})[label] = fullpath

            # turn each grouped entry into a flat row
            for (pid, slice_idx), entry in grouped.items():
                row = {
                    "patient_id": pid,
                    "slice_id":   slice_idx,
                    "orig":       entry["orig"],
                    "label":      entry["label"],
                }
                # now add fake_0…fake_4 (or whatever labels you have)
                fake_dict = entry.get("fake_classes", {})
                for lbl in range(5):  
                    row[f"fake_{lbl}"] = fake_dict.get(lbl, None)

                rows.append(row)
                
        self.df = pd.DataFrame(rows).sort_values(["patient_id","slice_id"], ignore_index=True)
        print(f"dataset line445 {self.split}dataset")
        print(self.df)
        print()

        print(f"dataset line446 {self.split}dataset real PI-RADS distribution")
        print(self.df['label'].value_counts().to_dict())

        # compute class-level weights for real data (weights computed by TRAINING only)
        if self.split == 'train':
            counts = self.df['label'].value_counts().sort_index().values.astype(np.float32)
            inv_counts = 1.0 / (counts + 1e-6)
            norm_weights = inv_counts / inv_counts.sum() * len(counts)
            self.class_weights = torch.tensor(norm_weights, dtype=torch.float)
            self.config['data']['train_class_weights'] = self.class_weights.tolist()
        else:
            w = self.config['data'].get('train_class_weights', None)
            if w is None:
                raise ValueError('train_class_weights must be set in config for val/test splits')
            self.class_weights = torch.tensor(w, dtype=torch.float)
        
        self.records = self.flatten_df()

        print(f"-------------------------- FINISH defining/loading {self.split} Dataset --------------------------\n\n")

        load_keys = ['image']

        # Define MONAI transforms
        if self.split == 'train': 
            
            self.transforms = Compose([
                LoadImaged(keys=load_keys),
                EnsureChannelFirstd(keys=load_keys),
                RandAffined(
                    keys=load_keys,
                    prob=0.5,
                    translate_range=(0, 16, 16)
                ),
                # ProbabilisticScaleIntensity(probability=0.5),
                RandFlipd(
                    keys=load_keys,
                    prob=0.5, 
                    spatial_axis=1
                ),
                RandRotated(
                    keys=load_keys,
                    range_x=12,   # Degrees of rotation for the x-axis (between -12 and 12)
                    range_y=12,   # Degrees of rotation for the y-axis (between -12 and 12)
                    range_z=0.0,  # No rotation for the z-axis
                    prob=0.5,     # Ensure that the rotation is always applied
                    keep_size=True, # Keep the same image size after rotation (reshape=False equivalent)
                    padding_mode="zeros"
                ),
                # CenterSpatialCropd(keys=load_keys, roi_size=(224, 224)),
                NormalizeIntensityd(keys=load_keys)
            ])

        else:
            self.transforms = Compose([
                LoadImaged(keys=load_keys),
                EnsureChannelFirstd(keys=load_keys),
                # CenterSpatialCropd(keys=load_keys, roi_size=(224, 224)),
                NormalizeIntensityd(keys=load_keys)
            ])

    def flatten_df(self):
        """
        Flatten each row into a list of dicts:
        - one for the original (paths in `orig`, label in `label`)
        - one for each fake_X path
        """
        records = []
        for _, row in self.df.iterrows():
            if self.split == "train" and self.config['data']['use_synthetic_data']:       # only append fake images/labels when training AND use_synthetic_data. For val/test, use true images/labels
                # synthetic slices
                for fake_lbl in range(5):
                    path = row[f"fake_{fake_lbl}"]
                    if path is not None:
                        records.append({
                            "patient_id":      row['patient_id'],
                            "slice_id":        row['slice_id'],
                            "orig_image_path": row["orig"],
                            "orig_label":      int(row["label"]),
                            "image":           path,        # fake_image_path --> image
                            "label":           fake_lbl     # fake_label --> label
                        })
            else:
                records.append({
                    "patient_id":      row['patient_id'],
                    "slice_id":        row['slice_id'],
                    "image":           row["orig"],         # orig_image_path --> image
                    "label":           int(row["label"])    # orig_label --> label
                })
        return records

    #  https://doi.org/10.1371/journal.pmed.1002699.s001 
    def loss(self, prediction, target):
        """
        Compute the weighted cross-entropy loss.

        Parameters:
        - prediction (Tensor): Model predictions.
        - target (Tensor): Ground truth labels.

        Returns:
        - nn.CrossEntropyLoss(): (Weighted) cross-entropy loss.
        """
        if self.split == 'train' and not self.config['data']['use_synthetic_data']:
            # use weighted CE
            weight = self.class_weights.to(prediction.device)
            criterion = torch.nn.CrossEntropyLoss(weight=weight)
            # print(f"[./utils/custom_data loss] using weighted CE {weight}")
            return criterion(prediction, target)
        else:
            # standard CE
            return torch.nn.CrossEntropyLoss()(prediction, target)
         
    def __getitem__(self, index):
        
        recs = self.records[index]

        transformed = self.transforms(recs)

        image = torch.FloatTensor(transformed['image'])
        label = torch.tensor(recs['label'], dtype=torch.long)
        
        patient_id = recs['patient_id']
        slice_id = recs['slice_id']

        return image, label, patient_id, slice_id


    def __len__(self):
        return len(self.flatten_df())


def save_images_from_loader(loader, save_dir, split,
                            n_images=16, saveims_format=('png',)):
    """
    Save up to n_images from the first batch of loader into save_dir.
    save_formats can be:
      - 'png'         : a single 4×4 grid image
      - 'nifti','nii' : individual .nii.gz slices
      - or a list/tuple containing any combination, e.g. ['png','nifti']
    """
    os.makedirs(save_dir, exist_ok=True)
    try:
        batch = next(iter(loader))
    except StopIteration:
        print(f"[save_images] No data in {split} loader.")
        return

    images, labels, patient_ids, slice_ids = batch
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


def load_data(config, datapath, labelpath, gland_maskpath, norm_type, augment, saveims, saveims_format, rundir, rank=0, world_size=1):
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

    # Create datasets
    train_dataset = FakeFastMRIDataset(config, datapath, labelpath, gland_maskpath, norm_type, augment, split='train')
    valid_dataset = FakeFastMRIDataset(config, datapath, labelpath, gland_maskpath, norm_type, augment=False, split='val')
    test_dataset = FakeFastMRIDataset(config, datapath, labelpath, gland_maskpath, norm_type, augment=False, split='test')
    
    if world_size > 1:
        # DDP case
        # TODO: Distributed dataloader
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(valid_dataset, batch_size=128, sampler=val_sampler, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=128, sampler=test_sampler, num_workers=0, pin_memory=True)

    else:
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=128, num_workers=0, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, num_workers=0, shuffle=False)

    if saveims and rank == 0:
        base_dir = os.path.join(rundir, "sample_images")
        save_images_from_loader(train_loader, os.path.join(base_dir, "train"), "train",
                                n_images=16, saveims_format=saveims_format)
        save_images_from_loader(val_loader, os.path.join(base_dir, "val"), "val",
                                n_images=16, saveims_format=saveims_format)
        save_images_from_loader(test_loader,  os.path.join(base_dir, "test"), "test",
                                n_images=16, saveims_format=saveims_format)
                                
    return train_loader, val_loader, test_loader
