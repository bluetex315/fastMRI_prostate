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

#This set up is for resampling ADC to match T2W and merge two channels for classification of slice level PI-RADS

class FastMRIDataset(data.Dataset):
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
        
        print(f"LOADING ADC        --> {self.config['concat_adc']}")
        print(f"LOADING gland_mask --> {self.config['concat_mask']}")

        # Create a dataframe with patient IDs, image paths, and labels
        data_list = []
        patient_ids = sorted(os.listdir(datapath))
        for pt_id in patient_ids:
            pt_folder = os.path.join(datapath, pt_id)
            if not os.path.isdir(pt_folder):
                raise FileNotFoundError(f"Patient folder not found: {pt_folder}")

            path_t2 = os.path.join(pt_folder, f"{pt_id}_T2W.nii.gz")
            if not os.path.exists(path_t2):
                raise FileNotFoundError(f"T2 image not found: {path_t2}")
                
            # Load ADC path only if config['concat_adc'] is True
            path_adc = None
            if self.config['concat_adc']:
                path_adc = os.path.join(pt_folder, f"{pt_id}_ADC.nii.gz")
                if not os.path.exists(path_adc):
                    raise FileNotFoundError(f"ADC image not found: {path_adc}")

            path_label = os.path.join(labelpath, f"{pt_id}", f"{pt_id}_T2W_PIRADS.npz")
            if not os.path.exists(path_label):
                raise FileNotFoundError(f"Label file not found: {path_label}")

            # Load gland mask path only if config['concat_mask'] is True
            path_gland_mask = None
            if self.config['concat_mask']:
                path_gland_mask = os.path.join(gland_maskpath, f"{pt_id}", f"{pt_id}_T2W_segm.nii.gz")
                if not os.path.exists(path_gland_mask):
                    raise FileNotFoundError(f"Gland mask file not found: {path_gland_mask}")

            data_entry = {
                "t2w": path_t2,
                "label": path_label,
                "patient_id": pt_id
            }

            # Conditionally add paths if they exist and are required by the configuration
            if path_adc and os.path.exists(path_adc):
                data_entry["adc"] = path_adc
            if path_gland_mask and os.path.exists(path_gland_mask):
                data_entry["gland_mask"] = path_gland_mask

            # patient level dataframe, path_t2 directs to 3D nii, path_label directs to a tensor (D,)
            data_list.append(data_entry)

        full_data_df = pd.DataFrame(data_list)

        # Split the dataset into train, validation, and test sets using train_test_split
        train_df, temp_df = train_test_split(full_data_df, test_size=0.3, random_state=self.config['seed'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=self.config['seed'])

        expanded_data_list = []
        
        if split == 'train':
            split_df = train_df.reset_index(drop=True)
        elif split == 'val':
            split_df = val_df.reset_index(drop=True)
        elif split == 'test':
            split_df = test_df.reset_index(drop=True)
        else:
            raise ValueError("Invalid split value. Must be 'train', 'val', or 'test'.")
        
        for _, row in split_df.iterrows():
            temp_dict = {'t2w': row['t2w']}

            # Conditionally add 'adc' only if config requires it
            if self.config['concat_adc']:
                temp_dict['adc'] = row['adc']
            # Conditionally add 'gland_mask' only if config requires it
            if self.config['concat_mask']:
                temp_dict['gland_mask'] = row['gland_mask']

            load_keys = list(temp_dict.keys())
            # resampling adc into t2 space
            transforms = [LoadImaged(keys=load_keys, image_only=True)]
            transforms.append(EnsureChannelFirstd(keys=load_keys))
            if 'adc' in temp_dict:
                transforms.append(ResampleToMatchd(keys='adc', key_dst='t2w', mode='bilinear'))

            image_loader = Compose(transforms)
            
            data = image_loader(temp_dict)
            t2w_data = data['t2w'][0]
            adc_data = data.get('adc', None)
            gland_mask_data = data.get('gland_mask', None)

            if adc_data is not None:
                adc_data = adc_data[0]
            if gland_mask_data is not None:
                gland_mask_data = gland_mask_data[0]

            label_data = np.load(row['label'])['pirads'][::-1]      # reverse label due to DICOM loading, the image is loaded in reverse

            start_idx = 0
            end_idx = t2w_data.shape[2] - 1

            if gland_mask_data is not None:
                if t2w_data.shape != gland_mask_data.shape:
                    print("problematic gland mask shape not align", temp_dict)
                    continue
                z_slices_sum = np.sum(gland_mask_data, axis=(0,1))
                z_indices = np.where(np.any(gland_mask_data, axis=(0, 1)))[0]
                
                # Calculate start and end slice indices with margins
                try:
                    # Try accessing z_indices assuming it's not empty
                    # start_idx = max(z_indices[0] - 3, 0)
                    # end_idx = min(z_indices[-1] + 3, gland_mask_data.shape[2] - 1)
                    start_idx = 0
                    end_idx = t2w_data.shape[2] - 1

                    count1 = sum(1 for x in label_data if x != 1 and x != 2)
                    count2 = sum(1 for x in label_data[start_idx:end_idx+1] if x != 1 and x != 2)

                    if count1 != count2:
                        print(temp_dict)
                        print(label_data)
                        print(label_data[start_idx:end_idx+1])

                except (IndexError, TypeError) as e:
                    # Handle cases where z_indices is empty or not iterable
                    print(f"An error occurred: {e}")
                    print("else case triggered", temp_dict)
                    # start_idx = 0
                    # end_idx = t2w_data.shape[2] - 1
                    continue

            for slice_idx in range(start_idx, end_idx + 1):
                expanded_data_entry = {
                    "t2w": t2w_data[:, :, slice_idx],
                    "label": (label_data[slice_idx] > 2).astype(np.int32),
                    "patient_id": row['patient_id'],
                    "slice_idx": slice_idx
                }
                # Conditionally add 'adc' if it exists
                if adc_data is not None:
                    expanded_data_entry["adc"] = adc_data[:, :, slice_idx]
                
                # Conditionally add 'gland_mask' if it exists
                if gland_mask_data is not None:
                    expanded_data_entry["gland_mask"] = gland_mask_data[:, :, slice_idx]

                # Append the constructed entry to the list
                expanded_data_list.append(expanded_data_entry)
        
        self.data_df = pd.DataFrame(expanded_data_list)
        print(self.data_df.head())
        print(f"Number of slices in {split} set: {len(self.data_df)}")
        print(f"Number of pos slices in {split} set: {np.sum(self.data_df['label'])}")

        self.labels = np.asarray(self.data_df['label'].values)                       
        neg_weight = np.mean(self.labels)                          
        self.weights = [neg_weight, 1 - neg_weight]                
        
        print("Weights for binary CE:{}".format(self.weights))     
        print("keys for transform", load_keys)

        norm_keys = ['t2w']
        if 'adc' in load_keys:
            norm_keys.append('adc')

        # Define MONAI transforms
        if split == 'train':  
            self.transforms = Compose([
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
                CenterSpatialCropd(keys=load_keys, roi_size=(224, 224)),
                NormalizeIntensityd(keys=norm_keys)
            ])

        else:
            self.transforms = Compose([
                EnsureChannelFirstd(keys=load_keys),
                CenterSpatialCropd(keys=load_keys, roi_size=(224, 224)),
                NormalizeIntensityd(keys=norm_keys)
            ])

    #  https://doi.org/10.1371/journal.pmed.1002699.s001 
    def weighted_loss(self, prediction, target):
        """
        Compute the weighted cross-entropy loss.

        Parameters:
        - prediction (Tensor): Model predictions.
        - target (Tensor): Ground truth labels.

        Returns:
        - loss (Tensor): Weighted cross-entropy loss.
        """
        weights_npy = np.array([self.weights[int(t)] for t in target.data])    
        weights_tensor = torch.FloatTensor(weights_npy).cuda()          
        if self.config['focal_loss']:
            loss = sigmoid_focal_loss(prediction, target, alpha=0.95, gamma=2, reduction='mean') 
        else:
            loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor)) 
        return loss

    def __getitem__(self, index):

        data_dict = {"t2w": self.data_df.iloc[index]['t2w']}

        # add 'adc' and 'mask' if the configuration requires it
        if self.config.get('concat_adc', True):
            data_dict["adc"] = self.data_df.iloc[index]['adc']
            # print(data_dict['adc'].shape)
        if self.config.get('concat_mask', True):
            data_dict["gland_mask"] = self.data_df.iloc[index]['gland_mask']
            # print(data_dict['gland_mask'].shape)
        
        label = self.data_df.iloc[index]['label']
        
        transformed = self.transforms(data_dict)

        image = torch.FloatTensor(transformed['t2w'])

        # Concatenate adc if it exists in the transformed data
        if self.config.get('concat_adc', True) and 'adc' in transformed:
            image = torch.cat((image, transformed['adc']), dim=0)
        if self.config.get('concat_mask', True) and 'gland_mask' in transformed:
            image = torch.cat((image, transformed['gland_mask']), dim=0)

        label = torch.FloatTensor([label])

        return image, label


    def __len__(self):
        return len(self.data_df)


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
        
        rows = []
        pattern = re.compile(
            r'(?P<epoch>\d+)_'           
            r'(?P<patient>\d+)_slice'    
            r'(?P<slice>\d+)_'           
            r'(?P<kind>class|fakeclass)'
            r'(?P<label>\d+)_'           
            r'(?P<suffix>orig|syn)\.nii\.gz'
        )     # naming pattern of fake images


        patient_dirs = [d for d in os.listdir(self.datapath) if d.startswith("patient_")]         # filter down to only the patient directories

        # Extract the numeric ID from each name
        patient_ids = []
        for d in patient_dirs:
            match = re.match(r"patient_(\d+)", d)
            if match:
                patient_ids.append(match.group(1))

        # split off 70% train, 30% temp
        train_ids, temp_ids = train_test_split(
            patient_ids,
            train_size=0.7,
            random_state=self.config['seed'],
            shuffle=True
        )

        # then split that 30% into equal val/test (0.15 each of original)
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=0.5,   # half of temp_ids → 0.15 of total
            random_state=self.config['seed'],
            shuffle=True
        )

        print(f"Train: {len(train_ids)} patients")
        print(f"Val:   {len(val_ids)} patients")
        print(f"Test:  {len(test_ids)} patients")
        print("")

        if self.split == "train":
            patient_ids = train_ids
        elif self.split == "val":
            patient_ids = val_ids
        else:
            patient_ids = test_ids

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
