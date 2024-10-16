# import modules
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirst, RandAffine, RandFlip, RandRotate, ScaleIntensityRange, CenterSpatialCrop, NormalizeIntensity
)
from monai.data import Dataset, DataLoader


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
    def __init__(self, t2w_datapath, labelpath, gland_maskpath, norm_type, augment, split):
        super().__init__()
        self.t2w_datapath = t2w_datapath
        self.labelpath = labelpath
        self.gland_maskpath = gland_maskpath
        self.norm_type = norm_type
        self.augment = augment
        self.split = split

        # Create a dataframe with patient IDs, image paths, and labels
        data_list = []
        patient_ids = sorted(os.listdir(t2w_datapath))
        for pt_id in patient_ids:
            pt_folder = os.path.join(t2w_datapath, pt_id)
            if not os.path.isdir(pt_folder):
                raise FileNotFoundError(f"Patient folder not found: {pt_folder}")

            path_t2 = os.path.join(pt_folder, f"{pt_id}_T2W.nii.gz")
            if not os.path.exists(path_t2):
                raise FileNotFoundError(f"T2 image not found: {path_t2}")

            path_label = os.path.join(labelpath, f"{pt_id}", f"{pt_id}_T2W_PIRADS.npz")
            if not os.path.exists(path_label):
                raise FileNotFoundError(f"Label file not found: {path_label}")

            path_gland_mask = os.path.join(gland_maskpath, f"{pt_id}", f"{pt_id}_T2W_segm.nii.gz")

            # patient level dataframe, path_t2 directs to 3D nii, path_label directs to a tensor (D,)
            data_list.append({
                "t2w": path_t2,
                "label": path_label,
                "gland_mask": path_gland_mask,
                "patient_id": pt_id
            })

        full_data_df = pd.DataFrame(data_list)

        # Split the dataset into train, validation, and test sets using train_test_split
        train_df, temp_df = train_test_split(full_data_df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

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
            temp_dict = {'t2w': row['t2w'], 'gland_mask': row['gland_mask']}
            # print(temp_dict)
            image_loader = LoadImageD(keys=['t2w', 'gland_mask'], image_only=True, reader='NibabelReader')
            data = image_loader(temp_dict)

            t2w_data = data['t2w']
            gland_mask_data = data['gland_mask']
            # print(t2w_data, t2w_data.shape)
            # print(gland_mask_data, gland_mask_data.shape)
            z_slices_sum = np.sum(gland_mask_data, axis=(0,1))
            z_indices = np.where(np.any(gland_mask_data, axis=(0, 1)))[0]
            print(z_indices)
            print(z_slices_sum)
            # Calculate start and end slice indices with margins
            try:
                # Try accessing z_indices assuming it's not empty
                start_idx = max(z_indices[0] - 5, 0)
                end_idx = min(z_indices[-1] + 5, gland_mask_data.shape[2] - 1)

                # start_idx = 0
                # end_idx = t2w_data.shape[2] - 1

            except (IndexError, TypeError) as e:
                # Handle cases where z_indices is empty or not iterable
                print(f"An error occurred: {e}")
                print("else case triggered", temp_dict)
                start_idx = 0
                end_idx = t2w_data.shape[2] - 1

            label_data = np.load(row['label'])['pirads']
            count1 = sum(1 for x in label_data if x != 1 and x != 2)
            count2 = sum(1 for x in label_data[start_idx:end_idx+1] if x != 1 and x != 2)

            if count1 != count2:
                print(temp_dict)
                print(label_data)
                print(label_data[start_idx:end_idx+1])
            print(count1, count2)
            print()

            for slice_idx in range(start_idx, end_idx + 1):
                expanded_data_list.append({
                    "image": t2w_data[:, :, slice_idx],
                    "label": (label_data[slice_idx] > 2).astype(np.int32),
                    "patient_id": row['patient_id'],
                    "slice_idx": slice_idx
                })
        
        self.data_df = pd.DataFrame(expanded_data_list)
        print(self.data_df.head())
        print(f"Number of slices in {split} set: {len(self.data_df)}")
        print(f"Number of pos slices in {split} set: {np.sum(self.data_df['label'])}")

        self.labels = np.asarray(self.data_df['label'].values)                       
        neg_weight = np.mean(self.labels)                          
        self.weights = [neg_weight, 1 - neg_weight]                
        
        print("Weights for binary CE:{}".format(self.weights))     
        
        # Define MONAI transforms
        if split == 'train':  
            self.transforms = Compose([
                EnsureChannelFirst(),
                RandAffine(
                    prob=0.5,
                    translate_range=(0,16,16)
                ),
                RandFlip(prob=0.5, spatial_axis=1),
                RandRotate(range_x=12,   # Degrees of rotation for the x-axis (between -12 and 12)
                    range_y=0.0,  # No rotation for the y-axis
                    range_z=0.0,  # No rotation for the z-axis
                    prob=0.5,     # Ensure that the rotation is always applied
                    keep_size=True, # Keep the same image size after rotation (reshape=False equivalent)
                    padding_mode="zeros"
                ),
                # ScaleIntensityRange(a_min=-500, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
                CenterSpatialCrop(roi_size=(224, 224)),
                NormalizeIntensity()
            ])
        
        else:
            self.transforms = Compose([
                EnsureChannelFirst(),
                # ScaleIntensityRange(a_min=-500, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
                CenterSpatialCrop(roi_size=(224, 224)),
                NormalizeIntensity()
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
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor)) 
        return loss


    def __getitem__(self, index):
        # print(index)
        image_2d = self.data_df.iloc[index]['image']
        label = self.data_df.iloc[index]['label']

        # Apply transforms
        transformed_image = self.transforms(image_2d)
        image = torch.FloatTensor(transformed_image)
        label = torch.FloatTensor([label])

        return image, label


    def __len__(self):
        return len(self.data_df)

 
def load_data(datapath, labelpath, gland_segmpath, norm_type, augment, saveims, rundir):
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
    train_dataset = FastMRIDataset(datapath, labelpath, gland_segmpath, norm_type, augment, split='train')
    valid_dataset = FastMRIDataset(datapath, labelpath, gland_segmpath, norm_type, augment=False, split='val')
    test_dataset = FastMRIDataset(datapath, labelpath, gland_segmpath, norm_type, augment=False, split='test')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    datapath = "/home/lc2382/project/fastMRI_NYU/nifti"
    labelpath = "/home/lc2382/project/fastMRI_NYU/labels/pirads_t2w_npz"
    norm_type = 2
        
    # dataset = FastMRIDataset(datapath, labelpath, norm_type, augment=False, split='train')
    # print(f"Total number of slices in dataset: {len(dataset)}")

    train_loader, val_loader, test_loader = load_data(datapath, labelpath, norm_type, augment=False)

    # print(len(train_loader))
    image, label = next(iter(train_loader))
    print(image.shape, label)
