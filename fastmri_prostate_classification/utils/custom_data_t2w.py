# import modules
import os
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirst, RandFlip, RandRotate, ScaleIntensityRange, CenterSpatialCrop, NormalizeIntensity
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
    def __init__(self, datapath, labelpath, norm_type, augment, split):
        super().__init__()
        self.datapath = datapath
        self.labelpath = labelpath
        self.norm_type = norm_type
        self.augment = augment
        self.split = split

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

            path_label = os.path.join(labelpath, f"{pt_id}", f"{pt_id}_T2W_PIRADS.npz")
            if not os.path.exists(path_label):
                raise FileNotFoundError(f"Label file not found: {path_label}")

            # patient level dataframe, path_t2 directs to 3D nii, path_label directs to a tensor (D,)
            data_list.append({
                "image": path_t2,
                "label": path_label,
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
            temp_dict = {'image': row['image']}
            image_loader = LoadImageD(keys=['image'], image_only=True)
            image_data = image_loader(temp_dict)['image']
            # print(row['label'])
            label_data = np.load(row['label'])['pirads']
            # print(image_data.shape, label_data.shape)
            h, w, depth = image_data.shape

            for slice_idx in range(depth):
                expanded_data_list.append({
                    "image": image_data[:, :, slice_idx],
                    "label": (label_data[slice_idx] > 2).astype(np.int32),
                    "patient_id": row['patient_id'],
                    "slice_idx": slice_idx
                })
        
        self.data_df = pd.DataFrame(expanded_data_list)
        print(self.data_df.head())
        print(f"Number of slices in {split} set: {len(self.data_df)}")

        # Define MONAI transforms
        if split == 'train':  
            self.transforms = Compose([
                EnsureChannelFirst(),
                RandFlip(prob=0.5, spatial_axis=1),
                RandRotate(range_x=np.pi / 15, prob=0.5),
                ScaleIntensityRange(a_min=-500, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
                CenterSpatialCrop(roi_size=(224, 224)),
                NormalizeIntensity()
            ])
        
        else:
            self.transforms = Compose([
                EnsureChannelFirst(),
                ScaleIntensityRange(a_min=-500, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
                CenterSpatialCrop(roi_size=(224, 224)),
                NormalizeIntensity()
            ])


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


def load_data(datapath, labelpath, norm_type, augment):
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
    train_dataset = FastMRIDataset(datapath, labelpath, norm_type, augment, split='train')
    valid_dataset = FastMRIDataset(datapath, labelpath, norm_type, augment=False, split='val')
    test_dataset = FastMRIDataset(datapath, labelpath, norm_type, augment=False, split='test')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4, shuffle=False)

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