# import modules
import os
import fnmatch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import shutil
# import find_files_with_pattern
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirstd, RandAffined, RandFlipd, RandRotated, ScaleIntensityRanged, CenterSpatialCropd, NormalizeIntensityd, ResampleToMatchd, EnsureChannelFirstd
)
from monai.data import Dataset, DataLoader

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

#This set up is for resampling t2w and mask into ADC and B1500 

def find_files_with_pattern(directory, pattern):
    # matching_files = []
    full_path = None  # Initialize full_path before the loop

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                full_path = os.path.join(root, filename)  # Assign value when a match is found
                # matching_files.append(full_path)
                
    # if full_path is None:
    #     print("No matching files found.")
    # else:
    #     print(f"Last found matching file: {full_path}")

    return full_path

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
        
        # print(f"LOADING B1500        --> {self.config['concat_b1500']}")
        print(f"LOADING T2W        --> {self.config['concat_t2w']}")
        print(f"LOADING gland_mask --> {self.config['concat_mask']}")

        # Create a dataframe with patient IDs, image paths, and labels
        data_list = []
        patient_ids = sorted(os.listdir(datapath))
        pattern = '*CALC_BVAL*' + '*.nii.gz'
        for pt_id in patient_ids:
            pt_folder = os.path.join(datapath, pt_id)
            
            if not os.path.isdir(pt_folder):
                raise FileNotFoundError(f"Patient folder not found: {pt_folder}")

            path_adc = os.path.join(pt_folder, f"{pt_id}_ADC.nii.gz")
            if not os.path.exists(path_adc):
                raise FileNotFoundError(f"ADC image not found: {path_adc}")

            path_b1500 = find_files_with_pattern(pt_folder, pattern)
            if not os.path.exists(path_b1500):
                raise FileNotFoundError(f"B1500 image not found: {path_b1500}")
                
            # Load ADC path only if config['concat_adc'] is True
            path_t2w = None
            if self.config['concat_t2w']:
                path_t2w = os.path.join(pt_folder, f"{pt_id}_T2W.nii.gz")
                if not os.path.exists(path_t2w):
                    raise FileNotFoundError(f"T2W image not found: {path_t2w}")

            path_label = os.path.join(labelpath, f"{pt_id}", f"{pt_id}_ADC_PIRADS.npz")
            if not os.path.exists(path_label):
                raise FileNotFoundError(f"Label file not found: {path_label}")

            # Load gland mask path only if config['concat_mask'] is True
            path_gland_mask = None
            if self.config['concat_mask']:
                path_gland_mask = os.path.join(gland_maskpath, f"{pt_id}", f"{pt_id}_T2W_segm.nii.gz")
                if not os.path.exists(path_gland_mask):
                    raise FileNotFoundError(f"Gland mask file not found: {path_gland_mask}")

            data_entry = {
                "adc": path_adc,
                "b1500": path_b1500,
                "label": path_label,
                "patient_id": pt_id
            }

            # Conditionally add paths if they exist and are required by the configuration
            if path_t2w and os.path.exists(path_t2w):
                data_entry["t2w"] = path_t2w
            if path_gland_mask and os.path.exists(path_gland_mask):
                data_entry["gland_mask"] = path_gland_mask

            # patient level dataframe, path_t2 directs to 3D nii, path_label directs to a tensor (D,)
            data_list.append(data_entry)

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
            temp_dict = {'adc': row['adc'],'b1500':row['b1500']}
            

            # Conditionally add 'adc' only if config requires it
            if self.config['concat_t2w']:
                temp_dict['t2w'] = row['t2w']
            # Conditionally add 'gland_mask' only if config requires it
            if self.config['concat_mask']:
                temp_dict['gland_mask'] = row['gland_mask']

            load_keys = list(temp_dict.keys())
            # resampling t2w and mask into adc space
            transforms = [LoadImageD(keys=load_keys, image_only=True)]
            transforms.append(EnsureChannelFirstd(keys=load_keys))
            if 't2w' in temp_dict:
                transforms.append(ResampleToMatchd(keys='t2w', key_dst='adc', mode='bilinear'))
            if 'gland_mask' in temp_dict:
                transforms.append(ResampleToMatchd(keys='gland_mask', key_dst='adc', mode='bilinear'))

            image_loader = Compose(transforms)
            
            data = image_loader(temp_dict)
            adc_data = data['adc'][0]
            b1500_data = data['b1500'][0]
            t2w_data = data.get('t2w', None)
            gland_mask_data = data.get('gland_mask', None)

            if t2w_data is not None:
                t2w_data = t2w_data[0]
            if gland_mask_data is not None:
                gland_mask_data = gland_mask_data[0]

            label_data = np.load(row['label'])['pirads'][::-1]

            start_idx = 0
            end_idx = adc_data.shape[2] - 1

            if gland_mask_data is not None:
                if adc_data.shape != gland_mask_data.shape:
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
                    "adc": adc_data[:, :, slice_idx],
                    "b1500": b1500_data[:, :, slice_idx],
                    "label": (label_data[slice_idx] > 2).astype(np.int32),
                    "patient_id": row['patient_id'],
                    "slice_idx": slice_idx
                }
                # Conditionally add 'adc' if it exists
                if t2w_data is not None:
                    expanded_data_entry["t2w"] = t2w_data[:, :, slice_idx]
                
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

        norm_keys = ['adc']
        if 't2w' in load_keys:
            norm_keys.append('t2w')

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
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor)) 
        return loss


    def __getitem__(self, index):

        data_dict = {"adc": self.data_df.iloc[index]['adc'], "b1500": self.data_df.iloc[index]['b1500']}

        # add 'adc' and 'mask' if the configuration requires it
        if self.config.get('concat_t2w', True):
            data_dict["t2w"] = self.data_df.iloc[index]['t2w']
            # print(data_dict['adc'].shape)
        if self.config.get('concat_mask', True):
            data_dict["gland_mask"] = self.data_df.iloc[index]['gland_mask']
            # print(data_dict['gland_mask'].shape)
        
        label = self.data_df.iloc[index]['label']
        
        transformed = self.transforms(data_dict)

        image = torch.FloatTensor(transformed['adc'])
        image = torch.cat((image, transformed['b1500']), dim=0)

        # Concatenate adc if it exists in the transformed data
        if self.config.get('concat_t2w', True) and 't2w' in transformed:
            image = torch.cat((image, transformed['t2w']), dim=0)
        if self.config.get('concat_mask', True) and 'gland_mask' in transformed:
            image = torch.cat((image, transformed['gland_mask']), dim=0)

        label = torch.FloatTensor([label])

        return image, label


    def __len__(self):
        return len(self.data_df)

 
def load_data(config, datapath, labelpath, gland_maskpath, norm_type, augment, saveims, rundir):
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
    train_dataset = FastMRIDataset(config, datapath, labelpath, gland_maskpath, norm_type, augment, split='train')
    valid_dataset = FastMRIDataset(config, datapath, labelpath, gland_maskpath, norm_type, augment=False, split='val')
    test_dataset = FastMRIDataset(config, datapath, labelpath, gland_maskpath, norm_type, augment=False, split='test')

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