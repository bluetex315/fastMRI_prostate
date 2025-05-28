from PIL import Image
import os
import re
from typing import List, Dict
import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_config(config_file):
    """Load a YAML configuration file and return a nested dictionary."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def flatten_config(config):
    """
    Merge nested configuration dictionaries into a single flat dictionary.
    
    For each top-level section (e.g., model_args, data_args, etc.), the inner
    dictionary is merged into a single dictionary. If there are duplicate keys,
    later sections will overwrite the earlier ones.
    """
    flat_config = {}
    for section, sub_config in config.items():
        if isinstance(sub_config, dict):
            flat_config.update(sub_config)
        else:
            flat_config[section] = sub_config
    return flat_config
    
def parse_syn_slices(
    patient_ids: List[str],
    syn_base_path: str,
    split: str  # one of 'train','val','test'
) -> List[Dict]:
    """
    Build a list of dicts, each containing:
      - "image":   full path to the .nii.gz
      - "label":   int class 0-4
      - "patient_id": str patient identifier (e.g. "001")
      - "slice_idx":  int slice number (parsed from "_slice{n}_")
    
    Behavior by split:
      * train -> matches "*_fakeclass{0..4}_syn.nii.gz"
      * val/test -> matches "*_class{0..4}_orig.nii.gz"
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train','val' or 'test', got {split!r}")

    # choose the right filename‐pattern for label
    if split == "train":
        label_re = re.compile(r"fakeclass(\d+)_syn\.nii\.gz$")
    else:
        label_re = re.compile(r"class(\d+)_orig\.nii\.gz$")
    # common slice‐index pattern
    slice_re = re.compile(r"_slice(\d+)_")

    slices: List[Dict] = []
    for pid in patient_ids:
        patient_dir = os.path.join(syn_base_path, f"patient_{pid}")
        if not os.path.isdir(patient_dir):
            raise FileNotFoundError(f"synthetic folder not found: {patient_dir}")

        for fname in sorted(os.listdir(patient_dir)):
            # first, does this file carry the right label‐pattern?
            m_lbl = label_re.search(fname)
            if not m_lbl:
                continue
                # raise ValueError(f"naming not correct {patient_dir}")

            # next, extract slice index
            m_slc = slice_re.search(fname)
            if not m_slc:
                continue
                # raise ValueError(f"naming not correct {patient_dir}")

            label = int(m_lbl.group(1))
            slice_idx = int(m_slc.group(1))
            img_path = os.path.join(patient_dir, fname)

            # reads in the 2D slice
            nii = nib.load(img_path)
            arr = nii.get_fdata(dtype=np.float32)       # shape (H, W)

            slices.append({
                "image": arr,
                "class_label": label,
                "patient_id": pid,
                "slice_idx": slice_idx,
            })

    return slices


def parse_3d_volumes(dset_dict, seg_type, label_csv_file=None):

    if label_csv_file is not None:
        labels_df = pd.read_csv(label_csv_file)
        # Expecting columns: 'patient_id', 'slice_idx', 'class_label'
        labels_dict = {}
        for _, row in labels_df.iterrows():
            # Convert patient_id to string and slice_idx to integer for consistency.
            key = (f"{int(row['fastmri_pt_id']):03d}", int(row['slice']-1))     # slice_idx: row['slice']-1 to match the Python indexing
            labels_dict[key] = row['PIRADS'] - 1        # PI-RADS - 1 as true class labels for nn.Embeddings
    else:
        labels_dict = {}
    
    patient_dict = {}
    
    # Process the original image paths.
    image_key = 'image'
    if image_key in dset_dict:
        for img_path in dset_dict[image_key]:
            # Extract the patient ID.
            basename = os.path.basename(img_path)  # e.g., "110_T2W.nii.gz"
            patient_id = basename.split('_')[0]    # e.g., "110"
            # Create an entry for the patient if not already present.
            if patient_id not in patient_dict:
                patient_dict[patient_id] = {}
            patient_dict[patient_id][image_key] = img_path
            
    # Process segmentation paths (any key that is not "image").
    for key in dset_dict:
        if key == image_key:
            continue
        for seg_path in dset_dict[key]:
            # Extract the patient ID.
            basename = os.path.basename(seg_path)  # e.g., "110_T2W_infer_seg.nii.gz"
            patient_id = basename.split('_')[0]     # e.g., "110"
            if patient_id not in patient_dict:
                patient_dict[patient_id] = {}
            patient_dict[patient_id][key] = seg_path

    selected_slices = []
    
    for patient_id, files in patient_dict.items():
        # Check that both image and segmentation exist for this patient.

        seg_key = 'seg_'+seg_type
        # Load the 3D volumes using nibabel.
        img_vol = nib.load(files['image']).get_fdata()
        seg_vol = nib.load(files[seg_key]).get_fdata()
    
        # Optionally check that image and segmentation have the same number of slices.
        if img_vol.shape[2] != seg_vol.shape[2]:
            print(f"Warning: patient {patient_id} image and segmentation volumes have different number of slices.")
            continue
        
        num_slices = img_vol.shape[2]
        
        # Identify slice indices where the segmentation is "active"
        valid_indices = [z for z in range(seg_vol.shape[2]) if np.count_nonzero(seg_vol[:, :, z]) > 0]
        
        if not valid_indices:
            print(f"No valid segmentation found for patient {patient_id}. Skipping patient.")
            continue
        
        # Determine the selection range.
        # For example, if the active segmentation is between slice 8 and 22, then include from
        # max(0, 8-neighbor_range) to min(num_slices-1, 22+neighbor_range)
        neighbor_range = 0
        min_valid = min(valid_indices)
        max_valid = max(valid_indices)
        start_idx = max(0, min_valid - neighbor_range)
        end_idx = min(num_slices - 1, max_valid + neighbor_range)
        
        # For each slice in the computed range, save the image and segmentation slices.
        for z in range(start_idx, end_idx + 1):
            slice_info = {
                'patient_id': patient_id,
                'slice_idx': z,
                'image': img_vol[:, :, z],
                # seg_key: seg_vol[:, :, z]
            }

            # @deprecated as no need for clean_left, clean_right, and seg_key
            # # Get the left (previous) slice or pad if at boundary.
            # if z - 1 >= 0:
            #     clean_left = img_vol[:, :, z - 1]
            # else:
            #     clean_left = np.zeros(img_vol[:, :, z].shape, dtype=img_vol.dtype)

            # # Get the right (next) slice or pad if at boundary.
            # if z + 1 < num_slices:
            #     clean_right = img_vol[:, :, z + 1]
            # else:
            #     clean_right = np.zeros(img_vol[:, :, z].shape, dtype=img_vol.dtype)

            # slice_info['clean_left'] = clean_left
            # slice_info['clean_right'] = clean_right
            
            # Add the slice-level class label if available.
            key = (patient_id, z)

            if key in labels_dict:
                slice_info['class_label'] = labels_dict[key]
            else:
                raise AssertionError (f"the key {key} does not have a corresponding label, problematic") 
                # slice_info['class_label'] = None  # or you might choose to skip the slice if no label exists.

            selected_slices.append(slice_info)
    
    return selected_slices

def get_patient_splits(datapath, test_size, val_size, seed, exclude_ids=None):
    # 1) list all patient‐folder names
    all_ids = sorted(
        d for d in os.listdir(datapath)
        if os.path.isdir(os.path.join(datapath, d))
    )
    # 2) remove any you don’t want
    if exclude_ids is not None:
        all_ids = [pid for pid in all_ids if pid not in set(exclude_ids)]

    # 3) do the train/val/test splits
    train_ids, temp_ids = train_test_split(
        all_ids, test_size=test_size, random_state=seed
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=val_size, random_state=seed
    )
    return train_ids, val_ids, test_ids


def split_dset_by_patient(dset_dict, split_ids):

    out = {}
    for modality, paths in dset_dict.items():
        # 1) Sort the file‑paths for reproducibility
        sorted_paths = sorted(paths)

        # 2) Filter: only keep files whose patient‑ID is in split_ids
        kept_paths = []
        for path in sorted_paths:
            filename = os.path.basename(path)          # e.g. "152_T2W.nii.gz"
            patient_id = filename.split("_", 1)[0]     # split on first underscore

            if patient_id in split_ids:
                kept_paths.append(path)

        # 3) Assign the filtered list back into the new dict
        out[modality] = kept_paths

    return out



def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def save_nifti(batch, seg_key, output_dir):
    """
    Save a batch of 2D slices as individual NIfTI files.

    Parameters
    ----------
    batch : torch.Tensor
        A tensor of shape [batch_size, channels, height, width], containing the image slices.
    seg_key: str
        Key to index segmentation mask from batch
    output_dir : str, optional
        The directory where the NIfTI files will be saved. Defaults to the current directory.

    Notes
    -----
    - This function assumes that each slice in `batch` corresponds to one patient ID and one slice index.
    - The `affine` used here is a simple identity matrix. Modify it if you have a known orientation and spacing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_size = len(batch['patient_id'])

    seg_guided = True if seg_key else False # Check whether we have segmentation data

    for i in range(batch_size):
        pid = batch['patient_id'][i]
        sidx = batch['slice_idx'][i]

        # Extract the i-th slice (assuming a single channel for image)
        img_tensor = batch['image'][i, 0, ...]  # shape: [height, width]
        img_np = img_tensor.cpu().numpy()  # Convert to NumPy

        # Create a NIfTI image for the main image
        nifti_img = nib.Nifti1Image(img_np, affine=np.eye(4))

        # Construct the output filename for the main image
        image_output_filename = os.path.join(output_dir, f"{pid}_slice_{sidx}_T2W.nii.gz")
        nib.save(nifti_img, image_output_filename)
        print(f"Saved Image: {image_output_filename}")

        # Save segmentation only if it exists
        if seg_guided:
            seg_tensor = batch[seg_key][i, 0, ...]
            seg_np = seg_tensor.cpu().numpy()
            seg_img = nib.Nifti1Image(seg_np, affine=np.eye(4))
            seg_output_filename = os.path.join(output_dir, f"{pid}_slice_{sidx}_{seg_key}.nii.gz")
            nib.save(seg_img, seg_output_filename)
            print(f"Saved Segm: {seg_output_filename}")