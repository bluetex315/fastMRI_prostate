import monai
import torch
import nibabel as nib
import h5py

# Example usage
path = "/data2/lc2382/fastMRI_prostate_T2_IDS_237_255/file_prostate_AXT2_250.h5"

with h5py.File(path, "r") as hf:
    im_recon_320 = hf["reconstruction_rss"][:]   
    im_recon_320 = im_recon_320[:, :,:]

print(im_recon_320.shape)
