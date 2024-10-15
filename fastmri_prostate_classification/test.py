import nibabel as nib

image_path = "/data2/joanna/FastMRI_Dataset/FastMRI_Dataset/nifti/102/102_T2W.nii.gz"

img = nib.load(image_path)

print(img)