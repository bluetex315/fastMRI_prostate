import os

# Set the path to your main folder
base_folder = "/home/lc2382/project/fastMRI_NYU/nifti"

# Iterate over all subfolders in the base folder
for folder in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder)
    if os.path.isdir(folder_path):
        # Iterate over all files in the subfolder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".nii.gz"):
                # Check if the file name matches the pattern
                if "CALC_BVAL" in file_name:
                    parts = file_name.split('_')
                    if len(parts) >= 5:
                        new_file_name = f"{folder}_b1500.nii.gz"
                        old_file_path = os.path.join(folder_path, file_name)
                        new_file_path = os.path.join(folder_path, new_file_name)
                        # Rename the file
                        os.rename(old_file_path, new_file_path)
                        print(f"Renamed: {old_file_path} -> {new_file_path}")
