import numpy as np
import os


# modify the paths
temp_save_dir = r"D:\testings\Python\TestingPython\SemesterProject2\data\temp"
hdf5_filename = r"G:\Legion\Research\Biomedical Imaging\Semester Project 2\dataset\dataset.hdf5"
data_root = r"D:\testings\Python\TestingPython\SemesterProject2\data\project"
data_filenames = {
    "volume": os.path.join(data_root, "imgFiles.pkl"),
    "bbox": os.path.join(data_root, "boundingbox_maskFiles.pkl"),
    "bbox_coord": os.path.join(data_root, "boundingbox_coordinatesFiles.pkl"),
    "mask": os.path.join(data_root, "maskFiles.pkl")
}
dataset_names = ("Melanoma_nii", "Breast_nii", "NSCLC_nii", "melanomapreopMRI_nii")
volume_filenames = {
    "volume": "reg_img_cropped_bias_correc.nii.gz",
    "bbox": "reg_mask_cropped_bounding_box.nii.gz",
    "bbox_coord": "bounding_boxes_coordinates.csv",
    "mask": "reg_mask_cropped_corrected.nii.gz"
}
volume_info_keys = ("volume", "bbox", "bbox_coord", "mask")
# {"train": dict, "test": dict}
data_splitted_filename = r"D:\testings\Python\TestingPython\SemesterProject2\dataset\dataset_by_pid.pkl"

train_test_split = 0.9
max_seed = 1000

spatial_transform_args = {
    "sigma_range": (5, 7),
    "magnitude_range": (0.3, 0.9),
    "prob": 0.5,
    "rotate_range": np.deg2rad([15, 15, 15]),
    "translate_range": (10, 10, 10),
    "padding_mode": "zeros"
}

intensity_transform_args = {
    "gaussian_noise": {
        "prob": 0.5,
        "mean": 0,
        "std": 0.1
    },
    "gaussian_blur": {
        "sigma_x": (0.25, 1.5),
        "sigma_y": (0.25, 1.5),
        "sigma_z": (0.25, 1.5),
        "prob": 0.5
    },
    "random_contrast": {
        "gamma": (0.5, 2),
        "prob": 0.5
    }
}
