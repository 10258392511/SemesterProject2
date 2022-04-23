import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import re
import pickle
import SemesterProject2.scripts.configs_data_aug as configs_data_aug

from torch.utils.data import Dataset
from monai.transforms import Rand3DElastic, RandGaussianSmooth, RandGaussianNoise, RandAdjustContrast, Compose
from pprint import pprint


class VolumetricDataset(Dataset):
    pass


def train_test_split_and_sort_by_pid():
    # read .pkl files
    filenames = dict()
    for key, data_filename in configs_data_aug.data_filenames.items():
        data_filename = os.path.join(configs_data_aug.data_root, data_filename)
        with open(data_filename, "rb") as rf:
            filenames[key] = pickle.load(rf)  # for each key: list of filenames

    # retrieve pid's
    # pids: {dataset1: [dataset1/pid1...], ...}
    pids = dict()
    key = iter(filenames.keys()).__next__()
    list_of_filenames = filenames[key]  # choose one list of filenames
    for dataset_name in configs_data_aug.dataset_names:
        pids[dataset_name] = []
        pattern_str = re.escape(dataset_name) + r"/[0-9-]+"
        pattern = re.compile(pattern_str)
        for filename_iter in list_of_filenames:
            matches = pattern.findall(filename_iter)
            if len(matches) > 0:
                pids[dataset_name].append(matches[0])

    # train-test split by pid
    train_pids = []
    test_pids = []
    for dataset_name in pids:
        pids_list = pids[dataset_name]
        split_ind = int(len(pids_list) * configs_data_aug.train_test_split)
        train_pids += pids_list[:split_ind]
        test_pids += pids_list[split_ind:]

    # re-sorted dataset dict: {pid1: {"volume": str, "bbox": str, "bbox_coord": str, "mask": str}}
    def construct_train_or_test_dataset_dict_(pid_list):
        dataset_dict = dict()
        for pid_iter in pid_list:
            dataset_dict[pid_iter] = dict()
            # iterate through two dict's: zip; this level: "volume", ...
            for key in filenames:
                assert key in configs_data_aug.volume_filenames, "key not matched"
                suffix = configs_data_aug.volume_filenames[key]
                pattern = re.compile(re.escape(pid_iter) + r"/" + re.escape(suffix))
                # filenames[key]: filenames for volumne / bbox ...
                for filename in filenames[key]:
                    matches = pattern.findall(filename)
                    if len(matches) > 0:
                        # key: "volumne", ...
                        dataset_dict[pid_iter][key] = matches[0]

        return dataset_dict

    train_dataset_dict = construct_train_or_test_dataset_dict_(train_pids)
    test_dataset_dict = construct_train_or_test_dataset_dict_(test_pids)

    # save dataset_dict's
    with open(configs_data_aug.data_splitted_filename, "wb") as wf:
        pickle.dump({"train": train_dataset_dict, "test": test_dataset_dict}, wf)

    return train_dataset_dict, test_dataset_dict


def save_for_visualization(vol: np.ndarray, seg: np.ndarray, save_dir):
    vol_sitk = sitk.GetImageFromArray(vol)
    seg_sitk = sitk.GetImageFromArray(seg)
    vol_path = os.path.join(save_dir, "vol.nii")
    seg_path = os.path.join(save_dir, "seg.nii")
    sitk.WriteImage(vol_sitk, vol_path)
    sitk.WriteImage(seg_sitk, seg_path)


def identity_transform(vol: np.ndarray, seg: np.ndarray, if_save=False):
    if if_save:
        save_for_visualization(vol, seg, configs_data_aug.temp_save_dir)

    return vol, seg


def normalize_volume(vol: np.ndarray, max_percent=98, min_percent=2):
    max_th = np.percentile(vol, max_percent)
    min_th = np.percentile(vol, min_percent)
    vol_out = np.clip((vol - min_th) / (max_th - min_th), 0, 1)

    return vol_out


def spatial_transform(vol: np.ndarray, seg: np.ndarray, if_save=False):
    seed = np.random.randint(configs_data_aug.max_seed)
    transformer = Rand3DElastic(**configs_data_aug.spatial_transform_args)
    transformer.set_random_state(seed=seed)
    vol_out = transformer(vol, mode="bilinear")
    transformer.set_random_state(seed=seed)
    seg_out = transformer(seg, mode="nearest")

    if if_save:
        save_for_visualization(vol_out, seg_out, configs_data_aug.temp_save_dir)

    return vol_out, seg_out


def intensity_transform(vol: np.ndarray, seg: np.ndarray, if_save=False):
    """
    No transform on "seg", which is passed in to be consistent in signature.
    """
    kwargs = configs_data_aug.intensity_transform_args
    gauss_noise = RandGaussianNoise(**kwargs["gaussian_noise"])
    gauss_blur = RandGaussianSmooth(**kwargs["gaussian_blur"])
    random_gamma = RandAdjustContrast(**kwargs["random_contrast"])
    transformer = Compose([gauss_blur, gauss_noise, random_gamma])
    # transformer = Compose([gauss_noise])
    vol_out = transformer(vol)

    if if_save:
        save_for_visualization(vol_out, seg, configs_data_aug.temp_save_dir)

    return vol_out, seg


def close_crop(vol: np.ndarray, seg: np.ndarray, if_save=False):
    vol_sitk = sitk.GetImageFromArray(vol)
    seg_sitk = sitk.GetImageFromArray(seg)
    inside_val, outside_val = 0, 1
    otsu_thresholder = sitk.OtsuThresholdImageFilter()
    otsu_thresholder.SetInsideValue(inside_val)
    otsu_thresholder.SetOutsideValue(outside_val)
    bin_vol = otsu_thresholder.Execute(vol_sitk)

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(bin_vol)
    bbox = label_shape_filter.GetBoundingBox(outside_val)
    bbox_half_len = len(bbox) // 2
    bbox_size = bbox[bbox_half_len:]
    bbox_start_ind = bbox[:bbox_half_len]
    vol_out_sitk = sitk.RegionOfInterest(vol_sitk, bbox_size, bbox_start_ind)
    seg_out_sitk = sitk.RegionOfInterest(seg_sitk, bbox_size, bbox_start_ind)

    vol_out = sitk.GetArrayFromImage(vol_out_sitk)
    seg_out = sitk.GetArrayFromImage(seg_out_sitk)

    if if_save:
        save_for_visualization(vol_out, seg_out, configs_data_aug.temp_save_dir)

    return vol_out, seg_out


if __name__ == '__main__':
    train_dataset_dict, test_dataset_dict = train_test_split_and_sort_by_pid()
    pprint(train_dataset_dict)
    print("-" * 200)
    pprint(test_dataset_dict)
    print(f"number of samples: train: {len(train_dataset_dict)}, test: {len(test_dataset_dict)}")
