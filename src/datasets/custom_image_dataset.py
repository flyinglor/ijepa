import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from logging import getLogger
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torchio as tio
import h5py

_GLOBAL_SEED = 0
logger = getLogger()


def make_adni(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    test=False
):
    dataset = get_adni_dataset(root_path, train=training, train_transform=transform, test=test)

    # if subset_file is not None:
    #     dataset = ImageNetSubset(dataset, subset_file)

    logger.info('ADNI dataset created')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    logger.info('ADNI unsupervised data loader created')

    return dataset, data_loader, dist_sampler

def make_ukb(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):
    dataset = get_ukb_dataset(root_path, train=training, train_transform=transform)

    # if subset_file is not None:
    #     dataset = ImageNetSubset(dataset, subset_file)

    logger.info('UKB dataset created')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    logger.info('UKB unsupervised data loader created')

    return dataset, data_loader, dist_sampler

def make_dzne(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    mode="train",
    fold=1,
):
    dataset = get_dzne_dataset(root_path, i=fold, mode=mode, train_transform=transform)

    # if subset_file is not None:
    #     dataset = ImageNetSubset(dataset, subset_file)

    logger.info(f'DZNE {mode}set created')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    logger.info(f'DZNE {mode}set data loader created')

    return dataset, data_loader, dist_sampler

def make_hospital(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    mode="train",
    fold=1,
):
    dataset = get_hos_dataset(root_path, i=fold, mode=mode, train_transform=transform)

    # if subset_file is not None:
    #     dataset = ImageNetSubset(dataset, subset_file)

    logger.info(f'Hospital {mode}set created')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    logger.info(f'Hospital {mode}set data loader created')

    return dataset, data_loader, dist_sampler


class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_ukb_dataset(root_path, train=True, train_transform=None):
    suffix = 'uk_train.h5' if train else 'uk_valid.h5'
    data_dir = os.path.join(root_path, suffix)

    image_train = []
    label_train = []
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if train_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = train_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            image_train.append(transformed_image)
            label_train.append("X")

    train_set = CustomImageDataset(image_train, label_train)
    return train_set

def get_adni_dataset(root_path, train=True, train_transform=None, test=False):
    suffix = 'train.h5' if train else 'valid.h5'
    data_dir = os.path.join(root_path, suffix)
    diagnosis = []
    image_train = []
    label_train = []
    i = 0
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if test and i >= 16:
                break
            if name == "stats":
                continue
            rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if train_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = train_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            image_train.append(transformed_image)
            label_train.append(group.attrs['DX'])
            i += 1

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_train = OH_encoder.transform(np.array(label_train).reshape(-1, 1))
    train_set = CustomImageDataset(image_train, label_train)
    return train_set

def get_hos_dataset(root_path, i=1, mode="train", train_transform=None):
    suffix = "238+19+72_tum.h5"
    data_dir = os.path.join(root_path, suffix)
    if mode=="train":
        train_data = np.load(f'{root_path}{i}-train.npy', allow_pickle=True)
    elif mode=="val":
        train_data = np.load(f'{root_path}{i}-valid.npy', allow_pickle=True)
    else:
        train_data = np.load(f'{root_path}{i}-test.npy', allow_pickle=True)
    diagnosis = []
    image_train = []
    label_train = []
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if train_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = train_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            if rid in train_data:
                image_train.append(transformed_image)
                label_train.append(group.attrs['DX'])

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_train = OH_encoder.transform(np.array(label_train).reshape(-1, 1))
    train_set = CustomImageDataset(image_train, label_train)
    return train_set

def get_dzne_dataset(root_path, i=1, mode="train", train_transform=None):
    suffix = "DZNE_CN_FTD_AD.h5"
    data_dir = os.path.join(root_path, suffix)
    if mode=="train":
        train_df = pd.read_csv(f'{root_path}{i}-train.csv')
        train_data = list(train_df["IMAGEID"])
    elif mode=="val":
        train_df = pd.read_csv(f'{root_path}{i}-valid.csv')
        train_data = list(train_df["IMAGEID"])
    else:
        train_df = pd.read_csv(f'{root_path}{i}-test.csv')
        train_data = list(train_df["IMAGEID"])

    diagnosis = []
    image_train = []
    label_train = []
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            rid = group.attrs["IMAGEID"]
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if train_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = train_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            if rid in train_data:
                image_train.append(transformed_image)
                label_train.append(group.attrs['DX'])


    # print(len(diagnosis))
    # print(len(label_train))
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_train = OH_encoder.transform(np.array(label_train).reshape(-1, 1))
    train_set = CustomImageDataset(image_train, label_train)
    return train_set