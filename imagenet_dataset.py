from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import pickle
import torch.utils.data as data


def triplets_inds(targets):
    """
    Original random triplets indices

    :param targets: labels for each images of the batch (batch_size, 1)
    :return: indices for p, q, n
    """
    q, p, n = [], [], []

    targets = np.array(targets)

    for i in list(set(targets)):

        pinds = np.argwhere(targets == i).flatten()
        ninds = np.argwhere(targets != i).flatten()

        pinds_shuffled = pinds.copy()
        np.random.shuffle(pinds_shuffled)
        ninds_shuffle = np.random.choice(ninds, len(pinds))

        q.extend(list(pinds))
        p.extend(list(pinds_shuffled))
        n.extend(list(ninds_shuffle))

    return q, p, n


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def create_data(train=True, i=0):
    """
    Create data lists for imagenet dataset

    :param train: boolean if we want training set or testing set
    :param i: which file to use for the training set
    :return: list of data and corresponding classes
    """
    all_train_list = ['train_data_batch_1', 'train_data_batch_2',
                      'train_data_batch_3', 'train_data_batch_4',
                      'train_data_batch_5', 'train_data_batch_6',
                      'train_data_batch_7', 'train_data_batch_8',
                      'train_data_batch_9', 'train_data_batch_10']

    train_list = [all_train_list[i]]

    test_list = ['val_data']

    data = []
    targets = []

    dir = "/data/data_pierre/data/Imagenet32"

    if train:
        downloaded_list = train_list
    else:
        downloaded_list = test_list

    print("list : ", downloaded_list[0])

    # now load the picked numpy arrays
    for file_name in downloaded_list:
        file_path = os.path.join(dir, file_name)

        d = unpickle(file_path)
        x = d['data']
        y = d['labels']

        x = x / np.float32(255)

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i - 1 for i in y]
        data_size = x.shape[0]

        if train:
            mean_image = d['mean']
            mean_image = mean_image / np.float32(255)
            x -= mean_image

        img_size2 = 32 * 32

        x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
        x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)

        # create mirrored images
        X_train = x[0:data_size, :, :, :]
        Y_train = y[0:data_size]

        data.extend(X_train)
        targets.extend(Y_train)

    return data, targets

class Imagenet32(data.Dataset):
    """
    Create dataset for Imagenet, allowed easy load and generate triplets on the fly
    """

    def __init__(self, data, targets, train,
                 transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.train = train  # training set or test set

        self.data = data
        self.targets = targets

        self.triplet_indices = triplets_inds(self.targets)

        i1 = self.triplet_indices[0][0]
        i2 = self.triplet_indices[1][0]
        i3 = self.triplet_indices[2][0]

        print("triplet targets : ", i1, self.targets[i1], i2, self.targets[i2], i3, self.targets[i3])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        print("data size : ", len(self.data))
        print("target size : ", len(self.targets))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        def create_image(index, current_samples):
            """
            generate image and its corresponding class for a given sample index / batch

            :param index: index of the samples to generate
            :param current_samples: corresponding triplet to the given index
            :return: image and corresponding class
            """

            index_c = current_samples[index]

            img, target = self.data[index_c], self.targets[index_c]

            img = Image.fromarray((img*255).astype('uint8'))

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

        sample_q, target_q = create_image(index, self.triplet_indices[0])

        # if we train we also need sample and target for positive and negative images.
        if self.train:
            sample_p, target_p = create_image(index, self.triplet_indices[1])
            sample_n, target_n = create_image(index, self.triplet_indices[2])

            return (sample_q, sample_p, sample_n), (target_q, target_p, target_n)

        else:
            return sample_q, target_q

    def __len__(self):
        return len(self.data)