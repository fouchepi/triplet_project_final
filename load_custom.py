from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import triplet_images_dataset as tid
import numpy as np
import pandas as pd

def load_data(data_dir, dataset, zsl=False, original_split=True):
    """
    Return everything needed in order to create a train and test data-loader for AWA2

    :param data_dir: the data-set folder
    :param dataset: the data-set to use (just AWA2 now for efficacy but can be change later)
    :param zsl: boolean if we do zero shot learning
    :param original_split: boolean if we use the original split or proposed split for zsl on AWA2
    :return: data_path, classes, class_to_idx, x_train, x_test, train_sampler, nb_classes, not_train
    """

    if dataset == "AWA2":
        print('* Using AWA2')
        data_path = data_dir + '/Animals_with_Attributes2/JPEGImages'

        # find all classes from the dataset
        classes, class_to_idx = tid.find_all_classes(data_path)
        print("classes to idx : ", class_to_idx)
        nb_classes = len(classes)
        print("number of classes : ", nb_classes)

        # create the dataset (list of images path and label)
        images = tid.make_dataset(data_path, classes, class_to_idx, '.jpg')

        # find the smallest nb of samples for one class
        min_samples_of_classes = Counter([image[1] for image in images]).most_common()[-1][1]
        # % of training samples respectively to testing samples
        train_size = 0.8

        # take for training samples for each class only the ratio "train_size" of the smallest class
        nb_train_by_classes = round(min_samples_of_classes * train_size)
        print("min : {}, train : {}".format(min_samples_of_classes, nb_train_by_classes))

        # Create the test and training dataset
        images_df = pd.DataFrame(images, columns=['path', 'class'])
        train_df = pd.DataFrame(columns=images_df.columns)
        test_df = images_df.copy()

        for c in range(nb_classes):
            c_df = images_df[images_df["class"] == c]

            not_test = list(np.random.choice(c_df.index, nb_train_by_classes, replace=False))

            train_df = train_df.append(c_df.loc[not_test])
            test_df = test_df.drop(not_test)

        x_train = [tuple(l) for l in train_df[['path', 'class']].values.tolist()]
        x_test = [tuple(l) for l in test_df[['path', 'class']].values.tolist()]

        # Just to see if the different classes are well distributed according to the number of samples
        print("train : ", Counter([image[1] for image in x_train]).most_common())
        print("test : ", Counter([image[1] for image in x_test]).most_common())

        print("Size of training set {}".format(len(x_train)))
        print("Size of testing set {}".format(len(x_test)))

        if not zsl:
            class_not_train = []
        elif original_split:
            # original split
            class_not_train = ["chimpanzee", "giant+panda", "leopard", "persian+cat", "pig",
                               "hippopotamus", "humpback+whale", "raccoon", "rat", "seal"]
        else:
            # proposed split (by https://arxiv.org/pdf/1707.00600.pdf)
            # avoid imagenet classes (harder)
            class_not_train = ["sheep", "dolphin", "bat", "seal", "blue+whale", "rat",
                               "horse", "walrus", "giraffe", "bobcat"]

        # preparing sampler for ZERO SHOT LEARNING
        not_train = [class_to_idx[nt] for nt in class_not_train]
        rate_targets_not_train = 0
        nb_targets_not_train = int(nb_classes * rate_targets_not_train)
        not_train.sort()

        # not all the classes have the same number of images, so we have to use this counter
        counter_classes = Counter([t for p, t in x_train])
        nb_trained = sum([n for c, n in counter_classes.most_common() if c not in not_train])

        print("targets not trained : {}, len : {}".format(not_train, len(not_train)))
        # we put to 0 the chance to take an image from the not_train list of classes
        weights = [0 if t in not_train else 1 / (counter_classes[t] * (nb_classes - nb_targets_not_train))
                   for p, t in x_train]

        # create the sampler for zero shot learning
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=nb_trained,
            replacement=False)

        return data_path, classes, class_to_idx, x_train, x_test, train_sampler, nb_classes, not_train



