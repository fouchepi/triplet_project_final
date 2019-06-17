import torch.utils.data as data
from PIL import Image
from lxml import etree, html
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import os.path
import operator


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir, nb_classes):
    """
    Function that find 'nb_classes' classes from a root 'dir'.
    It will take the more populated classes first.

    :param dir: the root directory with all the classes dir
    :param nb_classes: the number of classes we want to use
    :return: a list of selected classes and dict between their id and real name
    """
    classes_size = [(d, len(os.listdir(os.path.join(dir, d)))) for d in os.listdir(dir)
                    if os.path.isdir(os.path.join(dir, d))]
    classes_size.sort(key=operator.itemgetter(1), reverse=True)

    classes = [d[0] for d in classes_size[:nb_classes]]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, classes, class_to_idx, extensions):
    """
    Function that create a dataset from a root dir, the classes we want in our dataset
    and the extensions of the images we want to add in our dataset.
    The dataset is a tuple for each images, with its path and its class id

    :param dir: the root dir with the classes dir
    :param classes: the selected classes for the dataset
    :param class_to_idx: dict of classes and their idx
    :param extensions: the extensions of the dataset images
    :return: a list that depicts our dataset
    """
    contents = []
    images = []
    dir = os.path.expanduser(dir)
    for target in tqdm(classes):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):

                if has_file_allowed_extension(fname, extensions) and (fname[0] != '.'):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def make_dataset3(dir, classes, class_to_idx, extensions):
    """
    Same as previous ut only for PLANT dataset when we wanted to limit to type flowers
    (parsing of the xml files that describe the plant images)

    :param dir: the root dir with the classes dir
    :param classes: the selected classes for the dataset
    :param class_to_idx: dict of classes and their idx
    :param extensions: the extensions of the dataset images
    :return: a list that depicts our dataset
    """
    contents = []
    images = []
    dir = os.path.expanduser(dir)
    for target in tqdm(classes):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            fnames = list(set([os.path.splitext(fn)[0] for fn in fnames]))
            for fname in sorted(fnames):

                path = os.path.join(root, fname)
                parser = etree.XMLParser(recover=True)
                tree = etree.parse(path + ".xml", parser=parser)
                content = tree.find("Content").text
                contents.append(content)

                if content == "Flower":
                    item = (path + ".jpg", class_to_idx[target])
                    images.append(item)

    return contents, images

def find_all_classes(dir):
    """
    same as find classes but without number limits

    :param dir: the root directory with all the classes dir
    :return: a list of selected classes and dict between their id and real name
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset4():
    """
    tool function for the plant dataset that creates the plants_contents.csv file
    which allows to create new classes according to the type of the picture (flower, leaf)
    combined to the original class (parsing of the xml files that describe the pictures).
    """
    dir = '/data/data_pierre/data/plants2017/plantsTrain'
    classes, class_to_idx = find_all_classes(dir)

    images = []
    dir = os.path.expanduser(dir)
    for target in tqdm(classes):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            fnames = list(set([os.path.splitext(fn)[0] for fn in fnames]))
            for fname in sorted(fnames):

                path = os.path.join(root, fname)
                parser = etree.XMLParser(recover=True)
                tree = etree.parse(path + ".xml", parser=parser)
                content = tree.find("Content").text
                item = (path + ".jpg", target, content)
                images.append(item)

    images_df = pd.DataFrame(images)
    images_df.columns = ['path', 'target', 'content']
    #images_df = images_df[~pd.isna(images_df['content'])]

    images_df.to_csv("/data/data_pierre/data/plants2017/plants_contents_nan.csv", encoding='utf-8', index=False)

def triplets_inds(dataset, nb_classes):
    """
    Original random triplets indices

    :param targets: labels for each images of the batch (batch_size, 1)
    :return: indices for p, q, n
    """
    q, p, n = [], [], []
    targets = np.array([images[1] for images in dataset])
    dataset = np.array(dataset, dtype=[('foo', 'U500'),('bar', 'i4')])

    for i in list(set(targets)):

        pinds = np.argwhere(targets == i).flatten()
        ninds = np.argwhere(targets != i).flatten()

        pinds_shuffled = pinds.copy()
        np.random.shuffle(pinds_shuffled)
        ninds_shuffle = np.random.choice(ninds, len(pinds))

        q += list(dataset[pinds])
        p += list(dataset[pinds_shuffled])
        n += list(dataset[ninds_shuffle])

    return q, p, n


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# Custom Dataset class, that create a new dataset according to Image Folder organisation,
# but allowing to returns query, positive and negative images at each iteration (for our triplet network).
class TripletDatasetFolder(data.Dataset):

    def __init__(self, root, classes, class_to_idx, samples, train=True, extensions=IMG_EXTENSIONS,
                 loader=default_loader, transform=None, target_transform=None):

        # get the indices for q, p and n images according to the list of samples
        self.triplet_indices = triplets_inds(samples, len(classes))
        self.train = train

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
        print("triplet indices : ", self.triplet_indices[0][0], self.triplet_indices[1][0], self.triplet_indices[2][0])

    def __getitem__(self, index):
        # load and transform the images according to the given transformations
        def load(index, current_samples):
            path, target = current_samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target

        sample_q, target_q = load(index, self.triplet_indices[0])

        # if we train we also need sample and target for positive and negative images.
        if self.train:
            sample_p, target_p = load(index, self.triplet_indices[1])
            sample_n, target_n = load(index, self.triplet_indices[2])

            return (sample_q, sample_p, sample_n), (target_q, target_p, target_n)

        else:
            return sample_q, target_q

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str