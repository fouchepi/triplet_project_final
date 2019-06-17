from sklearn.cluster import KMeans
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
import triplet_images_dataset as tid
import imagenet_dataset as imda
import torch.nn.functional as F
import numpy as np
import torch
import time
import os


def ratio_triplet_loss(anchor, positive, negative, p=2, eps=1e-6, swap=False):
    """
    Variant if the usual triplet loss function.

    :param anchor: descriptor of the input image
    :param positive: descriptor of the pos image
    :param negative: descriptor of the neg image
    :param p: the norm degree for pairwise distance
    :param eps: small value to avoid division by zero for pairwise distance
    :param swap: do we do the query swap technique
    :return: loss value
    """
    d_p = F.pairwise_distance(anchor, positive, p, eps)
    d_n = F.pairwise_distance(anchor, negative, p, eps)

    if swap:
        d_s = F.pairwise_distance(positive, negative, p, eps)
        d_n = torch.min(d_n, d_s)

    exp_sum = torch.exp(d_p) + torch.exp(d_n)
    soft_p = torch.pow(torch.exp(d_p) / exp_sum, 2)
    soft_n = torch.pow(1 - (torch.exp(d_n) / exp_sum), 2)

    losses = soft_p + soft_n

    loss = torch.mean(losses)
    return loss


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False):
    """
    Usual triplet margin loss

    :param anchor: descriptor of the input image
    :param positive: descriptor of the pos image
    :param negative: descriptor of the neg image
    :param margin: margin parameter for the loss
    :param p: the norm degree for pairwise distance
    :param eps: small value to avoid division by zero for pairwise distance
    :param swap: do we do the query swap technique
    :return: loss value
    """

    d_p = F.pairwise_distance(anchor, positive, p, eps)
    d_n = F.pairwise_distance(anchor, negative, p, eps)

    if swap:
        d_s = F.pairwise_distance(positive, negative, p, eps)
        d_n = torch.min(d_n, d_s)

    dist_hinge = torch.clamp(margin + d_p - d_n, min=0.0)

    loss = torch.mean(dist_hinge)
    return loss


def train_triplet_custom(transform, loader_param, model_triplet, epochs, margin, optimizer_triplet, batch_size, device,
                         lr_scheduler, save, imagenet=False, **kwargs):
    """
    Train the triplet network

    :param transform: the transformation to use for the data
    :param loader_param: the parameters for the training data-loaders
    :param model_triplet: the model to train
    :param epochs: the number of epochs
    :param margin: constant used in the triplet loss
    :param optimizer_triplet: the optimizer used to make the optimization step (and update the weights of the network)
    :param batch_size: the number of samples take in one batch
    :param device: where to put the data (gpus / cpu)
    :param lr_scheduler: learning rate scheduler (change over the training epochs)
    :param save: save name
    :param imagenet: if we use imagenet as dataset
    :param kwargs: other arguments for other dataset
    :return:
    """

    print("Training : ")

    # if we have , load it (example if server crash during training)
    # over-write checkpoint at the end of each epoch if exists
    if os.path.isfile('/data/data_pierre/checkpoints/checkpoints.pth.tar'):
        print("Loading checkpoint ! ")
        state = torch.load('/data/data_pierre/checkpoints/checkpoints.pth.tar')
        model_triplet.load_state_dict(state['state_dict'])
        optimizer_triplet.load_state_dict(state['optimizer'])
        first_epoch = state['epoch']
    else:
        first_epoch = 0

    # special epoch for imagenet in order to loop over the 10 files of imagenet
    if imagenet:
        l_epochs = list(range(10)) * int(epochs/10)
    else:
        l_epochs = list(range(epochs))

    print("epochs : ", l_epochs)

    for i, epoch in enumerate(l_epochs[first_epoch:]):
        print("epoch : {}/{}".format(i + 1, epochs))
        start = time.time()
        sum_loss = 0
        model_triplet.train()

        number_train_samples = 0
        train_target = []
        test_i = 0

        if imagenet:
            # create dataset for imagenet
            x_train, y_train = imda.create_data(True, epoch)
            train_set = imda.Imagenet32(x_train, y_train, train=True, transform=transform)
        else:
            # create dataset for AWA2
            train_set = tid.TripletDatasetFolder(kwargs["dp"], kwargs["c"], kwargs["cti"], kwargs["xt"], train=True,
                                                 transform=transform)

        # create dataloader
        train_loader = DataLoader(train_set, sampler=kwargs["ts"], **loader_param)

        for batch_i, (batch_data, batch_targets) in enumerate(train_loader):
            time_batch_start = time.time()
            test_i += 1
            number_train_samples += len(batch_data[0])
            train_target = train_target + list(batch_targets[0].cpu().detach().numpy())

            (q, p, n), (q_y, p_y, n_y) = batch_data, batch_targets

            optimizer_triplet.zero_grad()

            q, p, n = q.to(device), p.to(device), n.to(device)

            realq = model_triplet(q)
            realp = model_triplet(p)
            realn = model_triplet(n)

            #swap => anchor swap if d(a,n) > d(p,n) swap p and a (take as anchor the closest point to n)
            #loss = ratio_triplet_loss(realq, realp, realn, swap=True)
            loss = triplet_margin_loss(realq, realp, realn, margin=margin, swap=True)
            loss.backward()
            sum_loss += loss.data.item()
            optimizer_triplet.step()

            if (batch_i % 10) == 0:
                time_batch = time.time() - time_batch_start
                print("batch : {} | loss: {} | time : {}".format(batch_i*batch_size, loss.data.item(), time_batch))

        time_epoch = time.time() - start
        print("sum loss : {} | time : {} s".format(sum_loss, time_epoch))
        print("number train samples : ", number_train_samples)
        print("number of batches : ", test_i)
        counter = Counter(train_target).most_common()
        counter.sort()
        print("train targets len : ", len(counter))

        print("Save checkpoint")
        state = {'epoch': i,
                 'state_dict': model_triplet.state_dict(),
                 'optimizer': optimizer_triplet.state_dict()}

        torch.save(state, '/data/data_pierre/checkpoints/checkpoints.pth.tar')

        print('-' * 10)
        lr_scheduler.step()

    torch.save(model_triplet.state_dict(), '/data/data_pierre/checkpoints/' + save + '.pt')


def compute_vectors(loader, model_triplet, batch_size, device):
    """
    Return the vector representations (descriptors) of the entire loaded set thanks to the trained model.

    :param loader: can be train or test loader
    :param model_triplet: pre-trained triplet model
    :param batch_size: number of inputs in the batch
    :param device: gpus or cpu to use
    :return: vector representations and corresponding class
    """

    model_triplet.eval()
    final_vect = []
    targets = []

    for batch_i, (batch_data, batch_targets) in enumerate(loader):

        batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)

        x_vect = model_triplet(batch_data)
        x_list = x_vect.cpu().detach().numpy().tolist()
        final_vect = final_vect + x_list

        batch_target = batch_targets.cpu().detach().numpy().tolist()
        targets = targets + batch_target

        if ((batch_i * batch_size) % (batch_size * 100) == 0):
            print("batch # : {}".format(batch_i * batch_size))

    final_vect = np.array(final_vect)
    return final_vect, targets


def compute_vectors_imagenet_val(model_triplet, batch_size, device, transform_test, loader_param):
    """
    Return the vector representations (descriptors) of the entire imagenet training set

    :param model_triplet: pre-trained triplet model
    :param batch_size: number of inputs in the batch
    :param device: gpus or cpu to use
    :param transform_test: the transformations to apply to the images
    :param loader_param: dataloader parameters
    :return: vector representations and corresponding classes
    """
    model_triplet.eval()
    final_vect = []
    targets = []

    # loop over the 10 imagnet files
    for file in range(10):

        x_train, y_train = imda.create_data(True, file)
        val_set = imda.Imagenet32(x_train, y_train, train=False, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(val_set, **loader_param)

        for batch_i, (batch_data, batch_targets) in enumerate(val_loader):

            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)

            x_vect = model_triplet(batch_data)
            x_list = x_vect.cpu().detach().numpy().tolist()
            final_vect = final_vect + x_list

            batch_target = batch_targets.cpu().detach().numpy().tolist()
            targets = targets + batch_target

            if ((batch_i * batch_size) % (batch_size * 100) == 0):
                print("batch # : {}".format(batch_i * batch_size))

    final_vect = np.array(final_vect)
    return final_vect, targets


def centroids(final_vect, targets, nb_classes, c=1):
    """
    Use K-means in order to clustered the descriptors of each classes and returns their centroids

    :param final_vect: the training descriptors to clustered
    :param targets: their classes
    :param nb_classes: the number of different classes
    :param c: the number of centroids (clusters)
    :return: the descriptors of the centroids with their corresponding classes
    """
    centroids = []
    centroids_targets = []
    for i in range(nb_classes):
        idx = np.where(np.array(targets) == i)[0]
        vect_i = np.take(final_vect, idx, axis=0)
        kmeans = KMeans(c).fit(vect_i)
        centers = kmeans.cluster_centers_
        centroids.extend(centers)
        centroids_targets.extend([i] * c)

    print("len targets : ", len(centroids_targets))
    return np.array(centroids), centroids_targets