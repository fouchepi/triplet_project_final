from load_custom import load_data
from triplet_network import resnetcustom, densenetcustom, densenet_mem_eff
from evaluate import compute_vectors, train_triplet_custom, compute_vectors_imagenet_val, centroids
from knn_functions import knn_classification_top
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
import triplet_images_dataset as tid
import imagenet_dataset as imda
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import torch
import time

######################################################################################

parser = argparse.ArgumentParser(description='Loading file for project')

parser.add_argument('--force_cpu',
                    action='store_true', default=False,
                    help = 'Keep tensors on the CPU, even if cuda is available (default False)')

parser.add_argument('--dataset',
                    type = str, default="AWA2",
                    help = 'change datasets, between AWA2, imagenet32')

#######################################################################################

# Get run information from arguments :
args = parser.parse_args()

if torch.cuda.is_available() and not args.force_cpu:
    cuda_available = True
else :
    cuda_available = False

# if we want to select manually one of the gpu (0 or 1)
#device = torch.device("cuda:0" if cuda_available else "cpu")
device = torch.device("cuda" if cuda_available else "cpu")
print("Device : {}".format(device))
# path to the data on the server
data_dir = '/data/data_pierre/data'
# get the name of the dataset we want to work on from the arguments
dataset = args.dataset

#######################################################################################

# Parameters of the run :
batch_size = 32
epochs = 10
margin = 1
lr = 0.001  # learning rate for the optimizer
scheduler_epoch = 11
out_features = 512  # descriptor size
train = True  # do we train our network
compute_vector = True  # do we have to compute the descriptors for knn (or we use the ones saved)
save_path = '/data/data_pierre/checkpoints/'
nb_clusters = 10
n_knn = 5
imagenet32 = (dataset == "imagenet32") # do we use imagenet
not_train = []
zsl = False  # do zero shot learning for AWA2
original_split = True  # chose the split to do for AWA2 if zsl
network = 'densenet'  # just for the save name
title_comp = '_imagenet32_mem_eff'  # if we want to add more info to the save name
# For memory efficient densenet
depth = 121
growth_rate = 32
small_inputs = True  # if image size 32
efficient = False  # save memory or not

# The parameters of the dataloader (different if gpus)
loader_param = {'batch_size': batch_size, 'shuffle': False}
if cuda_available: loader_param.update({'num_workers': 4, 'pin_memory': True})

# Name of the model, and the different files we save for each run.
# (important parameters in order to find it when we want to load it)
model_save = dataset + '_' + str(margin) + '_' + str(batch_size) + '_' + str(epochs) + '_' \
             + str(lr) + '_schedul_' + str(scheduler_epoch) + '_' + network + str(out_features) \
             + '_kmeans' + str(nb_clusters) + str(n_knn) + title_comp

print("Parameters : {} ".format(model_save))

if imagenet32:
    # more parameters if imagenet32
    image_size = 32
    nb_classes = 1000
    extra = {"ts": None}
    # the model we want to use (with the size of the final descriptors we want to use)
    model_triplet = densenet_mem_eff(out_features, growth_rate, depth, small_inputs, efficient)
else:
    # more parameters if AWA2
    image_size = 224
    data_path, classes, class_to_idx, x_train, x_test, train_sampler, nb_classes, not_train = load_data(data_dir, dataset, zsl, original_split)
    extra = {"dp": data_path, "c": classes, "cti":class_to_idx, "xt": x_train, "ts": train_sampler}
    model_triplet = densenetcustom(out_features, pretrained=True)
    # model_triplet = resnetcustom(out_features, pretrained=True)

print("imagenet {}, imagesize {}: ".format(imagenet32, image_size))
# transformation to apply to each dataset
# (can add other like : RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Use several GPUs if available
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model_triplet = nn.DataParallel(model_triplet)

# put the model on the gpus
model_triplet = model_triplet.to(device)
print("model : ", model_triplet) #long print of the model, not necessary

"""
# best imagenet model with classic classification (descriptors=1000, just one gpus)
save = "/data/data_pierre/data/densenet_results"
save_name = "_imagenet_all3_121_32_50_192"
print("save name : ", save_name)
model_triplet.load_state_dict(torch.load(os.path.join(save, 'model' + save_name + '.dat')))
"""

# the optimizer and the learning rate scheduler we decided to use
optimizer_triplet = optim.SGD(model_triplet.parameters(), lr=lr, momentum=0.9)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer_triplet, step_size=scheduler_epoch, gamma=0.3)

# if we don't want to compute the descriptors for knn that wants to say that we don't need either to do the training
if compute_vector:

    if train:
        start = time.time()
        # train our model
        train_triplet_custom(transform, loader_param, model_triplet, epochs, margin, optimizer_triplet, batch_size,
                             device, lr_scheduler, model_save, imagenet32, **extra)
        print("Total training time : {}".format(time.time() - start))
    else:
        # load the pretrained model
        # if we want to load a precise model we can modify model_save accordingly
        final_model = torch.load(save_path + model_save + '.pt')
        model_triplet.load_state_dict(final_model)
        model_triplet = model_triplet.to(device)

    print("compute train vectors : ")
    if imagenet32:
        # generate the descriptors for training dataset (used in order to find closest neighbors to testing samples)
        trX_vect, trY = compute_vectors_imagenet_val(model_triplet, batch_size, device, transform, loader_param)
        # generate testing data for imagenet32 and dataset
        x_test, y_test = imda.create_data(False)
        test_set = imda.Imagenet32(x_test, y_test, train=False, transform=transform)
    else:
        # generate the descriptors for training dataset (used in order to find closest neighbors to testing samples)
        validation_set = tid.TripletDatasetFolder(data_path, classes, class_to_idx, x_train, train=False, transform=transform)
        validation_loader = DataLoader(validation_set, **loader_param)
        trX_vect, trY = compute_vectors(validation_loader, model_triplet, batch_size, device)
        # generate testing dataset
        test_set = tid.TripletDatasetFolder(data_path, classes, class_to_idx, x_test, train=False, transform=transform)

    print(" train vector size : {}".format(trX_vect.shape))
    print("compute test vectors : ")
    test_loader = DataLoader(test_set, **loader_param)
    # generate testing descriptors to classify with the training ones
    teX_vect, teY = compute_vectors(test_loader, model_triplet, batch_size, device)
    print(" test vector size : {}".format(teX_vect.shape))

    # save the descriptors (can be useful if we want to test different knn parameters / kmeans)
    np.save(save_path + 'trX_vect.npy', trX_vect)
    np.save(save_path + 'trY.npy', trY)
    np.save(save_path + 'teX_vect.npy', teX_vect)
    np.save(save_path + 'teY.npy', teY)
else:
    trX_vect = np.load(save_path + 'trX_vect.npy')
    trY = np.load(save_path + 'trY.npy')
    teX_vect = np.load(save_path + 'teX_vect.npy')
    teY = np.load(save_path + 'teY.npy')

print("Number clusters : {} / knn {}".format(nb_clusters, n_knn))
# find centroids and replace our training descriptors
trX_vect, trY = centroids(trX_vect, trY, nb_classes, c=nb_clusters)
print("train vector size after kmeans: {}".format(trX_vect.shape))

print("knn prediction computation : ")
test_pred, test_pred_top, conf_mat, conf_mat_top = knn_classification_top(trX_vect, trY, teX_vect, teY, n_nei=n_knn)

acc = (teY == test_pred).sum() / len(teY)
print("Accuracy pred normal: {} %".format(acc*100))
acc_top = (teY == test_pred_top).sum() / len(teY)
print("Accuracy pred top5 : {} %".format(acc_top*100))

# compute accuracy for not train classes for zsl
if len(not_train) != 0:
    conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    conf_mat_not_train = np.take(conf_mat_norm.diagonal(), not_train)
    print("conf mat not train : {}".format(conf_mat_not_train))
    acc_not_train = np.mean(conf_mat_not_train)
    print("Accuracy pred not train : {} %".format(acc_not_train))
