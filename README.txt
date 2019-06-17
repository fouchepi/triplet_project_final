Run the python file run_triplet with 2 arguments :
	* The dataset to use : --dataset=AWA2 (default if nothing) or --dataset=imagenet32 
	* If we don't want to use GPUs even if we can : --force_cpu=False (default if nothing) or --force_cpu=True
Exemple : python run_triplet.py --dataset=AWA2

We can modify some parameters in run_triplet.py for the trinaing :
Example : batch_size, epochs, margin, ... all listed at the top of the file.

************************************************************

Files organisation on the server (important folders):


1) On the /data/data_pierre folder :

	*/checkpoints : 
		- the checkpoints (checkpoints.pth.tar) at the end of each epochs (to delete at the end of the training if no crash of the server, otherwise it will try to restart the next training to this point)
		- the final models at the end of the training (each save name depends of the parameters of the training / Ex : AWA2_1_32_10_0.001_schedul_11_densenet25_kmeans105_imagenet32_mem_eff.pt in order to easily find it)
		- the training and testing descriptors (trX_vect.npy, trY.npy, teX_vect.npy, teY.npy) used for the knn (save them if we want to try different knn/kmeans parameters to classify)
	*/data :
		- /mnist : mnist data (create with torchvision.datasets.MNIST(...))
		- /cifar10 and /cifar100 : cifar10/100 data (create with torchvision.datasets.CIFAR10(...) or CIFAR100(...))
		- /Imagenet32 : downsampled imagenet dataset (10 training files + validation data used for test)
		- /SUN_images : SUN dataset images
		- /CUB200/CUB200_images : CUB200 dataset images
		- /plants2017/plantsTrain : plant dataset images
		- /Animals_with_Attributes2/JPEGImages : AWA2 dataset images

###########################################

2) On /home/pierre/triplet_net_project/main_project/final_code (The final code of the project, without all the test and preliminary datasets/results) :
	* run_triplet.py : main program of the projet (run the training and classification)
	* load_custom.py : load AWA2 dataset (split train / test, zero shot learning ...)
	* evaluate.py : training functions, compute descriptors functions, centroids
	* knn_functions.py : classification functions
	* cifar_dataset.py : create cifar dataset (for easy dataloader + triplets)
	* imagenet_dataset.py : create imagenet dataset
	* triplet_images_dataset.py : create dataset of the form (folder = class)
	* densenet.py : memory efficient densenet from  https://github.com/gpleiss/efficient_densenet_pytorch
	* triplet_network.py : all the different model we can use (main : resnetcustom, densenetcustom, densenet_mem_eff)

