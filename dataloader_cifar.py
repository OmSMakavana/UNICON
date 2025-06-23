from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os
from autoaugment import CIFAR10Policy, ImageNetPolicy
from torchnet.meter import AUCMeter
import torch.nn.functional as F 
from Asymmetric_Noise import *
from sklearn.metrics import confusion_matrix



## If you want to use the weights and biases 
# import wandb
# wandb.init(project="noisy-label-project", entity="....")


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset): 
    def __init__(self, root_dir, train, transform, noise_file='', r=0, noise_mode='', index=None, pred=None, probability=None, log=None, use_kmeans=False, dataset='cifar10', num_classes=10, mode='all', sample_ratio=1.0, prob=None, indices=None):
        # Set dataset root path correctly
        if dataset == 'cifar10':
            self.root = './data/cifar10/'
        else:
            self.root = './data/cifar100/'
            
        # Initialize all required instance attributes
        self.train = train
        if isinstance(transform, list):
            self.transform = transform
        else:
            self.transform = transform
        self.noise_file = noise_file
        self.r = r
        self.noise_mode = noise_mode
        self.index = index
        self.pred = pred
        self.probability = probability
        self.log = log
        self.use_kmeans = use_kmeans
        self.class_ind = {}  # Initialize class indices dictionary
        self.dataset = dataset
        self.num_classes = num_classes
        self.mode = mode
        self.sample_ratio = sample_ratio
        self.root_dir = root_dir
        
        # Initialize data and label attributes
        self.data = None
        self.label = None
        self.noise_label = None
        self.clean_label = None
        self.noise_idx = None
        self.train_data = None
        self.test_data = None
        self.test_label = None
        
        # Accept prob and indices as aliases for probability and index
        if prob is not None:
            self.probability = prob
        if indices is not None:
            self.index = indices
        
        if self.train:
            # Load noise data if available
            if self.noise_file:
                try:
                    noise_data = np.load(self.noise_file)
                    self.noise_label = noise_data['label']
                    self.clean_label = noise_data['clean_label']
                    self.noise_idx = noise_data['noise_idx']
                    print(f"Loaded noise labels from {self.noise_file}")
                except Exception as e:
                    print(f"Error loading noise file: {e}")
                    self.noise_label = None
                    self.clean_label = None
                    self.noise_idx = None
            else:
                self.noise_label = None
                self.clean_label = None
                self.noise_idx = None
                
            # Load training data from original CIFAR files
            if self.dataset == 'cifar10':                                     
                train_data = []
                train_label = []
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(self.root,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                self.data = np.concatenate(train_data)
                self.data = self.data.reshape((50000, 3, 32, 32))
                self.data = self.data.transpose((0, 2, 3, 1))
                self.label = train_label
            elif self.dataset == 'cifar100':    
                train_dic = unpickle('%s/train'%self.root)
                self.data = train_dic['data']
                self.data = self.data.reshape((50000, 3, 32, 32))
                self.data = self.data.transpose((0, 2, 3, 1))
                self.label = train_dic['fine_labels']
            
            # Initialize clean labels if not loaded from file
            if self.noise_label is None:
                self.noise_label = self.label.copy()
                self.clean_label = self.label.copy()
                self.noise_idx = np.zeros(len(self.label), dtype=bool)
                
                # Generate noise labels only if not loaded from file
                if self.r > 0:
                    if self.noise_mode == 'sym':
                        self.symmetric_noise()
                    elif self.noise_mode == 'asym':
                        self.asymmetric_noise()
                    else:
                        raise ValueError(f"Unknown noise mode: {self.noise_mode}")
        else:
            # Use standard CIFAR test loader
            if self.dataset == 'cifar10':                                     
                test_dic = unpickle('%s/test_batch' % self.root)
                self.data = test_dic['data']
                self.data = self.data.reshape((10000, 3, 32, 32))
                self.data = self.data.transpose((0, 2, 3, 1))
                self.label = test_dic['labels']
            elif self.dataset == 'cifar100':    
                test_dic = unpickle('%s/test' % self.root)
                self.data = test_dic['data']
                self.data = self.data.reshape((10000, 3, 32, 32))
                self.data = self.data.transpose((0, 2, 3, 1))
                self.label = test_dic['fine_labels']
            
        # Initialize class indices
        for i in range(self.num_classes):
            self.class_ind[i] = np.where(self.label == i)[0]
            
        # Apply KMeans filtering only if enabled and indices are provided
        if self.use_kmeans and self.index is not None:
            # Convert to numpy array if not already
            if not isinstance(self.index, np.ndarray):
                self.index = np.array(self.index)
            if not isinstance(self.label, np.ndarray):
                self.label = np.array(self.label)
            
            self.data = self.data[self.index]
            self.label = self.label[self.index]
            if self.noise_label is not None:
                if not isinstance(self.noise_label, np.ndarray):
                    self.noise_label = np.array(self.noise_label)
                self.noise_label = self.noise_label[self.index]
            if self.clean_label is not None:
                if not isinstance(self.clean_label, np.ndarray):
                    self.clean_label = np.array(self.clean_label)
                self.clean_label = self.clean_label[self.index]
            if self.noise_idx is not None:
                if not isinstance(self.noise_idx, np.ndarray):
                    self.noise_idx = np.array(self.noise_idx)
                self.noise_idx = self.noise_idx[self.index]
            # Recompute class indices after filtering
            self.class_ind = {}
            for i in range(self.num_classes):
                self.class_ind[i] = np.where(self.label == i)[0]

        # Ensure noise_label is never None if mode depends on it
        if self.mode in ['all', 'labeled', 'unlabeled'] and self.noise_label is None:
            print("[Fix] noise_label was None. Initializing it as clean labels.")
            self.noise_label = np.array(self.label)

    def symmetric_noise(self):
        indices = np.random.permutation(len(self.label))
        for i, idx in enumerate(indices):
            if i < self.r * len(self.label):
                self.noise_label[idx] = np.random.randint(self.num_classes, dtype=np.int32)
                self.noise_idx[idx] = True

    def asymmetric_noise(self):
        for i in range(self.num_classes):
            indices = np.where(self.label == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.r * len(indices):
                    # CIFAR-10 asymmetric noise
                    if self.dataset == 'cifar10':
                        if i == 0:
                            self.noise_label[idx] = 1
                        elif i == 1:
                            self.noise_label[idx] = 0
                        elif i == 2:
                            self.noise_label[idx] = 8
                        elif i == 3:
                            self.noise_label[idx] = 9
                        elif i == 4:
                            self.noise_label[idx] = 7
                        elif i == 5:
                            self.noise_label[idx] = 6
                        elif i == 6:
                            self.noise_label[idx] = 5
                        elif i == 7:
                            self.noise_label[idx] = 4
                        elif i == 8:
                            self.noise_label[idx] = 2
                        elif i == 9:
                            self.noise_label[idx] = 3
                    # CIFAR-100 asymmetric noise
                    elif self.dataset == 'cifar100':
                        self.noise_label[idx] = (i + 1) % self.num_classes
                    self.noise_idx[idx] = True

    def __getitem__(self, index):
        if self.mode == 'test':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

        elif self.mode == 'labeled':
            img, target = self.data[index], self.noise_label[index]
            image = Image.fromarray(img)
            # Support both list of transforms and single transform
            if isinstance(self.transform, list) and len(self.transform) >= 2:
                img1 = self.transform[0](image)
                img2 = self.transform[1](image)
                # Create two more augmentations for consistency
                img3 = self.transform[0](image) if len(self.transform) < 3 else self.transform[2](image)
                img4 = self.transform[1](image) if len(self.transform) < 4 else self.transform[3](image)
            else:
                img1 = img2 = img3 = img4 = self.transform(image)
            
            # Use probability as w_x if available, else use 1.0
            if self.probability is not None:
                w_x = self.probability[index]
                if not isinstance(w_x, torch.Tensor):
                    w_x = torch.tensor(w_x, dtype=torch.float32)
            else:
                w_x = torch.tensor(1.0, dtype=torch.float32)
            return img1, img2, img3, img4, target, w_x

        elif self.mode == 'unlabeled':
            img = self.data[index]
            image = Image.fromarray(img)
            if isinstance(self.transform, list) and len(self.transform) >= 2:
                img1 = self.transform[0](image)
                img2 = self.transform[1](image)
                # Create two more augmentations for consistency
                img3 = self.transform[0](image) if len(self.transform) < 3 else self.transform[2](image)
                img4 = self.transform[1](image) if len(self.transform) < 4 else self.transform[3](image)
            else:
                img1 = img2 = img3 = img4 = self.transform(image)
            return img1, img2, img3, img4

        elif self.mode == 'all':
            img, target = self.data[index], self.noise_label[index]
            img = Image.fromarray(img)
            # For 'all' mode, if transform is a list, use first transform
            if isinstance(self.transform, list):
                img = self.transform[0](img)
            else:
                img = self.transform(img)            
            return img, target, index
           
    def __len__(self):
        return len(self.data)
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        
        # Set number of classes based on dataset
        self.num_classes = 10 if dataset == 'cifar10' else 100
        
        if self.dataset=='cifar10':
            # Base transforms for both weak and strong
            base_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            # Strong transform adds AutoAugment and Cutout
            strong_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.RandomErasing(p=0.5, scale=(0.125, 0.2), ratio=(0.99, 1.0))
            ])

            self.transforms = {
                "warmup": base_transform,
                "unlabeled": [base_transform, strong_transform],
                "labeled": [base_transform, strong_transform],
            }

            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        elif self.dataset=='cifar100':
            # Base transforms for both weak and strong
            base_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

            # Strong transform adds AutoAugment and Cutout
            strong_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                transforms.RandomErasing(p=0.5, scale=(0.125, 0.2), ratio=(0.99, 1.0))
            ])

            self.transforms = {
                "warmup": base_transform,
                "unlabeled": [base_transform, strong_transform],
                "labeled": [base_transform, strong_transform],
            }        
            
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
                   
    def run(self, noise_threshold, mode, prob=None, indices=None):
        if mode == 'warmup':
            all_dataset = cifar_dataset(
                root_dir=self.root_dir,
                train=True,
                transform=self.transforms["warmup"],
                noise_file=self.noise_file,
                r=self.r,
                noise_mode=self.noise_mode,
                use_kmeans=False,  # Never use KMeans during warmup
                dataset=self.dataset,  # Pass dataset parameter
                num_classes=self.num_classes  # Pass number of classes
            )
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True
            )
            return trainloader
                                     
        elif mode == 'train':
            sample_ratio = noise_threshold
            if sample_ratio == 0:
                pred_idx = []
            else:
                if indices is not None:
                    # Ensure indices is a numpy array
                    pred_idx = np.array(indices) if not isinstance(indices, np.ndarray) else indices
                else:
                    pred_idx = np.zeros(int(sample_ratio * len(self.eval_dataset)))
                    class_len = int(sample_ratio * len(self.eval_dataset) / self.num_classes)
                    size_pred = 0
                    for i in range(self.num_classes):
                        class_indices = self.eval_dataset.class_ind[i]
                        if len(class_indices) == 0:
                            continue
                        prob1 = np.argsort(prob[class_indices].cpu().numpy()) if prob is not None else np.arange(len(class_indices))
                        size1 = len(class_indices)
                        try:
                            pred_idx[size_pred:size_pred + class_len] = np.array(class_indices)[prob1[0:class_len].astype(int)].squeeze()
                            size_pred += class_len
                        except:
                            pred_idx[size_pred:size_pred + size1] = np.array(class_indices)
                            size_pred += size1
                    # No need to convert to list

            labeled_dataset = cifar_dataset(
                root_dir=self.root_dir,
                train=True,
                transform=self.transforms["labeled"],
                noise_file=self.noise_file,
                r=self.r,
                noise_mode=self.noise_mode,
                index=pred_idx,
                probability=prob,
                use_kmeans=(indices is not None),  # Use KMeans only if indices are provided
                dataset=self.dataset,  # Pass dataset parameter
                num_classes=self.num_classes,  # Pass number of classes
                mode='labeled'  # Explicitly set mode to 'labeled'
            )

            unlabeled_dataset = cifar_dataset(
                root_dir=self.root_dir,
                train=True,
                transform=self.transforms["unlabeled"],
                noise_file=self.noise_file,
                r=self.r,
                noise_mode=self.noise_mode,
                index=np.array([i for i in range(len(self.eval_dataset)) if i not in pred_idx]),  # Convert to numpy array
                probability=prob,
                use_kmeans=(indices is not None),  # Use KMeans only if indices are provided
                dataset=self.dataset,  # Pass dataset parameter
                num_classes=self.num_classes,  # Pass number of classes
                mode='unlabeled'  # Explicitly set mode to 'unlabeled'
            )

            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True
            )

            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True
            )

            return labeled_trainloader, unlabeled_trainloader                
        
        elif mode == 'test':
            test_dataset = cifar_dataset(
                root_dir=self.root_dir,
                train=False,
                transform=self.transform_test,
                use_kmeans=False,  # Never use KMeans for test set
                dataset=self.dataset,  # Pass dataset parameter
                num_classes=self.num_classes  # Pass number of classes
            )
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            return test_loader
        
        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(
                root_dir=self.root_dir,
                train=True,
                transform=self.transform_test,
                noise_file=self.noise_file,
                r=self.r,
                noise_mode=self.noise_mode,
                use_kmeans=False,  # Never use KMeans for evaluation
                dataset=self.dataset,  # Pass dataset parameter
                num_classes=self.num_classes  # Pass number of classes
            )
            self.eval_dataset = eval_dataset  # <-- Fix: assign eval_dataset to self
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            return eval_loader        
        
