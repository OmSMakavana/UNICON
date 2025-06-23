from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import random
import os
import argparse
import numpy as np
from PreResNet_cifar import *
import dataloader_cifar as dataloader
from math import log2
from Contrastive_loss import *
from memory_selector import KMeansMemorySelector

import collections.abc
from collections.abc import MutableMapping


## For plotting the logs
import wandb
# Remove explicit API key login - let wandb use the default credentials
# wandb.login(key="c32bf6afc559dde97ed51a342fd948131b5f5697", relogin=True)

## Arguments to pass 
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=30, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=350, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--d_u',  default=0.7, type=float)
parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
parser.add_argument('--metric', type=str, default = 'JSD', help='Comparison Metric')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool, help = 'Resume from the warmup checkpoint')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data/cifar10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--loss_type', default='ce', choices=['ce', 'gce', 'gce_cutmix'], help='Loss type: ce | gce | gce_cutmix')
parser.add_argument('--use_kmeans', action='store_true', help='Enable KMeans-based memory selection')
parser.add_argument('--kmeans_clusters', type=int, default=100, help='Number of clusters for KMeans selection')
args = parser.parse_args()

## Initialize wandb
wandb.init(
    project="UNICON-CIFAR", # Changed to a more generic project name
    # Let wandb use the default entity from ~/.netrc
    # entity="ommakavana15",
    name=f"{args.dataset}_{args.noise_mode}_{args.r}_seed{args.seed}_kmeans{args.use_kmeans}",
    config={
        "dataset": args.dataset,
        "noise_mode": args.noise_mode,
        "noise_ratio": args.r,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_u": args.lambda_u,
        "lambda_c": args.lambda_c,
        "alpha": args.alpha,
        "T": args.T,
        "loss_type": args.loss_type,
        "kmeans_clusters": args.kmeans_clusters if args.use_kmeans else None,
        "epochs": args.num_epochs,
        "use_kmeans": args.use_kmeans,
        "resume": args.resume,
        "seed": args.seed,
        "d_u": args.d_u,
        "tau": args.tau,
    }
)

## Initialize KMeans selector if enabled
if args.use_kmeans:
    kmeans_selector = KMeansMemorySelector(num_clusters=args.kmeans_clusters)
else:
    kmeans_selector = None

## GPU Setup 
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

## Download the Datasets
if args.dataset== 'cifar10':
    torchvision.datasets.CIFAR10(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR10(args.data_path,train=False, download=True)
else:
    torchvision.datasets.CIFAR100(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR100(args.data_path,train=False, download=True)

## Checkpoint Location
model_save_loc = os.path.join('checkpoint', f"{args.dataset}_{args.noise_mode}_{args.r}_seed{args.seed}")
os.makedirs(model_save_loc, exist_ok=True)

## Log files
stats_log = open(os.path.join(model_save_loc, f'{args.dataset}_{args.r}_{args.noise_mode}_stats.txt'), 'a')
test_log = open(os.path.join(model_save_loc, f'{args.dataset}_{args.r}_{args.noise_mode}_acc.txt'), 'a')
test_loss_log = open(os.path.join(model_save_loc, 'test_loss.txt'), 'a')
train_acc = open(os.path.join(model_save_loc, 'train_acc.txt'), 'a')
train_loss = open(os.path.join(model_save_loc, 'train_loss.txt'), 'a')

print(f"\n📁 Checkpoint directory: {model_save_loc}")
print(f"📄 Checkpoint files:")
print(f"   - Net1: {os.path.join(model_save_loc, f'{args.dataset}_{args.noise_mode}_{args.r}_Net1_checkpoint.pth')}")
print(f"   - Net2: {os.path.join(model_save_loc, f'{args.dataset}_{args.noise_mode}_{args.r}_Net2_checkpoint.pth')}\n")

# GCE Loss function
def gce_loss(logits, targets, q=0.7):
    # logits: (batch, num_classes), targets: (batch,) (class indices)
    probs = torch.softmax(logits, dim=1)
    targets_onehot = torch.zeros_like(probs).scatter_(1, targets.view(-1, 1), 1)
    gce = (1 - torch.sum(targets_onehot * probs ** q, dim=1)) / q
    return gce.mean()

# CutMix helper
def cutmix(inputs, targets, alpha=1.0):
    '''
    inputs: (batch, ...), targets: (batch,) (class indices)
    returns: mixed_inputs, mixed_targets
    '''
    batch_size = inputs.size(0)
    indices = torch.randperm(batch_size).to(inputs.device)
    shuffled_inputs = inputs[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    mixed_inputs = inputs.clone()
    mixed_inputs[:, :, bbx1:bbx2, bby1:bby2] = shuffled_inputs[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))
    return mixed_inputs, targets, shuffled_targets, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# SSL-Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net2.eval() # Freeze one network and train the other
    net.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1

    ## Loss statistics
    loss_x = 0
    loss_u = 0
    loss_scl = 0
    loss_ucl = 0
    total = 0
    correct = 0

    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)
        
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()
        
        # --- CutMix for labeled/unlabeled if requested ---
        if args.loss_type == 'gce_cutmix':
            # For labeled
            mixed_inputs_x3, targets_x3, shuffled_targets_x3, lam_x3 = cutmix(inputs_x3, torch.argmax(labels_x, dim=1))
            mixed_inputs_x4, targets_x4, shuffled_targets_x4, lam_x4 = cutmix(inputs_x4, torch.argmax(labels_x, dim=1))
            # For unlabeled (use pseudo-labels)
            with torch.no_grad():
                # Label co-guessing of unlabeled samples
                _, outputs_u11 = net(inputs_u)
                _, outputs_u12 = net(inputs_u2)
                _, outputs_u21 = net2(inputs_u)
                _, outputs_u22 = net2(inputs_u2)            
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
                ptu = pu**(1/args.T)
                targets_u = ptu / ptu.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()
                pseudo_labels_u = torch.argmax(targets_u, dim=1)
            mixed_inputs_u3, targets_u3, shuffled_targets_u3, lam_u3 = cutmix(inputs_u3, pseudo_labels_u)
            mixed_inputs_u4, targets_u4, shuffled_targets_u4, lam_u4 = cutmix(inputs_u4, pseudo_labels_u)
            # Replace all_inputs and all_targets for MixMatch
            all_inputs  = torch.cat([mixed_inputs_x3, mixed_inputs_x4, mixed_inputs_u3, mixed_inputs_u4], dim=0)
            # For targets, use one-hot mix for labeled, and for unlabeled use pseudo-labels
            all_targets = torch.cat([
                lam_x3 * F.one_hot(targets_x3, args.num_class) + (1-lam_x3) * F.one_hot(shuffled_targets_x3, args.num_class),
                lam_x4 * F.one_hot(targets_x4, args.num_class) + (1-lam_x4) * F.one_hot(shuffled_targets_x4, args.num_class),
                lam_u3 * F.one_hot(targets_u3, args.num_class) + (1-lam_u3) * F.one_hot(shuffled_targets_u3, args.num_class),
                lam_u4 * F.one_hot(targets_u4, args.num_class) + (1-lam_u4) * F.one_hot(shuffled_targets_u4, args.num_class)
            ], dim=0).float()
        else:
            # --- original MixMatch logic ---
            with torch.no_grad():
                # Label co-guessing of unlabeled samples
                _, outputs_u11 = net(inputs_u)
                _, outputs_u12 = net(inputs_u2)
                _, outputs_u21 = net2(inputs_u)
                _, outputs_u22 = net2(inputs_u2)            
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
                ptu = pu**(1/args.T)
                targets_u = ptu / ptu.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()
            # Label refinement
            _, outputs_x  = net(inputs_x)
            _, outputs_x2 = net(inputs_x2)            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T)    ## Temparature sharpening 
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)           
            targets_x = targets_x.detach()
            all_inputs  = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b   = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        ## Mixup
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
        mixed_input  = l * input_a  + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
        _, logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]

        # --- Loss selection ---
        if args.loss_type == 'gce' or args.loss_type == 'gce_cutmix':
            # For labeled: use GCE loss
            true_labels = torch.argmax(mixed_target[:batch_size*2], dim=1)
            Lx = gce_loss(logits_x, true_labels)
            Lu = 0.0
            lamb = 0.0
        else:
            # For labeled: use original loss
            Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)

        ## Unsupervised Contrastive Loss
        f1, _ = net(inputs_u3)
        f2, _ = net(inputs_u4)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # Debug prints
        print(f"[DEBUG][Epoch {epoch}] Clean memory count: {len(clean_idx) if 'clean_idx' in locals() else 'N/A'}")
        print(f"[DEBUG][Epoch {epoch}] Pseudo-labels high confidence count: {((px1 * py).sum(1) > args.p_threshold).sum().item() if 'px1' in locals() and 'py' in locals() else 'N/A'}")
        try:
            print(f"[DEBUG] loss_u: {loss_u}, loss_c: {loss_simCLR}")
        except NameError:
            print(f"[DEBUG] loss_u: {loss_u}, loss_c: N/A")

        loss_simCLR = contrastive_criterion(features)

        ## Regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        ## Total Loss
        loss = Lx + lamb * Lu + args.lambda_c*loss_simCLR + penalty

        ## Accumulate Loss
        loss_x += Lx.item() if isinstance(Lx, torch.Tensor) else Lx
        loss_u += Lu.item() if isinstance(Lu, torch.Tensor) else Lu
        # loss_ucl += loss_simCLR.item()

        # Compute gradient and Do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy for labeled data
        with torch.no_grad():
            _, pred = torch.max(logits_x, 1)
            _, true_labels = torch.max(mixed_target[:batch_size*2], 1)
            total += true_labels.size(0)
            correct += pred.eq(true_labels).cpu().sum().item()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f Contrastive Loss:%.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss_x/(batch_idx+1), loss_u/(batch_idx+1),  loss_ucl/(batch_idx+1)))
        sys.stdout.flush()

    train_loss_epoch = loss_x/(batch_idx+1)
    train_acc_epoch = 100.*correct/total if total > 0 else 0.0
    return train_loss_epoch, train_acc_epoch


## For Standard Training 
def warmup_standard(epoch,net,optimizer,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _, outputs = net(inputs)               
        # --- Loss selection ---
        if args.loss_type == 'gce' or args.loss_type == 'gce_cutmix':
            loss = gce_loss(outputs, labels)
        else:
            loss    = CEloss(outputs, labels)    
        if args.noise_mode=='asym':     # Penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        else:   
            L = loss
        L.backward()  
        optimizer.step()                

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

## For Training Accuracy
def warmup_val(epoch,net,optimizer,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    total = 0
    correct = 0
    loss_x = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
            inputs, labels = inputs.cuda(), labels.cuda() 
            optimizer.zero_grad()
            _, outputs  = net(inputs)               
            _, predicted = torch.max(outputs, 1)    
            loss    = CEloss(outputs, labels)    
            loss_x += loss.item()                      

            total   += labels.size(0)
            correct += predicted.eq(labels).cpu().sum().item()

    acc = 100.*correct/total
    print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch, acc))  
    
    train_loss.write(str(loss_x/(batch_idx+1)))
    train_acc.write(str(acc))
    train_acc.flush()
    train_loss.flush()

    return acc

## Test Accuracy
def test(epoch,net1,net2):
    net1.eval()
    net2.eval()

    num_samples = 1000
    correct = 0
    total = 0
    loss_x = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 3:
                inputs, targets, index = batch
            else:
                inputs, targets = batch
                index = None
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1 = net1(inputs)
            _, outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
            loss = CEloss(outputs, targets)  
            loss_x += loss.item()

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()  

    acc = 100.*correct/total
    test_loss_epoch = loss_x/(batch_idx+1)
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write(str(acc)+'\n')
    test_loss_log.write(str(test_loss_epoch)+'\n')
    test_loss_log.flush()
    return acc, test_loss_epoch


# KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

## Calculate JSD
def Calculate_JSD(model1, model2, num_samples):  
    JS_dist = Jensen_Shannon()
    JSD   = torch.zeros(num_samples)    

    for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])     
            out2 = torch.nn.Softmax(dim=1).cuda()(model2(inputs)[1])

        ## Get the Prediction
        out = (out1 + out2)/2     

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        dist = JS_dist(out,  F.one_hot(targets, num_classes = args.num_class))
        JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist

    return JSD


## Unsupervised Loss coefficient adjustment 
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

## Choose Warmup period based on Dataset
num_samples = 50000
if args.dataset=='cifar10':
    warm_up = 10
else:
    warm_up = 30

## Call the dataloader
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=4,\
    root_dir=model_save_loc,log=stats_log, noise_file=os.path.join(model_save_loc, f'clean_{args.r}_{args.noise_mode}.npz'))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

## Semi-Supervised Loss
criterion  = SemiLoss()

## Optimizer and Scheduler
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 280, 2e-4)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, 280, 2e-4)

## Loss Functions
CE       = nn.CrossEntropyLoss(reduction='none')
CEloss   = nn.CrossEntropyLoss()
MSE_loss = nn.MSELoss(reduction= 'none')
contrastive_criterion = SupConLoss()

if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

## Define checkpoint filenames
checkpoint_name_1 = f"{args.dataset}_{args.noise_mode}_{args.r}_Net1_checkpoint.pth"
checkpoint_name_2 = f"{args.dataset}_{args.noise_mode}_{args.r}_Net2_checkpoint.pth"
checkpoint_path_1 = os.path.join(model_save_loc, checkpoint_name_1)
checkpoint_path_2 = os.path.join(model_save_loc, checkpoint_name_2)

## Initialize training state
start_epoch = 0
best_acc = 0

## Resume training if requested
if args.resume:
    print("🔍 Looking for checkpoints to resume training...")
    if os.path.isfile(checkpoint_path_1) and os.path.isfile(checkpoint_path_2):
        print(f"✅ Found checkpoints: {checkpoint_path_1} and {checkpoint_path_2}")
        try:
            # Load checkpoint for net1
            checkpoint1 = torch.load(checkpoint_path_1)
            net1.load_state_dict(checkpoint1['model_state_dict'])
            optimizer1.load_state_dict(checkpoint1['optimizer_state_dict'])
            start_epoch = checkpoint1['epoch']
            best_acc = checkpoint1.get('best_acc', 0)
            
            # Load checkpoint for net2
            checkpoint2 = torch.load(checkpoint_path_2)
            net2.load_state_dict(checkpoint2['model_state_dict'])
            optimizer2.load_state_dict(checkpoint2['optimizer_state_dict'])
            
            print(f"✅ Successfully loaded checkpoints from epoch {start_epoch}")
            print(f"   Best accuracy so far: {best_acc:.2f}%")
            print(f"🔁 Resuming training from epoch {start_epoch} to {args.num_epochs}")
        except Exception as e:
            print(f"❌ Error loading checkpoints: {e}")
            print("⚠️ Starting training from scratch")
            start_epoch = 0
    else:
        print("⚠️ No checkpoints found. Starting training from scratch")
        start_epoch = 0

## Training loop
for epoch in range(start_epoch, args.num_epochs + 1):
    test_loader = loader.run(0, 'test')
    eval_loader = loader.run(0, 'eval_train')   
    warmup_trainloader = loader.run(0, 'warmup')
    
    ## Warmup Stage 
    if epoch < warm_up:       
        warmup_trainloader = loader.run(0, 'warmup')

        print('Warmup Model')
        warmup_standard(epoch, net1, optimizer1, warmup_trainloader)   

        print('\nWarmup Model')
        warmup_standard(epoch, net2, optimizer2, warmup_trainloader) 
        
        # Save selected indices after warmup completes
        if epoch == warm_up - 1:  # On the last warmup epoch
            if hasattr(loader, 'dataset') and hasattr(loader.dataset, 'train_idx'):
                pred_idx = loader.dataset.train_idx
                np.savez(os.path.join(model_save_loc, f"clean_{args.r}_{args.noise_mode}.npz"), index=pred_idx)
                print(f"Saved clean indices after warmup to {model_save_loc}")
    
    else:
        ## Calculate JSD values and Filter Rate
        prob = Calculate_JSD(net2, net1, num_samples)           
        threshold = torch.mean(prob)
        if threshold.item()>args.d_u:
            threshold = threshold - (threshold-torch.min(prob))/args.tau
        SR = torch.sum(prob<threshold).item()/num_samples            

        print('Train Net1\n')
        # === KMeans-based memory selection after each epoch ===
        if args.use_kmeans and epoch >= warm_up and kmeans_selector is not None:
            kmeans_selector.reset()
            net1.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_loader):
                    inputs, targets, index = batch[0], batch[1], batch[2]
                    inputs = inputs.cuda()
                    feats = net1.extract_features(inputs)
                    feats = feats.cpu().numpy()
                    targets = targets.cpu().numpy()
                    kmeans_selector.collect(feats, targets, index.cpu().numpy())
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
            
            selected_feats, selected_labels, selected_indices = kmeans_selector.select()
            
            if selected_indices is None or len(selected_indices) < 100:
                print("[KMeans] Invalid selection, falling back to JSD-based selection")
                labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob)
            else:
                print(f"[KMeans] Selected {len(selected_feats)} samples after epoch {epoch}.")
                labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob, selected_indices)
        else:
            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob)

        train_loss_epoch, train_acc_epoch = train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader)    

        ## Calculate JSD values and Filter Rate
        prob = Calculate_JSD(net2, net1, num_samples)           
        threshold = torch.mean(prob)
        if threshold.item()>args.d_u:
            threshold = threshold - (threshold-torch.min(prob))/args.tau
        SR = torch.sum(prob<threshold).item()/num_samples            

        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob)     
        train_loss_epoch2, train_acc_epoch2 = train(epoch, net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader)       
    acc, test_loss_epoch = test(epoch,net1,net2)
    scheduler1.step()
    scheduler2.step()

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch,
        "test_accuracy": acc,
        "test_loss": test_loss_epoch,
        "train_loss": train_loss_epoch if epoch >= warm_up else 0,
        "train_accuracy": train_acc_epoch if epoch >= warm_up else 0,
        "sample_ratio": SR if epoch >= warm_up else 1.0,
        "learning_rate": scheduler1.get_last_lr()[0]
    })

    # === Per-epoch logging ===
    if epoch >= warm_up:
        train_acc.write(f"{epoch},{train_acc_epoch:.6f}\n")
        train_loss.write(f"{epoch},{train_loss_epoch:.6f}\n")
        test_loss_log.write(f"{epoch},{test_loss_epoch:.6f}\n")
        test_log.write(f"{epoch},{acc:.6f}\n")
        train_acc.flush()
        train_loss.flush()
        test_loss_log.flush()
        test_log.flush()

    # Save best model
    if acc > best_acc:
        best_acc = acc
        if epoch < warm_up:
            model_name_1 = f"{args.dataset}_{args.noise_mode}_{args.r}_Net1_warmup.pth"
            model_name_2 = f"{args.dataset}_{args.noise_mode}_{args.r}_Net2_warmup.pth"
        else:
            model_name_1 = f"{args.dataset}_{args.noise_mode}_{args.r}_Net1.pth"
            model_name_2 = f"{args.dataset}_{args.noise_mode}_{args.r}_Net2.pth"            

        print("Save the Model-----")
        torch.save({
            'net': net1.state_dict(),
            'epoch': epoch,
        }, os.path.join(model_save_loc, model_name_1))
        torch.save({
            'net': net2.state_dict(),
            'epoch': epoch,
        }, os.path.join(model_save_loc, model_name_2))

    # Save checkpoint every epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': net1.state_dict(),
        'optimizer_state_dict': optimizer1.state_dict(),
        'best_acc': best_acc,
    }, checkpoint_path_1)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': net2.state_dict(),
        'optimizer_state_dict': optimizer2.state_dict(),
        'best_acc': best_acc,
    }, checkpoint_path_2)

    print(f"✅ Saved checkpoint for epoch {epoch}")

