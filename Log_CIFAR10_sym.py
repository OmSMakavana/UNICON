import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from dataloader_cifar import cifar_dataset
from PreResNet_cifar import ResNet18
import torchvision.transforms as transforms

model_path = 'checkpoint/cifar10_sym_0.5'
save_loc = model_path

batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Load training dataset
trainset = cifar_dataset(
    dataset='cifar10',
    sample_ratio=1,
    r=0.5,
    noise_mode='sym',
    root_dir='./data/',
    transform=transform_train,
    mode='all'
)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load test dataset
testset = cifar_dataset(
    dataset='cifar10',
    sample_ratio=1,
    r=0.5,
    noise_mode='sym',
    root_dir='./data/',
    transform=transform_test,
    mode='test'
)

test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load model
net1 = ResNet18(num_classes=10).to(device)
checkpoint = torch.load(os.path.join(model_path, 'Net1.pth'), map_location=torch.device('cpu'))
net1.load_state_dict(checkpoint['net'])  # only load the actual model weights
net1.eval()

criterion = nn.CrossEntropyLoss()

def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels, _, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = loss_sum / total
    acc = correct / total
    return avg_loss, acc

# Run evaluation
train_loss, train_acc = evaluate(net1, train_loader)
test_loss, test_acc = evaluate(net1, test_loader)

# Write to logs
with open(os.path.join(save_loc, 'train_loss.txt'), 'w') as f:
    f.write(f"{train_loss:.6f}\n")
with open(os.path.join(save_loc, 'train_acc.txt'), 'w') as f:
    f.write(f"{train_acc:.6f}\n")
with open(os.path.join(save_loc, 'test_loss.txt'), 'w') as f:
    f.write(f"{test_loss:.6f}\n")
with open(os.path.join(save_loc, 'cifar10_0.5_sym_acc.txt'), 'w') as f:
    f.write(f"{test_acc:.6f}\n")

