import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

def filter_cats_dogs(dataset):
    data = []
    labels = []
    if hasattr(dataset, "targets"):
        labels_list = dataset.targets
    elif dataset.train:
        labels_list = dataset.train_labels
    else:
        labels_list = dataset.test_labels
    
    for i in range(len(dataset)):
        label = labels_list[i]
        if label == 3:
            data.append(dataset.data[i])
            labels.append(0)
        elif label == 5:
            data.append(dataset.data[i])
            labels.append(1)

    return np.array(data), np.array(labels)

print("Filtering dataset for cats and dogs")
train_data, train_labels = filter_cats_dogs(train_dataset_full)
test_data, test_labels = filter_cats_dogs(test_dataset_full)

print(f"Train data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")