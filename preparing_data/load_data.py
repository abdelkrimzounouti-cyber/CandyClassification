
import torch
from torchvision import datasets, transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),

    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),

    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
}

data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'dataset',
    'classification_split_v2'
)

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        data_transforms[x]
    )
    for x in ['train', 'val', 'test']
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=8,
        shuffle=(x == 'train'),
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    for x in ['train', 'val', 'test']
}

dataset_sizes = {
    x: len(image_datasets[x])
    for x in ['train', 'val', 'test']
}

class_names = image_datasets['train'].classes

print("Class names:", class_names)
print("Dataset sizes:", dataset_sizes)
