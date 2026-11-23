
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir = os.path.join(
    BASE_DIR,
    'dataset',
    'classification_multitask'
)

class_names = [
    'crocodile',
    'dragibus',
    'fraise',
    'oufe',
    'ourson',
    'schtroumpf'
]

color_names = [
    'blanc',
    'bleu',
    'orange',
    'or',
    'rouge',
    'vert'
]

class CandyDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.images = []

        for classe in class_names:
            chemin_classe = os.path.join(root_dir, classe)

            for couleur in color_names:
                chemin_couleur = os.path.join(
                    chemin_classe,
                    couleur
                )

                if not os.path.isdir(chemin_couleur):
                    continue

                for fichier in os.listdir(chemin_couleur):

                    if fichier.lower().endswith(
                        ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
                    ):
                        chemin = os.path.join(chemin_couleur, fichier)

                        label_type = class_names.index(classe)
                        label_couleur = color_names.index(couleur)

                        self.images.append(
                            (chemin, label_type, label_couleur)
                        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        chemin, label_type, label_couleur = self.images[index]

        image = Image.open(chemin).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor(label_type, dtype=torch.long),
            torch.tensor(label_couleur, dtype=torch.long)
        )


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
    ])
}

image_datasets = {
    x: CandyDataset(
        os.path.join(data_dir, x),
        data_transforms[x]
    )
    for x in ['train', 'val', 'test']
}

dataloaders = {
    x: DataLoader(
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

print("\nClasses:", class_names)
print("Colors:", color_names)
print("Dataset sizes:", dataset_sizes)

# Vérification d'un batch
images, labels_type, labels_couleur = next(iter(dataloaders['train']))

print("\n========== BATCH TEST ==========")
print("Images shape:", images.shape)
print("Labels type:", labels_type)
print("Labels couleur:", labels_couleur)
