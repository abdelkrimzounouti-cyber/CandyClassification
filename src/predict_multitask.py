
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from preparing_data.load_data_multitask import device


# Architecture du modèle
class ResNetMultiTask(nn.Module):

    def __init__(self, nb_classes, nb_couleurs):
        super().__init__()

        self.resnet = models.resnet50(
            weights=None
        )

        nb_entrees = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.fc_type = nn.Linear(nb_entrees, nb_classes)
        self.fc_couleur = nn.Linear(nb_entrees, nb_couleurs)

    def forward(self, x):

        features = self.resnet(x)

        sorties_type = self.fc_type(features)
        sorties_couleur = self.fc_couleur(features)

        return sorties_type, sorties_couleur


# Charger le modèle sauvegardé
MODEL_PATH = 'preparing_data/modele_bonbons_multitask.pth'
TEST_DIR = 'dataset/photo_test'

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

class_names = checkpoint['class_names']
color_names = checkpoint['color_names']

model = ResNetMultiTask(
    len(class_names),
    len(color_names)
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# Prétraitement des images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


print("Device:", device)

if device.type == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0))


print("\n========== MULTITASK PREDICTIONS ==========\n")

with torch.no_grad():

    for i in range(1, 13):

        image_name = f'test{i}.jpg'
        image_path = os.path.join(TEST_DIR, image_name)

        if not os.path.exists(image_path):
            print(f"{image_name} : image introuvable")
            continue

        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        sorties_type, sorties_couleur = model(image)

        probabilites_type = torch.softmax(sorties_type, dim=1)
        probabilites_couleur = torch.softmax(sorties_couleur, dim=1)

        confiance_type, prediction_type = probabilites_type.max(dim=1)
        confiance_couleur, prediction_couleur = probabilites_couleur.max(dim=1)

        nom_type = class_names[prediction_type.item()]
        nom_couleur = color_names[prediction_couleur.item()]

        print(f"Image : {image_name}")
        print(f"Type prédit     : {nom_type}")
        print(f"Confiance type  : {confiance_type.item():.2%}")
        print(f"Couleur prédite : {nom_couleur}")
        print(f"Confiance couleur: {confiance_couleur.item():.2%}")
        print("-" * 45)
