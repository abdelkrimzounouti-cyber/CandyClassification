
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from preparing_data.load_data_multitask import (
    device,
    dataloaders,
    dataset_sizes,
    class_names,
    color_names
)


# Architecture ResNet50 avec deux sorties
class ResNetMultiTask(nn.Module):

    def __init__(self, nb_classes, nb_couleurs):
        super().__init__()

        self.resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        # Geler les couches pré-entraînées
        for param in self.resnet.parameters():
            param.requires_grad = False

        nb_entrees = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.fc_type = nn.Linear(nb_entrees, nb_classes)
        self.fc_couleur = nn.Linear(nb_entrees, nb_couleurs)

    def forward(self, x):

        features = self.resnet(x)

        sorties_type = self.fc_type(features)
        sorties_couleur = self.fc_couleur(features)

        return sorties_type, sorties_couleur


# Créer le modèle
model = ResNetMultiTask(
    len(class_names),
    len(color_names)
).to(device)

print(model.fc_type)
print(model.fc_couleur)

# Fonction de perte
critere = nn.CrossEntropyLoss()

# Optimiseur : entraîner uniquement les deux nouvelles couches
optimiseur = optim.Adam(
    list(model.fc_type.parameters()) +
    list(model.fc_couleur.parameters()),
    lr=0.001
)

print("Loss function:", critere)

# Mixed precision pour économiser la mémoire GPU
scaler = torch.amp.GradScaler(
    'cuda',
    enabled=(device.type == 'cuda')
)

nb_epochs = 15
meilleure_precision = 0.0

for epoch in range(nb_epochs):

    # Entraînement
    model.train()
    model.resnet.eval()

    loss_total = 0.0

    for images, labels_type, labels_couleur in dataloaders['train']:

        # Envoyer les données sur GPU
        images = images.to(device, non_blocking=True)
        labels_type = labels_type.to(device, non_blocking=True)
        labels_couleur = labels_couleur.to(device, non_blocking=True)

        optimiseur.zero_grad()

        with torch.autocast(
            device_type='cuda',
            dtype=torch.float16,
            enabled=(device.type == 'cuda')
        ):
            sorties_type, sorties_couleur = model(images)

            loss_type = critere(sorties_type, labels_type)
            loss_couleur = critere(sorties_couleur, labels_couleur)

            loss = loss_type + loss_couleur

        scaler.scale(loss).backward()
        scaler.step(optimiseur)
        scaler.update()

        loss_total += loss.item() * images.size(0)

    moyenne_loss_train = loss_total / dataset_sizes['train']

    # Validation
    model.eval()

    bonnes_reponses_type = 0
    bonnes_reponses_couleur = 0
    loss_val_total = 0.0

    with torch.no_grad():

        for images, labels_type, labels_couleur in dataloaders['val']:

            images = images.to(device, non_blocking=True)
            labels_type = labels_type.to(device, non_blocking=True)
            labels_couleur = labels_couleur.to(device, non_blocking=True)

            with torch.autocast(
                device_type='cuda',
                dtype=torch.float16,
                enabled=(device.type == 'cuda')
            ):
                sorties_type, sorties_couleur = model(images)

                loss_type = critere(sorties_type, labels_type)
                loss_couleur = critere(sorties_couleur, labels_couleur)

                loss = loss_type + loss_couleur

            predictions_type = sorties_type.argmax(dim=1)
            predictions_couleur = sorties_couleur.argmax(dim=1)

            bonnes_reponses_type += (
                predictions_type == labels_type
            ).sum().item()

            bonnes_reponses_couleur += (
                predictions_couleur == labels_couleur
            ).sum().item()

            loss_val_total += loss.item() * images.size(0)

    precision_type = bonnes_reponses_type / dataset_sizes['val']
    precision_couleur = bonnes_reponses_couleur / dataset_sizes['val']

    precision_moyenne = (
        precision_type + precision_couleur
    ) / 2

    moyenne_loss_val = loss_val_total / dataset_sizes['val']

    print(
        f"\nEpoch {epoch+1}/{nb_epochs}"
        f"\nTrain Loss: {moyenne_loss_train:.4f}"
        f"\nVal Loss: {moyenne_loss_val:.4f}"
        f"\nVal Accuracy Type: {precision_type:.2%}"
        f"\nVal Accuracy Couleur: {precision_couleur:.2%}"
        f"\nVal Accuracy Moyenne: {precision_moyenne:.2%}"
    )

    # Sauvegarder le meilleur modèle
    if precision_moyenne > meilleure_precision:

        meilleure_precision = precision_moyenne

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'color_names': color_names
            },
            'preparing_data/modele_bonbons_multitask.pth'
        )

        print("Meilleur modèle sauvegardé !")

print("\nEntraînement terminé.")
print(f"Meilleure précision moyenne: {meilleure_precision:.2%}")
