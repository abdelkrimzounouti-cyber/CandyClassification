
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from preparing_data.load_data import (
    device,
    dataloaders,
    dataset_sizes,
    class_names
)


model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

# Freeze all pretrained layers
for param in model.parameters():
    param.requires_grad = False

nb_entrees = model.fc.in_features

model.fc = nn.Linear(nb_entrees, len(class_names))

# Move model to GPU
model = model.to(device)

print(model.fc)

critere = nn.CrossEntropyLoss()

optimiseur = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

print("Loss function:", critere)

# Mixed precision for GPU memory saving
scaler = torch.amp.GradScaler(
    'cuda',
    enabled=(device.type == 'cuda')
)

nb_epochs = 10

meilleure_precision = 0.0

for epoch in range(nb_epochs):

    # Sur train
    model.eval()
    model.fc.train()

    loss_total = 0.0

    for images, labels in dataloaders['train']:

        # Send images and labels to GPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiseur.zero_grad()

        with torch.autocast(
            device_type='cuda',
            dtype=torch.float16,
            enabled=(device.type == 'cuda')
        ):
            sorties = model(images)
            loss = critere(sorties, labels)

        scaler.scale(loss).backward()
        scaler.step(optimiseur)
        scaler.update()

        loss_total += loss.item() * images.size(0)

    moyenne_loss_train = loss_total / dataset_sizes['train']

    # Vérification val
    model.eval()

    bonnes_reponses = 0
    loss_val_total = 0.0

    with torch.no_grad():

        for images, labels in dataloaders['val']:

            # Send validation data to GPU
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type='cuda',
                dtype=torch.float16,
                enabled=(device.type == 'cuda')
            ):
                sorties = model(images)
                loss = critere(sorties, labels)

            predictions = sorties.argmax(dim=1)

            bonnes_reponses += (
                predictions == labels
            ).sum().item()

            loss_val_total += loss.item() * images.size(0)

    precision = bonnes_reponses / dataset_sizes['val']
    moyenne_loss_val = loss_val_total / dataset_sizes['val']

    print(
        f"\nEpoch {epoch+1}/{nb_epochs}"
        f"\nTrain Loss: {moyenne_loss_train:.4f}"
        f"\nVal Loss: {moyenne_loss_val:.4f}"
        f"\nVal Accuracy: {precision:.2%}"
    )

    # Sauvegarder le meilleur modèle
    if precision > meilleure_precision:
        meilleure_precision = precision

        torch.save(
         model.state_dict(),
         'preparing_data/modele_bonbons.pth' )

        print("Meilleur modèle sauvegardé !")

print("\nEntraînement terminé.")
print(f"Meilleure précision validation: {meilleure_precision:.2%}")
