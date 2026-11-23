
import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    'dataset',
    'classification_split_v2',
    'test'
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    'preparing_data',
    'modele_bonbons.pth'
)

transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    DATA_DIR,
    transformation
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == 'cuda')
)

class_names = test_dataset.classes

print("\nClasses:", class_names)
print("Number of test images:", len(test_dataset))

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

# Move model to GPU
model = model.to(device)

model.eval()

nb = len(class_names)

matrice = [[0] * nb for _ in range(nb)]

bonnes = 0

with torch.no_grad():

    for images, labels in test_loader:

        # Transfer images and labels to GPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Get predictions
        predictions = model(images).argmax(dim=1)

        # Update confusion matrix and accuracy
        for vrai, predit in zip(labels, predictions):

            vrai = vrai.item()
            predit = predit.item()

            matrice[vrai][predit] += 1

            if vrai == predit:
                bonnes += 1


accuracy = bonnes / len(test_dataset)

print("\n========== TEST RESULTS ==========")

print(
    f"\nAccuracy : "
    f"{bonnes}/{len(test_dataset)} = {accuracy:.2%}\n"
)

print("CONFUSION MATRIX")

print("Lignes = vraie classe, colonnes = prédiction\n")

print(" " * 12 + "".join(f"{c[:6]:>8}" for c in class_names))

for i, ligne in enumerate(matrice):
    print(
        f"{class_names[i]:>12}"
        + "".join(f"{v:>8}" for v in ligne)
    )


# Calcul des métriques par classe

print("\n========== CLASSIFICATION REPORT ==========")

for i, classe in enumerate(class_names):

    vrais_positifs = matrice[i][i]

    faux_positifs = sum(matrice[j][i] for j in range(nb)) - vrais_positifs

    faux_negatifs = sum(matrice[i]) - vrais_positifs

    precision = (
        vrais_positifs / (vrais_positifs + faux_positifs)
        if vrais_positifs + faux_positifs > 0 else 0
    )

    rappel = (
        vrais_positifs / (vrais_positifs + faux_negatifs)
        if vrais_positifs + faux_negatifs > 0 else 0
    )

    f1 = (
        2 * precision * rappel / (precision + rappel)
        if precision + rappel > 0 else 0
    )

    print(
        f"{classe:12} "
        f"Precision: {precision:.2%} "
        f"Recall: {rappel:.2%} "
        f"F1-score: {f1:.2%}"
    )
