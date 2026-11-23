
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    'preparing_data',
    'modele_bonbons.pth'
)

TEST_DIR = os.path.join(
    BASE_DIR,
    'dataset',
    'photo_test'
)

class_names = [
    'crocodile',
    'dragibus',
    'fraise',
    'oufe',
    'ourson',
    'schtroumpf'
]

transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

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

model = model.to(device)
model.eval()

print("\n========== PREDICTIONS ==========")

with torch.no_grad():
    for racine, _, fichiers in os.walk(TEST_DIR):
        for fichier in sorted(fichiers):
            if fichier.lower().endswith(('.jpg', '.jpeg', '.png')):
                chemin = os.path.join(racine, fichier)

                image = Image.open(chemin).convert('RGB')
                image = transformation(image).unsqueeze(0).to(device)

                sorties = model(image)
                probabilites = torch.softmax(sorties, dim=1)

                confiance, prediction = probabilites.max(dim=1)

                classe_predite = class_names[prediction.item()]
                vraie_classe = os.path.basename(racine)

                print(f"\nImage : {fichier}")
                print(f"Vraie classe : {vraie_classe}")
                print(f"Classe prédite : {classe_predite}")
                print(f"Confiance : {confiance.item():.2%}")

                if classe_predite == vraie_classe:
                    print("Résultat : CORRECT")
                else:
                    print("Résultat : ERREUR")
