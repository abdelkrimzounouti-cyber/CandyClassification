
import os
import random
import shutil
from collections import defaultdict

SOURCE_DIR = 'dataset/classification'
DEST_DIR = 'dataset/classification_multitask'

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Les couleurs des classes sans sous-dossiers
couleurs_fixes = {
    'fraise': 'rouge',
    'oufe': 'blanc'
}

# Vérifier que le dossier de destination n'existe pas
if os.path.exists(DEST_DIR):
    raise FileExistsError(
        f"Le dossier {DEST_DIR} existe déjà. "
        "Aucune image n'a été modifiée."
    )

# Récupérer les images par combinaison type-couleur
groupes = defaultdict(list)

for classe in sorted(os.listdir(SOURCE_DIR)):
    chemin_classe = os.path.join(SOURCE_DIR, classe)

    if not os.path.isdir(chemin_classe):
        continue

    if classe in couleurs_fixes:
        couleur = couleurs_fixes[classe]

        for fichier in os.listdir(chemin_classe):
            if fichier.lower().endswith(extensions):
                groupes[(classe, couleur)].append(
                    os.path.join(chemin_classe, fichier)
                )

    else:
        for couleur in sorted(os.listdir(chemin_classe)):
            chemin_couleur = os.path.join(chemin_classe, couleur)

            if not os.path.isdir(chemin_couleur):
                continue

            for fichier in os.listdir(chemin_couleur):
                if fichier.lower().endswith(extensions):
                    groupes[(classe, couleur)].append(
                        os.path.join(chemin_couleur, fichier)
                    )

# Créer les dossiers de destination
ensembles = ['train', 'val', 'test']

for ensemble in ensembles:
    os.makedirs(os.path.join(DEST_DIR, ensemble), exist_ok=True)

# Répartition des images
compteurs = defaultdict(int)

for (classe, couleur), fichiers in sorted(groupes.items()):

    random.shuffle(fichiers)

    total = len(fichiers)

    nb_train = int(total * 0.70)
    nb_val = int(total * 0.15)

    repartition = {
        'train': fichiers[:nb_train],
        'val': fichiers[nb_train:nb_train + nb_val],
        'test': fichiers[nb_train + nb_val:]
    }

    for ensemble, images in repartition.items():

        dossier = os.path.join(
            DEST_DIR,
            ensemble,
            classe,
            couleur
        )

        os.makedirs(dossier, exist_ok=True)

        for image in images:
            nom = f"{compteurs[(ensemble, classe, couleur)]}_{os.path.basename(image)}"

            destination = os.path.join(dossier, nom)

            shutil.copy2(image, destination)

            compteurs[(ensemble, classe, couleur)] += 1

# Afficher les résultats
print("\n========== NOUVELLE REPARTITION ==========")

for ensemble in ensembles:

    total = 0

    print(f"\n========== {ensemble.upper()} ==========")

    for (classe, couleur), _ in sorted(groupes.items()):

        nombre = compteurs[(ensemble, classe, couleur)]

        print(f"{classe:12} {couleur:10} : {nombre:4} images")

        total += nombre

    print(f"\nTotal {ensemble}: {total}")

print("\n========== TERMINÉ ==========")
print("Dataset créé dans :", DEST_DIR)
