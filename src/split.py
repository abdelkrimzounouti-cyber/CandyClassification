
import random
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_DIR / "dataset" / "classification"
OUTPUT_DIR = PROJECT_DIR / "dataset" / "classification_split_v2"
# Paramètres
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

random.seed(RANDOM_SEED)

# Protection contre l'écrasement d'un ancien split
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Le dossier {OUTPUT_DIR} existe déjà. "
        "Choisis un autre nom ou sauvegarde-le avant de continuer."
    )

# Création des dossiers
for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True)

# Parcours des classes
for class_dir in sorted(SOURCE_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    # Récupération récursive des images, y compris dans les sous-dossiers
    images = [
        p for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)

    # Calcul des proportions
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)

    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]

    # Copie physique des images
    for split_name, split_images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images)
    ]:

        destination = OUTPUT_DIR / split_name / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:

            # Préserve l'unicité des noms en tenant compte des sous-dossiers
            relative_path = image_path.relative_to(class_dir)
            new_name = "__".join(relative_path.parts)

            shutil.copy2(
                image_path,
                destination / new_name
            )

    # Affichage des résultats
    print(f"\nClasse : {class_name}")
    print(f"Total : {total}")
    print(f"Train : {len(train_images)} ({len(train_images)/total*100:.1f}%)")
    print(f"Val   : {len(val_images)} ({len(val_images)/total*100:.1f}%)")
    print(f"Test  : {len(test_images)} ({len(test_images)/total*100:.1f}%)")

print("\nSplit terminé avec succès !")
print(f"Dataset créé dans : {OUTPUT_DIR}")
