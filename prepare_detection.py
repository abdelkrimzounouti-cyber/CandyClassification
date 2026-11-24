from pathlib import Path
from zipfile import ZipFile
import random
import shutil

# Chemins
source_images = Path("dataset/detection")
archive = Path.home() / "Downloads/dataset_detection.zip"
output = Path("dataset/detection_yolo")

# Répartition reproductible
random.seed(42)

images = sorted(
    p for p in source_images.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

random.shuffle(images)

split_index = int(len(images) * 0.8)
train_images = images[:split_index]
val_images = images[split_index:]

# Créer les dossiers
for split in ("train", "val"):
    (output / "images" / split).mkdir(parents=True, exist_ok=True)
    (output / "labels" / split).mkdir(parents=True, exist_ok=True)

# Copier images et annotations
with ZipFile(archive) as z:
    for split, selected_images in [
        ("train", train_images),
        ("val", val_images)
    ]:
        for image in selected_images:
            shutil.copy2(
                image,
                output / "images" / split / image.name
            )

            label_name = f"labels/train/{image.stem}.txt"
            label_data = z.read(label_name)

            (output / "labels" / split / f"{image.stem}.txt").write_bytes(
                label_data
            )

# Créer data.yaml
yaml_content = """path: .
train: images/train
val: images/val

names:
  0: crocodile
  1: dragibus
  2: fraise
  3: oufe
  4: ourson
  5: schtroumpf
"""

(output / "data.yaml").write_text(yaml_content)

print("Dataset YOLO créé avec succès !")
print("Images train :", len(train_images))
print("Images val :", len(val_images))
print("Total :", len(images))
print("Emplacement :", output)
