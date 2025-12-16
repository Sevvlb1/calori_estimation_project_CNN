import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

DATA_PATH = "data"
train_dir = os.path.join(DATA_PATH, "train")
test_dir  = os.path.join(DATA_PATH, "test")


def get_class_distribution(directory):
    labels = []
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])
            labels.append((class_name, count))
    return labels

train_dist = get_class_distribution(train_dir)
test_dist  = get_class_distribution(test_dir)

print("Train Class Distribution:")
for c, n in train_dist:
    print(f"{c}: {n}")

print("Test Class Distribution:")
for c, n in test_dist:
    print(f"{c}: {n}")

classes = [x[0] for x in train_dist]
counts  = [x[1] for x in train_dist]

plt.figure(figsize=(10,5))
plt.bar(classes, counts)
plt.xticks(rotation=45)
plt.title("Training Set Class Distribution")
plt.ylabel("Number of Images")
plt.show()


image_shapes = []

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                image_shapes.append(img.shape[:2])  

image_shapes = np.array(image_shapes)

print("Image Size Statistics:")
print("Min:", image_shapes.min(axis=0))
print("Max:", image_shapes.max(axis=0))
print("Mean:", image_shapes.mean(axis=0))



plt.figure(figsize=(10,6))
i = 1

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    img_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if len(img_files) == 0:
        continue

    img_path = os.path.join(class_path, img_files[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(3, 4, i)
    plt.imshow(img)
    plt.title(class_name)
    plt.axis("off")
    i += 1

    if i > 12:
        break

plt.suptitle("Sample Images From Dataset")
plt.show()
