import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


DATA_PATH = "data"
train_dir = os.path.join(DATA_PATH, "train")
test_dir  = os.path.join(DATA_PATH, "test")

base_model = VGG16(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

print(model.summary())

def extract_feature(image_path):
    img = load_img(image_path, target_size=(224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feature = model.predict(arr, verbose=0)
    return feature[0]

def extract_dataset_features(directory):
    X = []
    y = []

    class_names = [
        d for d in sorted(os.listdir(directory)) 
        if os.path.isdir(os.path.join(directory, d))
    ]

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(directory, class_name)

        print(f"\nExtracting → {class_name}")

        for img_file in os.listdir(class_folder):

            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_folder, img_file)

            try:
                feat = extract_feature(img_path)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print("Skipped:", img_path, "| Error:", e)

    return np.array(X), np.array(y), class_names

print("Extracting TRAIN features...")
X_train, y_train, classes = extract_dataset_features(train_dir)

print("Extracting TEST features...")
X_test, y_test, _ = extract_dataset_features(test_dir)

print("Final Shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


os.makedirs("features", exist_ok=True)
np.save("features/X_train.npy", X_train)
np.save("features/y_train.npy", y_train)
np.save("features/X_test.npy", X_test)
np.save("features/y_test.npy", y_test)
np.save("features/classes.npy", classes)

print("Feature extraction finished and saved!")
