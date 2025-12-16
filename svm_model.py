import os
import numpy as np
import joblib

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


DATA_PATH = "data"
train_dir = os.path.join(DATA_PATH, "train")
test_dir  = os.path.join(DATA_PATH, "test")



base_model = VGG16(weights="imagenet")
feature_model = Model(
    inputs=base_model.input,
    outputs=base_model.layers[-2].output  # fc2 → 4096
)

def extract_feature(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = feature_model.predict(x, verbose=0)
    return feature[0]



X_train = np.load("features/X_train.npy")
y_train = np.load("features/y_train.npy")
X_test  = np.load("features/X_test.npy")
y_test  = np.load("features/y_test.npy")
class_map = np.load("features/classes.npy", allow_pickle=True)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma=0.001,
    class_weight="balanced",
    probability=True,
    random_state=42
)

print("Training SVM...")
svm_model.fit(X_train_scaled, y_train)
print("Training completed.")


#METRICS
y_pred_accuracy = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred_accuracy)
print(f"Accuracy Score of SVM: {accuracy:.4f}")


y_pred_f1 = svm_model.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred_f1, average="macro")
print(f"F1 Score of SVM: {f1:.4f}")


y_pred_precision = svm_model.predict(X_test_scaled)
precision = precision_score(y_test, y_pred_precision, average="macro")
print(f"Precision Score of SVM: {precision:.4f}")


y_pred_recall = svm_model.predict(X_test_scaled)
recall = recall_score(y_test, y_pred_recall, average="macro")
print(f"Recall Score of SVM: {recall:.4f}")


y_pred_confusion = svm_model.predict(X_test_scaled)
confusion = confusion_matrix(y_test, y_pred_confusion)
print(f"Confusion Matrix of SVM: {confusion}")


y_prob_svm = svm_model.predict_proba(X_test_scaled)
roc_auc = roc_auc_score(y_test, y_prob_svm, multiclass="ovr")
print(f"ROC-AUC of SVM: {roc_auc:.4f}")



os.makedirs("models", exist_ok=True)
joblib.dump(
    (svm_model, scaler, class_map),
    "models/SVM_food_model.joblib"
)

print("SVM model saved.")



calorie_table = {
    "Bread": 350,
    "Dairy Production": 300,
    "Dessert": 200,
    "Egg": 450,
    "Elma": 550,
    "Fried Food": 150,
    "Meat": 180,
    "Noodles-Pasta": 260,
    "Rice": 400,
    "Seafood": 320,
    "Soup": 120,
    "Vegetable-Fruit": 100
}

def predict_single_image(image_path):
    svm_model, scaler, class_map = joblib.load("models/SVM_food_model.joblib")

    feat = extract_feature(image_path)
    feat_scaled = scaler.transform([feat])

    pred_class = svm_model.predict(feat_scaled)[0]
    prob       = svm_model.predict_proba(feat_scaled)[0]

    class_name = class_map[pred_class]
    confidence = np.max(prob)
    calories = calorie_table.get(class_name, "N/A")


    return class_name, calories, confidence



while True:
    path = input("Image path (q to quit): ")
    if path.lower() == "q":
        break

    class_predicted, cal, confidence = predict_single_image(path)
    print(f"Predicted Class: {class_predicted}")
    print(f"Predicted Calories: {cal}")
    print(f"Confidence: {confidence:.2f}")
