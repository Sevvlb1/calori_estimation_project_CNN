import os
import numpy as np
import joblib

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


print("Training Random Forest")
rf_model.fit(X_train_scaled, y_train)
print("Training completed.")


#METRICS
y_pred_accuracy = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred_accuracy)
print(f"Accuracy Score of Random Forest: {accuracy:.4f}")


y_pred_f1 = rf_model.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred_f1, average="macro")
print(f"F1 Score of Random Forest: {f1:.4f}")


y_pred_precision = rf_model.predict(X_test_scaled)
precision = precision_score(y_test, y_pred_precision, average="macro")
print(f"Precision Score of Random Forest: {precision:.4f}")


y_pred_recall = rf_model.predict(X_test_scaled)
recall = recall_score(y_test, y_pred_recall, average="macro")
print(f"Recall Score of Random Forest: {recall:.4f}")


y_pred_confusion = rf_model.predict(X_test_scaled)
confusion = confusion_matrix(y_test, y_pred_confusion)
print(f"Confusion Matrix of Random Forest: {confusion}")


y_prob_rf = rf_model.predict_proba(X_test_scaled)
roc_auc = roc_auc_score(y_test, y_prob_rf, multi_class="ovr")
print(f"ROC-AUC of Random Forest: {roc_auc:.4f}")




os.makedirs("models", exist_ok=True)
joblib.dump(
    (rf_model, scaler, class_map),
    "models/rf_food_model_independent.joblib"
)

print("Random Forest model saved.")




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
    rf_model, scaler, class_map = joblib.load("models/rf_food_model_independent.joblib")

    feat = extract_feature(image_path)
    feat_scaled = scaler.transform([feat])

    pred_class = rf_model.predict(feat_scaled)[0]
    prob       = rf_model.predict_proba(feat_scaled)[0]

    class_name = class_map[pred_class]
    confidence = np.max(prob)
    calories = calorie_table.get(class_name, "N/A")


    return class_name, calories, confidence



while True:
    path = input("Image path (q to quit): ")
    if path.lower() == "q":
        break

    img_class, predicted_calori, confidence= predict_single_image(path)
    print(f"Predicted Class: {img_class}")
    print(f"Predicted Calories: {predicted_calori}")
    print(f"Confidence: {confidence:.2f}")


