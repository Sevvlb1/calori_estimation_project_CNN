import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize


DATA_PATH = "data"
test_dir = os.path.join(DATA_PATH, "test")

#test için preprocess
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False # Etiketlerin sırasını korumak için zorunlu
)


y_true = test_data.classes
n_classes = test_data.num_classes



print("")
print("")
print("")
print("CNN EVALUATION")

cnn_model = load_model("models/food_model.h5")
y_prob_cnn = cnn_model.predict(test_data)
y_pred_cnn = np.argmax(y_prob_cnn, axis=1)



accuracy = accuracy_score(y_true, y_pred_cnn)
precision = precision_score(y_true, y_pred_cnn, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred_cnn, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred_cnn, average='macro', zero_division=0)
confusion = confusion_matrix(y_true, y_pred_cnn)
roc_auc_cnn = roc_auc_score(y_true, y_prob_cnn, multi_class="ovr")


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro Average): {precision:.4f}")
print(f"Recall (Macro Average): {recall:.4f}")
print(f"F1-Score (Macro Average): {f1:.4f}")
print(f"Confusion Matrix: {confusion}")
print(f"ROC-AUC: {roc_auc_cnn:.4f}")


sns.heatmap(confusion, annot=True, fmt="d")
plt.title("CNN Confusion Matrix")
plt.show()


print("")
print("")
print("")
print("SVM EVALUATION")


X_test = np.load("features/X_test.npy")
y_test = np.load("features/y_test.npy")

svm_model, svm_scaler, _ = joblib.load("models/SVM_food_model_independent2.joblib")
X_test_scaled = svm_scaler.transform(X_test)

y_pred_svm = svm_model.predict(X_test_scaled)
y_prob_svm = svm_model.predict_proba(X_test_scaled)



accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='macro', zero_division=0)
recall_svm = recall_score(y_test, y_pred_svm, average='macro', zero_division=0)
f1_svm = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)
confusion_svm = confusion_matrix(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_prob_svm, multi_class="ovr")


print(f"Accuracy: {accuracy_svm:.4f}")
print(f"Precision (Macro Average): {precision_svm:.4f}")
print(f"Recall (Macro Average): {recall_svm:.4f}")
print(f"F1-Score (Macro Average): {f1_svm:.4f}")
print(f"Confusion Matrix: {confusion_svm}")
print(f"ROC-AUC: {roc_auc_svm:.4f}")


sns.heatmap(confusion_svm, annot=True, fmt="d")
plt.title("SVM Confusion Matrix")
plt.show()




print("")
print("")
print("")
print("RANDOM FOREST EVALUATION")


rf_model, rf_scaler, _ = joblib.load("models/rf_food_model_independent.joblib")
X_test_scaled_rf = rf_scaler.transform(X_test)

y_pred_rf = rf_model.predict(X_test_scaled_rf)
y_prob_rf = rf_model.predict_proba(X_test_scaled_rf)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='macro', zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, average='macro', zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, average='macro', zero_division=0)
confusion_rf = confusion_matrix(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf, multi_class="ovr")


print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision (Macro Average): {precision_rf:.4f}")
print(f"Recall (Macro Average): {recall_rf:.4f}")
print(f"F1-Score (Macro Average): {f1_rf:.4f}")
print(f"Confusion Matrix: {confusion_rf}")
print(f"ROC-AUC: {roc_auc_rf:.4f}")


sns.heatmap(confusion_rf, annot=True, fmt="d")
plt.title("Random Forest Confusion Matrix")
plt.show()



y_test_bin = label_binarize(y_test, classes=range(n_classes))

# CNN
y_true_bin_cnn = label_binarize(y_true, classes=range(n_classes))
fpr_cnn, tpr_cnn, _ = roc_curve(y_true_bin_cnn.ravel(), y_prob_cnn.ravel())
auc_cnn = auc(fpr_cnn, tpr_cnn)

# SVM
fpr_svm, tpr_svm, _ = roc_curve(y_test_bin.ravel(), y_prob_svm.ravel())
auc_svm = auc(fpr_svm, tpr_svm)

# RF
fpr_rf, tpr_rf, _ = roc_curve(y_test_bin.ravel(), y_prob_rf.ravel())
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_cnn, tpr_cnn, label=f"CNN (AUC = {auc_cnn:.4f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc_svm:.4f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.4f})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()