import pandas as pd

results = {
    "Model": [
        "CNN (MobileNetV2)",
        "Random Forest (VGG16)",
        "SVM (VGG16)"
    ],
    "Accuracy": [
        0.9116,
        0.8089,
        0.6050
    ],
    "Precision (Macro)": [
        0.9171,
        0.8549,
        0.8412
    ],
    "Recall (Macro)": [
        0.9272,
        0.8074,
        0.5860
    ],
    "F1-Score (Macro)": [
        0.9214,
        0.8215,
        0.6377
    ],
    "ROC-AUC (OVR)": [
        0.9955,
        0.9805,
        0.9619
    ]
}


df_results = pd.DataFrame(results)

print("MODEL PERFORMANCE COMPARISON TABLE")
print(df_results.to_string(index=False))


df_results.to_csv("model_comparison_results.csv", index=False)

