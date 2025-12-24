import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def main(args):
    # -------------------------
    # Load data
    # -------------------------
    data = pd.read_excel(args.input_xlsx)

    if "Source" not in data.columns:
        raise KeyError("Missing required column: 'Source'")
    if "Image" not in data.columns:
        raise KeyError("Missing required column: 'Image'")

    # Keep only required classes (binary: Real vs Syn)
    data = data[data["Source"].isin(["Syn", "Real"])].copy()
    if data.empty:
        raise ValueError("No rows left after filtering Source to ['Syn','Real'].")

    # -------------------------
    # Features + Labels
    # -------------------------
    feature_cols = ["SSIM", "Naturalness", "confidence", "complexity"]
    missing = [c for c in (feature_cols + ["Source", "Image"]) if c not in data.columns]
    if missing:
        raise KeyError(f"Missing columns in file: {missing}")

    X = data[feature_cols].copy()
    y_text = data["Source"].astype(str).copy()
    meta = data[["Image"]].copy()

    # Explicit label mapping: Syn=1 (positive), Real=0
    label_map = {"Real": 0, "Syn": 1}
    bad = sorted(set(y_text.unique()) - set(label_map.keys()))
    if bad:
        raise ValueError(f"Unexpected labels in Source: {bad}")

    y = y_text.map(label_map).astype(int).to_numpy()

    # -------------------------
    # Split (keep meta aligned)
    # -------------------------
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        meta,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed
    )

    # Reset indices for clean export
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    meta_train = meta_train.reset_index(drop=True)
    meta_test = meta_test.reset_index(drop=True)

    # -------------------------
    # Impute + Scale (fit on train only)
    # -------------------------
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Keep imputed (unscaled) feature values for Excel export
    X_train_imp_df = pd.DataFrame(X_train_imp, columns=feature_cols)
    X_test_imp_df = pd.DataFrame(X_test_imp, columns=feature_cols)

    # -------------------------
    # Train SVM (RBF kernel)
    # -------------------------
    svm_clf = SVC(kernel="rbf", probability=True, random_state=args.seed)

    print("\nTraining SVM (RBF kernel)...")
    svm_clf.fit(X_train_scaled, y_train)

    # Predict
    y_pred = svm_clf.predict(X_test_scaled)

    # Prob of positive class (Syn=1)
    proba = svm_clf.predict_proba(X_test_scaled)
    pos_index = list(svm_clf.classes_).index(1)
    y_pred_proba = proba[:, pos_index]

    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    TN, FP, FN, TP = cm.ravel()
    TPR = TP / (TP + FN) if (TP + FN) else 0.0
    FPR = FP / (FP + TN) if (FP + TN) else 0.0
    auc = roc_auc_score(y_test, y_pred_proba)

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"\nAccuracy: {acc * 100:.2f}%")
    print(f"TPR: {TPR:.4f}")
    print(f"FPR: {FPR:.4f}")
    print(f"AUC: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Syn"]))

    # -------------------------
    # Prepare Excel outputs
    # -------------------------
    inv_label = {0: "Real", 1: "Syn"}

    train_df = pd.concat([meta_train, X_train_imp_df], axis=1)
    train_df["Source"] = [inv_label[int(v)] for v in y_train]

    test_df = pd.concat([meta_test, X_test_imp_df], axis=1)
    test_df["True_Source"] = [inv_label[int(v)] for v in y_test]
    test_df["Predicted_Source"] = [inv_label[int(v)] for v in y_pred]
    test_df["Classify"] = np.where(test_df["True_Source"] == test_df["Predicted_Source"], "Accurate", "Wrong")
    test_df["Predicted_Prob_Syn"] = y_pred_proba

    # -------------------------
    # Save to Excel
    # -------------------------
    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as writer:
        train_df.to_excel(writer, sheet_name="training", index=False)
        test_df.to_excel(writer, sheet_name="testing", index=False)

    print(f"\nSaved: {args.output_xlsx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/test SVM (RBF) on SSIM, Naturalness, confidence, complexity.")
    parser.add_argument("--input_xlsx", required=True, help="Input Excel file path")
    parser.add_argument("--output_xlsx", required=True, help="Output Excel file path")
    parser.add_argument("--test_size", type=float, default=0.5, help="Test split ratio (default 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()

    main(args)
