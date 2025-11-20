import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ---------- LOAD DATA ----------
df = pd.read_csv('single_attribute.csv')

# ---------- FEATURES ----------
audio_features = [
    'zcr', 'rms', 'centroid', 'bw', 'rolloff', 'contrast', 'tonnetz',
    'chroma', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4',
    'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10',
    'mfcc_11', 'mfcc_12', 'mfcc_13', 'duration'
]

# ---------- TARGETS ----------
label_targets = ['diagnosis', 'cough_type', 'severity', 'quality']  # Edit for your data

label_encoders = {}
for target in label_targets:
    # Clean all labels robustly
    df[target] = df[target].apply(lambda x: str(x).strip().lower() if pd.notnull(x) and str(x).strip().lower() not in ["", "nan", "none", "unknown"] else "unknown")
    le = LabelEncoder()
    df[target + '_enc'] = le.fit_transform(df[target])
    label_encoders[target] = le

X = df[audio_features]
Y = df[[target + '_enc' for target in label_targets]]

# ---------- STRATIFIED SPLIT ----------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y['diagnosis_enc']
)

# --- Print baseline class balance ---
for i, target in enumerate(label_targets):
    counts = pd.Series(Y_train.iloc[:, i]).value_counts()
    print(f"\nLabel balance in TRAIN [{target}]:\n{counts}")

# ---------- OVERSAMPLING (SMOTE) PER TARGET LABEL ----------
# Only works with single target; so oversample diagnosis, then use for other multilabel training
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, Y_train['diagnosis_enc'])

print(f"\nAfter SMOTE, TRAIN shape: {X_train_bal.shape}, label balance: {pd.Series(y_train_bal).value_counts()}")

# Replace X_train/Y_train for diagnosis with the oversampled data to build better diagnosis model
rf_diag = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

# Hyperparameter tuning
param_grid = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 200]
}
grid = GridSearchCV(rf_diag, param_grid, cv=5)
grid.fit(X_train_bal, y_train_bal)
print("Best RF params for diagnosis:", grid.best_params_)

best_rf = grid.best_estimator_
y_pred_diag = best_rf.predict(X_test)
le_diag = label_encoders['diagnosis']

print("\n===== DIAGNOSIS (Tuned & Balanced) =====")
print(classification_report(Y_test['diagnosis_enc'], y_pred_diag, target_names=le_diag.classes_, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(Y_test['diagnosis_enc'], y_pred_diag))

# ---------- MULTILABEL RF (other labels, no oversampling) ----------
rf_multi = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
clf = MultiOutputClassifier(rf_multi)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

for idx, target in enumerate(label_targets):
    le = label_encoders[target]
    y_true = Y_test.iloc[:, idx]
    y_pred = Y_pred[:, idx]
    print(f"\n===== {target.upper()} (MultiLabel RF) =====")
    print(classification_report(y_true, y_pred, target_names=[str(c) for c in le.classes_], zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# ---------- FEATURE IMPORTANCE FOR DIAGNOSIS ----------
importances = best_rf.feature_importances_
print("\nTop Features (by avg importance, DIAGNOSIS):")
for name, imp in sorted(zip(audio_features, importances), key=lambda x: -x[1]):
    print(f"{name}: {imp:.4f}")
