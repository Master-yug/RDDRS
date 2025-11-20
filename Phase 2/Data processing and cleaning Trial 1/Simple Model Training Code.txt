import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ---------- SETTINGS ----------
FILENAME = 'single_attribute.csv' # Put your final CSV name here

audio_features = [
    'zcr', 'rms', 'centroid', 'bw', 'rolloff', 'contrast',
    'tonnetz', 'chroma', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4',
    'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10',
    'mfcc_11', 'mfcc_12', 'mfcc_13', 'duration'
]

label_fields = [
    "quality_1","diagnosis_1","quality_2","diagnosis_2",
    "quality_3","diagnosis_3","quality_4","diagnosis_4" # Add other labels as needed
]

# ---------- LOAD DATA ----------
df = pd.read_csv(FILENAME)

# ---------- PICK FILLED LABELS ONLY, IGNORE x ----------
label_list = [
    "diagnosis_1","diagnosis_2","diagnosis_3","diagnosis_4"
]
def choose_first_label(row):
    for col in label_list:
        val = str(row[col]).strip()
        if val and val not in ["", "nan", "None"]:
            return val
    return None

df['diagnosis'] = df.apply(choose_first_label, axis=1)
df_valid = df[df['diagnosis'].notnull()]

print(f"Usable samples: {len(df_valid)}")

# ---------- ENCODE LABEL ----------
le = LabelEncoder()
y = le.fit_transform(df_valid['diagnosis'])

# ---------- FEATURES ----------
X = df_valid[audio_features]

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- TRAIN ----------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ---------- EVALUATE ----------
y_pred = rf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- BONUS: Feature Importance ----------
importances = rf.feature_importances_
for name, imp in sorted(zip(audio_features, importances), key=lambda x: -x[1]):
    print(f"{name}: {imp:.4f}")

# ---------- PREDICT on new audio sample (if you have data) ----------
# new_X = ... # shape (1, n_features)
# rf.predict(new_X)
