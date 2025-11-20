import pandas as pd
import os

# Load both files
features = pd.read_csv('All Audio Features.csv')
labels = pd.read_csv('COUGHVID OG Labels.csv')

# Extract uuid from file column and normalize
def clean_uuid(raw):
    return os.path.basename(str(raw)).replace('_fixed.wav','').replace('.wav','').strip().lower()

features['uuid'] = features['file'].apply(clean_uuid)
labels['uuid'] = labels['uuid'].astype(str).str.strip().str.lower()

# Remove any uuids in short_files.txt
bad_uuids = set()
if os.path.exists('short_files.txt'):
    with open('short_files.txt') as f:
        for line in f:
            line = line.strip()
            if line:
                bad_uuids.add(clean_uuid(line))

features = features[~features['uuid'].isin(bad_uuids)]
labels = labels[~labels['uuid'].isin(bad_uuids)]

# Deduplicate (sanity check)
features = features.drop_duplicates(subset='uuid')
labels = labels.drop_duplicates(subset='uuid')

# Now merge: only those in BOTH
merged = pd.merge(features, labels, on='uuid', how='inner')

merged.to_csv('features_with_labels.csv', index=False)
print(f"Final merge: {len(merged)} matches. Saved as Merged Documents.csv")
