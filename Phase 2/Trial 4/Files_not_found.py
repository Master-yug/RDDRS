import os
import pandas as pd

FILTERED_CSV = r"C:\Users\Tejash\Experiential Learning\Phase 2\Data processing and cleaning Trial 1\Trial 4\Consolidated 1 set filled features+labels.csv"
SOURCE_AUDIO_DIR = r"C:\Users\Tejash\OneDrive\Desktop\FFmpeg-Builds-latest\coughvid_20211012\fixed"

df = pd.read_csv(FILTERED_CSV)
filename_col = "file_features"

required_files = df[filename_col].dropna().astype(str).str.strip().unique()
available_files = set(f for f in os.listdir(SOURCE_AUDIO_DIR) if f.lower().endswith((".wav", ".mp3")))

not_found = []

for f in required_files:
    name = f
    if not name.lower().endswith((".wav", ".mp3")):
        found = False
        for ext in [".wav", ".mp3"]:
            if name + ext in available_files:
                found = True
                break
        if not found:
            not_found.append(name)
    else:
        if name not in available_files:
            not_found.append(name)

print(f"Total not found: {len(not_found)}")
print("First 30 not found:")
for x in not_found[:30]:
    print(x)

# save full list to CSV for inspection
out_path = r"C:\Users\Tejash\Experiential Learning\Phase 2\Data processing and cleaning Trial 1\Trial 4\not_found_files.csv"
pd.DataFrame({"file_features_not_found": not_found}).to_csv(out_path, index=False)
print(f"\nSaved full list to: {out_path}")
