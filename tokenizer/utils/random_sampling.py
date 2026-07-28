"""
Randomly subsample train.csv to a target number of hours.

This is useful for datasets without speaker IDs or reliable speaker metadata,
such as datasets that only contain audio paths and transcripts.

What this script does:
----------------------
1. Loads train/dev/test CSVs
2. Randomly shuffles train.csv
3. Selects utterances until target hours is reached
4. Keeps dev.csv and test.csv unchanged
5. Saves new train/dev/test CSVs to a new folder
"""

import os
import pandas as pd

# =====================
# CONFIGURATION
# =====================
TARGET_TRAIN_HOURS = 100
SEED = 42

TOKENIZER_PATH = "/path/to/tokenizer/trained/example_language"

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Path does not exist: {TOKENIZER_PATH}")

INPUT_CSV_DIR = f"{TOKENIZER_PATH}/save"
OUTPUT_CSV_DIR = f"{TOKENIZER_PATH}_{TARGET_TRAIN_HOURS}h_random/save"

# =====================
# LOAD CSVs
# =====================
train = pd.read_csv(os.path.join(INPUT_CSV_DIR, "train.csv"))
dev = pd.read_csv(os.path.join(INPUT_CSV_DIR, "dev.csv"))
test = pd.read_csv(os.path.join(INPUT_CSV_DIR, "test.csv"))

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# =====================
# RANDOM SAMPLE TRAIN
# =====================
target_seconds = TARGET_TRAIN_HOURS * 3600

# Shuffle all train utterances
train_shuffled = train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

# Select utterances until target duration is reached
train_sampled = train_shuffled[
    train_shuffled["duration"].cumsum() <= target_seconds
].reset_index(drop=True)

# =====================
# SAVE OUTPUT
# =====================
train_sampled.to_csv(os.path.join(OUTPUT_CSV_DIR, "train.csv"), index=False)
dev.to_csv(os.path.join(OUTPUT_CSV_DIR, "dev.csv"), index=False)
test.to_csv(os.path.join(OUTPUT_CSV_DIR, "test.csv"), index=False)

# =====================
# REPORT
# =====================
print("Saved new CSVs to:", OUTPUT_CSV_DIR)

print("\nOriginal train:")
print(f"hours: {train['duration'].sum() / 3600:.2f}")
print(f"utterances: {len(train)}")

print("\nSampled train:")
print(f"hours: {train_sampled['duration'].sum() / 3600:.2f}")
print(f"utterances: {len(train_sampled)}")

print("\nDev/test unchanged:")
print(f"dev hours: {dev['duration'].sum() / 3600:.2f}")
print(f"test hours: {test['duration'].sum() / 3600:.2f}")
