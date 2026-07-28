"""
Subsample train.csv to a target number of hours while respecting speaker distribution.

Expected CSV columns:
ID,duration,wav,spk_id,wrd

What this script does:
----------------------
1. Loads train/dev/test CSVs
2. Computes each speaker's share of total train duration
3. Allocates target duration per speaker using that distribution
4. Randomly samples utterances within each speaker until that speaker's quota is reached
5. Keeps dev.csv and test.csv unchanged
6. Saves new train/dev/test CSVs to a new folder
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
OUTPUT_CSV_DIR = f"{TOKENIZER_PATH}_{TARGET_TRAIN_HOURS}h_spkdist/save"

TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"

SPEAKER_COL = "spk_id"
DURATION_COL = "duration"

# =====================
# LOAD CSVs
# =====================
train = pd.read_csv(os.path.join(INPUT_CSV_DIR, TRAIN_CSV))
dev = pd.read_csv(os.path.join(INPUT_CSV_DIR, DEV_CSV))
test = pd.read_csv(os.path.join(INPUT_CSV_DIR, TEST_CSV))

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

target_seconds = TARGET_TRAIN_HOURS * 3600

# =====================
# BASIC CHECKS
# =====================
required_cols = {SPEAKER_COL, DURATION_COL}
missing = required_cols - set(train.columns)

if missing:
    raise ValueError(f"Missing required columns in train.csv: {missing}")

train[DURATION_COL] = pd.to_numeric(train[DURATION_COL], errors="coerce")

if train[DURATION_COL].isna().any():
    raise ValueError("Some duration values are invalid or NaN.")

original_seconds = train[DURATION_COL].sum()

if target_seconds > original_seconds:
    raise ValueError(
        f"Target hours ({TARGET_TRAIN_HOURS}) is larger than original train hours "
        f"({original_seconds / 3600:.2f})."
    )

# =====================
# COMPUTE SPEAKER QUOTAS
# =====================
speaker_duration = train.groupby(SPEAKER_COL)[DURATION_COL].sum()

speaker_ratio = speaker_duration / original_seconds
speaker_target_seconds = speaker_ratio * target_seconds

# =====================
# SAMPLE PER SPEAKER
# =====================
sampled_parts = []

for spk_id, spk_df in train.groupby(SPEAKER_COL):
    quota = speaker_target_seconds.loc[spk_id]

    spk_shuffled = spk_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    spk_sampled = spk_shuffled[
        spk_shuffled[DURATION_COL].cumsum() <= quota
    ]

    sampled_parts.append(spk_sampled)

train_sampled = pd.concat(sampled_parts, ignore_index=True)

# =====================
# OPTIONAL: FILL SMALL GAP
# =====================
# Because of utterance-level sampling, total duration may be slightly below target.
# Fill remaining time by randomly sampling from utterances not already selected.

remaining_seconds = target_seconds - train_sampled[DURATION_COL].sum()

if remaining_seconds > 0:
    selected_ids = set(train_sampled["ID"])

    leftovers = train[~train["ID"].isin(selected_ids)]
    leftovers = leftovers.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    filler = leftovers[
        leftovers[DURATION_COL].cumsum() <= remaining_seconds
    ]

    train_sampled = pd.concat([train_sampled, filler], ignore_index=True)

# Final shuffle
train_sampled = train_sampled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

# =====================
# SAVE OUTPUT
# =====================
train_sampled.to_csv(os.path.join(OUTPUT_CSV_DIR, TRAIN_CSV), index=False)
dev.to_csv(os.path.join(OUTPUT_CSV_DIR, DEV_CSV), index=False)
test.to_csv(os.path.join(OUTPUT_CSV_DIR, TEST_CSV), index=False)

# =====================
# REPORT
# =====================
print("Saved new CSVs to:", OUTPUT_CSV_DIR)

print("\nOriginal train:")
print(f"hours: {train[DURATION_COL].sum() / 3600:.2f}")
print(f"utterances: {len(train)}")
print(f"speakers: {train[SPEAKER_COL].nunique()}")

print("\nSampled train:")
print(f"hours: {train_sampled[DURATION_COL].sum() / 3600:.2f}")
print(f"utterances: {len(train_sampled)}")
print(f"speakers: {train_sampled[SPEAKER_COL].nunique()}")

print("\nDev/test unchanged:")
print(f"dev hours: {dev[DURATION_COL].sum() / 3600:.2f}")
print(f"test hours: {test[DURATION_COL].sum() / 3600:.2f}")

# =====================
# SPEAKER DISTRIBUTION CHECK
# =====================
orig_spk_dist = (
    train.groupby(SPEAKER_COL)[DURATION_COL].sum()
    / train[DURATION_COL].sum()
)

sampled_spk_dist = (
    train_sampled.groupby(SPEAKER_COL)[DURATION_COL].sum()
    / train_sampled[DURATION_COL].sum()
)

dist_check = pd.DataFrame({
    "orig_ratio": orig_spk_dist,
    "sampled_ratio": sampled_spk_dist
}).fillna(0)

dist_check["abs_diff"] = (
    dist_check["orig_ratio"] - dist_check["sampled_ratio"]
).abs()

print("\nSpeaker distribution difference:")
print(dist_check["abs_diff"].describe())
