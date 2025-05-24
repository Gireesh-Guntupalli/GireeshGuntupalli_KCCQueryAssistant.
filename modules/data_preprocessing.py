# # data_preprocessing.py

# import pandas as pd
# import re


# def clean_text(text):
#     if pd.isna(text):
#         return ""
#     text = text.lower().strip()
#     text = re.sub(r"\s+", " ", text)  # remove extra whitespace
#     text = re.sub(r"[^a-z0-9\s.,]", "", text)  # basic character filtering
#     return text


# def is_mostly_numbers(text):
#     if pd.isna(text):
#         return True
#     placeholders = {
#         "",
#         "no answer",
#         "none",
#         "n/a",
#         "na",
#         "not available",
#         "nan",
#         "test call",
#     }
#     text = str(text).strip().lower()
#     if text in placeholders:
#         return True
#     tokens = re.split(r"[\s:;,-]+", text)
#     num_count = sum(token.replace(".", "", 1).isdigit() for token in tokens if token)
#     return len(tokens) > 0 and num_count / len(tokens) > 0.7


# def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
#     # Drop 'Season' column
#     if "Season" in df.columns:
#         df = df.drop(columns=["Season"])

#     # Drop rows with NaN values
#     df = df.dropna()

#     # Clean text columns
#     df["QueryText"] = df["QueryText"].apply(clean_text)
#     df["KccAns"] = df["KccAns"].apply(clean_text)

#     # Normalize categorical columns
#     for col in ["Crop", "DistrictName", "Category", "QueryType", "Sector", "StateName"]:
#         if col in df.columns:
#             df[col] = df[col].astype(str).str.lower().str.strip()

#     # Remove rows where KccAns is mostly numeric or placeholder text
#     df_cleaned = df[~df["KccAns"].apply(is_mostly_numbers)].copy()

#     return df_cleaned


import pandas as pd
import re


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Initial dataset shape: {df.shape}")

    # Drop 'Season' column
    if "Season" in df.columns:
        df.drop(columns=["Season"], inplace=True)
        print("✅ Dropped 'Season' column.")
    else:
        print("⚠️ 'Season' column not found.")

    # Check and drop rows with NaNs
    nan_before = df.isnull().sum().sum()
    print(f"🧪 Total NaN values before dropping: {nan_before}")
    df.dropna(inplace=True)
    nan_after = df.isnull().sum().sum()
    print(f"🧪 Total NaN values after dropping: {nan_after}")
    print(f"✅ Dropped rows with NaN values. New shape: {df.shape}")

    # Normalize text fields
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)  # remove extra whitespace
        text = re.sub(r"[^a-z0-9\s.,]", "", text)  # basic character filtering
        return text

    for col in ["QueryText", "KccAns"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
            print(f"✅ Cleaned text column '{col}'.")
        else:
            print(f"⚠️ Column '{col}' not found for text cleaning.")

    # Normalize categorical columns
    categorical_cols = [
        "Crop",
        "DistrictName",
        "Category",
        "QueryType",
        "Sector",
        "StateName",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            print(f"✅ Normalized categorical column '{col}'.")
        else:
            print(f"⚠️ Column '{col}' not found for normalization.")

    # Remove meaningless KccAns rows
    placeholders = {
        "",
        "no answer",
        "none",
        "n/a",
        "na",
        "not available",
        "nan",
        "test call",
    }

    def is_mostly_numbers(text):
        if pd.isna(text):
            return True
        text = str(text).strip().lower()
        if text in placeholders:
            return True
        tokens = re.split(r"[\s:;,-]+", text)
        num_count = sum(
            token.replace(".", "", 1).isdigit() for token in tokens if token
        )
        return len(tokens) > 0 and num_count / len(tokens) > 0.7

    original_rows = df.shape[0]
    df_cleaned = df[~df["KccAns"].apply(is_mostly_numbers)].copy()
    removed_rows = original_rows - df_cleaned.shape[0]
    print(f"✅ Removed {removed_rows} rows with meaningless 'KccAns'.")
    print(f"Final cleaned dataset shape: {df_cleaned.shape}")

    return df_cleaned
