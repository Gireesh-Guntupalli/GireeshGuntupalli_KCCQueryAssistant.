import pandas as pd
from modules.data_preprocessing import preprocess_dataframe


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print("🔹 Dataset shape:", df.shape)
    df_cleaned = preprocess_dataframe(df)
    return df_cleaned
