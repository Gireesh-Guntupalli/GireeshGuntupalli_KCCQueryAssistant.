import pandas as pd
import uuid

metadata_cols = [
    "StateName",
    "DistrictName",
    "Crop",
    "Category",
    "QueryType",
    "Sector",
    "Year",
    "Month",
    "Day",
]


def create_qa_docs(df: pd.DataFrame) -> pd.DataFrame:
    def make_qa_doc(row):
        return {
            "doc_id": str(uuid.uuid4()),
            "query": row["QueryText"],
            "answer": row["KccAns"],
            "metadata": {col: row[col] for col in metadata_cols},
        }

    qa_docs = df.apply(make_qa_doc, axis=1).tolist()
    return pd.DataFrame(qa_docs)


def export_qa_docs(qa_df: pd.DataFrame, json_path: str, csv_path: str):
    qa_df.to_json(json_path, orient="records", lines=True)
    print(f"✅ Q&A chunks saved to '{json_path}'")
    qa_df.to_csv(csv_path, index=False)
    print(f"✅ Preprocessed Q&A data also saved as '{csv_path}'")


def export_cleaned_raw(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"✅ Raw cleaned data saved as '{path}'")
