import pandas as pd
import os

ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))


RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "train.parquet")
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "train_processed.parquet")

def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw parquet file with memory optimization."""
    # We will work on train split
    df = pd.read_parquet("hf://datasets/Dingdong-Inc/FreshRetailNet-50K/data/train.parquet")
    
    # Optimize dtypes to reduce memory footprint
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering and cleaning steps."""

    df['dt'] = pd.to_datetime(df['dt'], yearfirst=True)

    # Removing products provided free of cost
    df = df[df['discount'] != 0]

    # Droping full day stockout data
    df = df[df['stock_hour6_22_cnt'] != 16]

    # Latent Demand Recovery (Simple Mean Filling)
    den = 16 - df['stock_hour6_22_cnt']
    num = df['stock_hour6_22_cnt']
    df['sale_amount'] = df['sale_amount'] + (df['sale_amount'] / den) * num

    drop_cols = [
        "city_id",
        "hours_stock_status",
        "hours_sale",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "stock_hour6_22_cnt"
    ]
    df.drop(columns=drop_cols, inplace=True)

    return df


def save_processed_data(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH):
    """Save processed data for later use."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def get_processed_data(force_reprocess: bool = False) -> pd.DataFrame:
   
    if os.path.exists(PROCESSED_DATA_PATH) and not force_reprocess:
        return pd.read_parquet(PROCESSED_DATA_PATH)

    df = load_raw_data()
    df = transform_data(df)
    save_processed_data(df)
    return df


if __name__ == '__main__':
    df = get_processed_data()
    print(df.head())