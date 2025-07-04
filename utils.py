import pandas as pd

def load_data():
    df = pd.read_csv("data_2025.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_state_monthly_data(state_code: str):
    df = pd.read_csv("state_wise_monthly_2025.csv")
    df = df[df["State"].str.upper() == state_code.upper()]
    df["Date"] = pd.to_datetime(df["Date"])
    return df
