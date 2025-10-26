
import pandas as pd
import streamlit as st

@st.cache_data

def load_iata_dict():
    try:
        df = pd.read_csv("airport-codes.csv")
        df_filtered = df[df['iata_code'].apply(lambda x: isinstance(x, str) and len(x) == 3)]
        return dict(zip(df_filtered['iata_code'], df_filtered['name']))
    except Exception as e:
        st.error(f"Erreur lors du chargement des codes IATA : {e}")
        return {}

def suggest_airport_name(iata_code: str, iata_dict: dict) -> str:
    return iata_dict.get(iata_code.upper(), "")
