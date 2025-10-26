
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

st.markdown("---")
st.subheader("🔎 Recherche d'aéroport par code IATA")
iata_dict = load_iata_dict()
iata_input = st.text_input("Entrez un code IATA (3 lettres)", max_chars=3).upper()
if iata_input:
    airport_name = suggest_airport_name(iata_input, iata_dict)
    if airport_name:
        st.success(f"✈️ {iata_input} = {airport_name}")
    else:
        st.warning("Code IATA inconnu ou non référencé.")
else:
    st.info("Veuillez saisir un code IATA pour obtenir le nom de l'aéroport.")
