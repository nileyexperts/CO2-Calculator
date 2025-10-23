
import streamlit as st

# Facteurs d'émission approximatifs en kg CO2e par tonne.km
EMISSION_FACTORS = {
    "Aérien": 0.500,
    "Maritime": 0.015,
    "Routier": 0.100,
    "Ferroviaire": 0.030
}

st.title("Calculateur d'empreinte carbone logistique")
st.markdown("""
Ce calculateur estime les émissions de CO₂e (équivalent CO₂) pour une expédition selon :
- le **mode de transport**
- la **distance** (en kilomètres)
- le **poids** de la cargaison (en tonnes)
""")

mode = st.selectbox("Mode de transport", list(EMISSION_FACTORS.keys()))
distance = st.number_input("Distance (en km)", min_value=1.0, value=1000.0)
poids = st.number_input("Poids de la cargaison (en tonnes)", min_value=0.01, value=1.0)

if st.button("Calculer l'empreinte carbone"):
    facteur = EMISSION_FACTORS[mode]
    emission = distance * poids * facteur
    st.success(f"Émissions estimées : {emission:.2f} kg CO₂e")
    st.caption(f"(Facteur utilisé : {facteur} kg CO₂e / tonne.km)")
