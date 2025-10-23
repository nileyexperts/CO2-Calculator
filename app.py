
st.markdown("""
<style>
body {
    background-color: #002E49;
    color: white;
    font-family: 'Maven Pro', Arial, sans-serif;
}
.segment-box {
    background-color: #DFEDF5;
    border: 2px solid #BB9357;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    color: #002E49;
}
</style>
""", unsafe_allow_html=True)

import streamlit as st
from opencage.geocoder import OpenCageGeocode
from geopy.distance import great_circle

# Facteurs d'émission en kg CO2e par tonne.km
EMISSION_FACTORS = {
    "Routier 🚚": 0.100,
    "Aérien ✈️": 0.500,
    "Maritime 🚢": 0.015,
    "Ferroviaire 🚆": 0.030
}

# Configuration de l'API OpenCage
API_KEY = st.secrets["OPENCAGE_KEY"]
geocoder = OpenCageGeocode(API_KEY)

st.markdown("""
    <div style='background-color:#002E49;padding:20px;border-radius:10px'>
        <h1 style='color:white;text-align:center;'>🧭 Calculateur d'empreinte carbone multimodal</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .segment-box {
        background-color: #DFEDF5;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
Ce calculateur vous permet d'estimer les émissions de CO₂e pour un trajet multimodal.
Ajoutez plusieurs segments de transport avec leur origine, destination, mode et poids.
""")

segments = []
num_legs = st.number_input("Nombre de segments de transport", min_value=1, max_value=10, value=1, step=1)

for i in range(num_legs):
    st.markdown(f"<div class='segment-box'><h4>Segment {i+1}</h4>", unsafe_allow_html=True)
    origin = st.text_input(f"Origine du segment {i+1}", key=f"origin_{i}")
    destination = st.text_input(f"Destination du segment {i+1}", key=f"dest_{i}")
    mode = st.selectbox(f"Mode de transport du segment {i+1}", list(EMISSION_FACTORS.keys()), key=f"mode_{i}")
    weight_kg = st.number_input(f"Poids transporté (kg) pour le segment {i+1}", min_value=1.0, value=1000.0, key=f"weight_{i}")
    st.markdown("</div>", unsafe_allow_html=True)

    segments.append({
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "weight_kg": weight_kg
    })

if st.button("Calculer l'empreinte carbone totale"):
    total_emissions = 0
    for idx, seg in enumerate(segments):
        try:
            loc1 = geocoder.geocode(seg["origin"])
            loc2 = geocoder.geocode(seg["destination"])
            if not loc1 or not loc2:
                st.error(f"Segment {idx+1} : Lieu introuvable.")
                continue
            coord1 = (loc1[0]['geometry']['lat'], loc1[0]['geometry']['lng'])
            coord2 = (loc2[0]['geometry']['lat'], loc2[0]['geometry']['lng'])
            distance_km = great_circle(coord1, coord2).km
            weight_tonnes = seg["weight_kg"] / 1000
            emissions = distance_km * weight_tonnes * EMISSION_FACTORS[seg["mode"]]
            total_emissions += emissions
            st.success(f"Segment {idx+1} : {distance_km:.1f} km, {emissions:.2f} kg CO₂e")
        except Exception as e:
            st.error(f"Erreur dans le segment {idx+1} : {e}")

    st.markdown(f"""
    <div style='background-color:#B89357;padding:15px;border-radius:10px;margin-top:20px'>
        <h3 style='color:white;text-align:center;'>🌍 Émissions totales estimées : {total_emissions:.2f} kg CO₂e</h3>
    </div>
    """, unsafe_allow_html=True)
