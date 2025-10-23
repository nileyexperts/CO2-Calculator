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

def suggere_villes(partial_query):
    results = geocoder.geocode(partial_query, limit=5, no_annotations=1, language='fr')
    villes = []
    for r in results:
        nom = r['formatted']
        if nom not in villes:
            villes.append(nom)
    return villes

st.markdown("## 🧭 Calculateur d'empreinte carbone multimodal", unsafe_allow_html=True)
st.markdown("Ce calculateur vous permet d'estimer les émissions de CO₂e pour un trajet multimodal.")

segments = []
num_legs = st.number_input("Nombre de segments de transport", min_value=1, max_value=10, value=1, step=1)

# Option pour reprendre automatiquement le poids du premier segment
st.checkbox("Reprendre automatiquement le poids du premier segment", value=True, key="reuse_weight")

for i in range(num_legs):
    st.markdown(f"##### Segment {i+1}", unsafe_allow_html=True)

    partial_origin = st.text_input(f"Commencez à taper l'origine du segment {i+1}", key=f"partial_origin_{i}")
    suggestions_origin = suggere_villes(partial_origin) if partial_origin else []
    origin = st.selectbox(f"Choisissez l'origine du segment {i+1}", suggestions_origin, key=f"origin_{i}")

    partial_dest = st.text_input(f"Commencez à taper la destination du segment {i+1}", key=f"partial_dest_{i}")
    suggestions_dest = suggere_villes(partial_dest) if partial_dest else []
    destination = st.selectbox(f"Choisissez la destination du segment {i+1}", suggestions_dest, key=f"dest_{i}")

    mode = st.selectbox(f"Mode de transport du segment {i+1}", list(EMISSION_FACTORS.keys()), key=f"mode_{i}")

    if i == 0 or not st.session_state.get("reuse_weight", True):
        weight_kg = st.number_input(f"Poids transporté (kg) pour le segment {i+1}", min_value=1.0, value=1000.0, key=f"weight_{i}")
    else:
        weight_kg = st.session_state.get("weight_0", 1000.0)

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

    st.markdown(f"#### 🌍 Émissions totales estimées : {total_emissions:.2f} kg CO₂e", unsafe_allow_html=True)
