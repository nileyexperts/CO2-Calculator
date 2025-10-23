
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import great_circle

# Facteurs d'émission (kg CO2e / tonne.km)
EMISSION_FACTORS = {
    "Routier 🚚": 0.100,
    "Aérien ✈️": 0.500,
    "Maritime 🚢": 0.015,
    "Ferroviaire 🚆": 0.030
}

# Configuration de la page
st.set_page_config(page_title="Calculateur CO₂ Multimodal", layout="wide")

# Style personnalisé avec charte graphique
st.markdown("""
    <style>
        body {
            background-color: #DFEDF5;
        }
        .main {
            font-family: 'Arial';
            color: #002E49;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #002E49;
        }
        .segment-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #B89357;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🧭 Calculateur d'empreinte carbone multimodal</div>", unsafe_allow_html=True)
st.markdown("""
Ce calculateur vous permet d'estimer les émissions de CO₂e pour un trajet composé de plusieurs segments de transport.
""")

# Initialisation
geolocator = Nominatim(user_agent="co2_app")

# Nombre de segments
nb_segments = st.number_input("Nombre de segments de transport", min_value=1, max_value=10, value=2, step=1)

segments = []

for i in range(nb_segments):
    st.markdown(f"### Segment {i+1}")
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            origine = st.text_input(f"Origine (segment {i+1})", key=f"origine_{i}")
        with col2:
            destination = st.text_input(f"Destination (segment {i+1})", key=f"destination_{i}")
        with col3:
            mode = st.selectbox(f"Mode de transport (segment {i+1})", list(EMISSION_FACTORS.keys()), key=f"mode_{i}")
        with col4:
            poids_kg = st.number_input(f"Poids (kg) (segment {i+1})", min_value=1.0, value=1000.0, key=f"poids_{i}")

        segments.append({
            "origine": origine,
            "destination": destination,
            "mode": mode,
            "poids_kg": poids_kg
        })

# Calcul
if st.button("Calculer l'empreinte carbone totale"):
    total_emissions = 0.0
    st.markdown("## Résultats par segment")
    for i, seg in enumerate(segments):
        try:
            loc1 = geolocator.geocode(seg["origine"])
            loc2 = geolocator.geocode(seg["destination"])
            if loc1 and loc2:
                coord1 = (loc1.latitude, loc1.longitude)
                coord2 = (loc2.latitude, loc2.longitude)
                distance_km = great_circle(coord1, coord2).km
                poids_tonnes = seg["poids_kg"] / 1000
                facteur = EMISSION_FACTORS[seg["mode"]]
                emission = distance_km * poids_tonnes * facteur
                total_emissions += emission
                st.markdown(f"<div class='segment-box'><strong>Segment {i+1}</strong><br>"
                            f"Origine : {seg['origine']} → Destination : {seg['destination']}<br>"
                            f"Mode : {seg['mode']}<br>"
                            f"Distance estimée : {distance_km:.1f} km<br>"
                            f"Poids : {seg['poids_kg']} kg<br>"
                            f"Émissions : {emission:.2f} kg CO₂e</div>", unsafe_allow_html=True)
            else:
                st.warning(f"Impossible de géocoder les villes du segment {i+1}.")
        except Exception as e:
            st.error(f"Erreur dans le segment {i+1} : {e}")

    st.markdown(f"### 🌍 Émissions totales estimées : **{total_emissions:.2f} kg CO₂e**")
