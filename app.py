
import streamlit as st
from opencage.geocoder import OpenCageGeocode
from geopy.distance import great_circle
# Bouton de réinitialisation
if st.button("🔄 Réinitialiser le formulaire"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

# Ajout de l'image de fond depuis GitHub
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/background.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)

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
        <h1 style='color:white;text-align:center;'> Calculateur d'empreinte carbone multimodal - NILEY EXPERTS</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.segment-box {
    background-color: #DFEDF5;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    border: 2px solid #BB9357;
}
.stButton > button {
    background-color: #BB9357;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
}
.stTextInput > div > input {
    background-color: #DFEDF5;
    border: 2px solid #BB9357;
    border-radius: 5px;
}
.stNumberInput > div > input {
    background-color: #DFEDF5;
    border: 2px solid #BB9357;
    border-radius: 5px;
}
.stSelectbox > div > div {
    background-color: #DFEDF5;
    border: 2px solid #BB9357;
    border-radius: 5px;
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
    origin_input = st.text_input(f"Origine du segment {i+1}", key=f"origin_input_{i}")
    origin_suggestions = geocoder.geocode(origin_input, no_annotations=1, limit=5) if origin_input else []
    origin_options = [result['formatted'] for result in origin_suggestions] if origin_suggestions else []
    origin = st.selectbox(f"Suggestions pour l'origine", origin_options, key=f"origin_select_{i}") if origin_options else origin_input
    dest_input = st.text_input(f"Destination du segment {i+1}", key=f"dest_input_{i}")
    dest_suggestions = geocoder.geocode(dest_input, no_annotations=1, limit=5) if dest_input else []
    dest_options = [result['formatted'] for result in dest_suggestions] if dest_suggestions else []
    destination = st.selectbox(f"Suggestions pour la destination", dest_options, key=f"dest_select_{i}") if dest_options else dest_input
    mode = st.selectbox(f"Mode de transport du segment {i+1}", list(EMISSION_FACTORS.keys()), key=f"mode_{i}")
    if i == 0:
        weight_kg = st.number_input(f"Poids transporté (kg) pour le segment {i+1}", min_value=1.0, value=1000.0, key=f"weight_{i}")
    else:
        weight_kg = st.session_state.get('weight_0', 1000.0)
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
