
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import great_circle

# Facteurs d'émission en kg CO2e par tonne.km
EMISSION_FACTORS = {
    "Routier 🚚": 0.100,
    "Aérien ✈️": 0.500,
    "Maritime 🚢": 0.015,
    "Ferroviaire 🚆": 0.030
}

st.set_page_config(page_title="Calculateur CO2 Multimodal", page_icon="🌍")
st.title("🌿 Calculateur d'Empreinte Carbone Multimodal")
st.markdown("""
Ce calculateur vous permet d'estimer les émissions de CO₂e pour des trajets multimodaux.
Ajoutez plusieurs segments de transport avec origine, destination, mode et poids.
""")

geolocator = Nominatim(user_agent="co2_app")

segments = []
num_legs = st.number_input("Nombre de segments de transport", min_value=1, max_value=10, value=2, step=1)

for i in range(num_legs):
    st.subheader(f"Segment {i+1}")
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input(f"Origine du segment {i+1}", key=f"origin_{i}")
    with col2:
        destination = st.text_input(f"Destination du segment {i+1}", key=f"dest_{i}")

    mode = st.selectbox(f"Mode de transport du segment {i+1}", list(EMISSION_FACTORS.keys()), key=f"mode_{i}")
    weight_kg = st.number_input(f"Poids transporté (kg) pour le segment {i+1}", min_value=1.0, value=1000.0, key=f"weight_{i}")

    segments.append({
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "weight_kg": weight_kg
    })

if st.button("Calculer l'empreinte carbone totale"):
    total_emissions = 0.0
    st.markdown("## 🧾 Résumé des segments")
    for i, seg in enumerate(segments):
        try:
            loc1 = geolocator.geocode(seg["origin"])
            loc2 = geolocator.geocode(seg["destination"])
            if not loc1 or not loc2:
                st.error(f"Impossible de localiser {seg['origin']} ou {seg['destination']}")
                continue
            coord1 = (loc1.latitude, loc1.longitude)
            coord2 = (loc2.latitude, loc2.longitude)
            distance_km = great_circle(coord1, coord2).km
            weight_tonnes = seg["weight_kg"] / 1000
            emissions = distance_km * weight_tonnes * EMISSION_FACTORS[seg["mode"]]
            total_emissions += emissions

            st.markdown(f"**Segment {i+1}** : {seg['origin']} → {seg['destination']}  ")
            st.markdown(f"Mode : {seg['mode']} | Distance : {distance_km:.1f} km | Poids : {seg['weight_kg']} kg  ")
            st.markdown(f"💨 Émissions : **{emissions:.2f} kg CO₂e**")
            st.markdown("---")
        except Exception as e:
            st.error(f"Erreur pour le segment {i+1} : {e}")

    st.success(f"🌍 Émissions totales estimées : {total_emissions:.2f} kg CO₂e")
