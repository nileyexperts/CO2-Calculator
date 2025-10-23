
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import great_circle

# Facteurs d'émission en kg CO2e par tonne.km
EMISSION_FACTORS = {
    "Aérien": 0.500,
    "Maritime": 0.015,
    "Routier": 0.100,
    "Ferroviaire": 0.030
}

st.title("Calculateur d'empreinte carbone multimodal")

st.markdown("""
Ajoutez plusieurs segments de transport (origine, destination, mode, poids en **kilogrammes**) pour estimer les émissions totales de CO₂e.
""")

geolocator = Nominatim(user_agent="co2_app")

segments = []
num_legs = st.number_input("Nombre de segments de transport", min_value=1, max_value=10, value=1, step=1)

for i in range(num_legs):
    st.subheader(f"Segment {i+1}")
    origin = st.text_input(f"Origine du segment {i+1}", key=f"origin_{i}")
    destination = st.text_input(f"Destination du segment {i+1}", key=f"destination_{i}")
    mode = st.selectbox(f"Mode de transport du segment {i+1}", list(EMISSION_FACTORS.keys()), key=f"mode_{i}")
    weight_kg = st.number_input(f"Poids transporté (en kg) pour le segment {i+1}", min_value=1.0, value=1000.0, key=f"weight_{i}")

    if origin and destination:
        try:
            loc1 = geolocator.geocode(origin)
            loc2 = geolocator.geocode(destination)
            if loc1 and loc2:
                coord1 = (loc1.latitude, loc1.longitude)
                coord2 = (loc2.latitude, loc2.longitude)
                distance_km = great_circle(coord1, coord2).km
                factor = EMISSION_FACTORS[mode]
                emissions = distance_km * (weight_kg / 1000) * factor  # conversion kg -> tonnes
                segments.append({
                    "origine": origin,
                    "destination": destination,
                    "mode": mode,
                    "distance_km": distance_km,
                    "poids_kg": weight_kg,
                    "emissions": emissions
                })
            else:
                st.warning(f"Impossible de géocoder {origin} ou {destination}.")
        except:
            st.error("Erreur lors du calcul de la distance.")

if segments:
    st.subheader("Résumé des segments")
    total_emissions = 0
    for seg in segments:
        st.markdown(f"**{seg['origine']} → {seg['destination']}** ({seg['mode']})")
        st.markdown(f"Distance : {seg['distance_km']:.1f} km | Poids : {seg['poids_kg']} kg | Émissions : {seg['emissions']:.2f} kg CO₂e")
        total_emissions += seg['emissions']

    st.success(f"**Émissions totales estimées : {total_emissions:.2f} kg CO₂e**")
