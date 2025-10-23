
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import great_circle

EMISSION_FACTORS = {
    "Aérien": 0.500,
    "Maritime": 0.015,
    "Routier": 0.100,
    "Ferroviaire": 0.030
}

geolocator = Nominatim(user_agent="co2_calculator")

def get_coordinates(city):
    location = geolocator.geocode(city)
    if location:
        return (location.latitude, location.longitude)
    else:
        return None

def calculate_emission(origin, destination, mode, weight):
    coords1 = get_coordinates(origin)
    coords2 = get_coordinates(destination)
    if coords1 and coords2:
        distance_km = great_circle(coords1, coords2).km
        emission = distance_km * weight * EMISSION_FACTORS[mode]
        return distance_km, emission
    else:
        return None, None

st.title("Calculateur d'empreinte carbone multimodal")

st.markdown("""
Ajoutez plusieurs segments de transport (origine, destination, mode, poids) pour calculer l'empreinte carbone totale.
""")

segments = []
num_legs = st.number_input("Nombre de segments de transport", min_value=1, max_value=10, value=1, step=1)

for i in range(num_legs):
    st.subheader(f"Segment {i+1}")
    origin = st.text_input(f"Origine du segment {i+1}", key=f"origin_{i}")
    destination = st.text_input(f"Destination du segment {i+1}", key=f"destination_{i}")
    mode = st.selectbox(f"Mode de transport du segment {i+1}", list(EMISSION_FACTORS.keys()), key=f"mode_{i}")
    weight = st.number_input(f"Poids (tonnes) du segment {i+1}", min_value=0.01, value=1.0, key=f"weight_{i}")
    segments.append((origin, destination, mode, weight))

if st.button("Calculer l'empreinte carbone totale"):
    total_emission = 0
    total_distance = 0
    for i, (origin, destination, mode, weight) in enumerate(segments):
        distance, emission = calculate_emission(origin, destination, mode, weight)
        if distance is not None:
            st.write(f"Segment {i+1} : {origin} → {destination} | {distance:.1f} km | {emission:.2f} kg CO₂e")
            total_emission += emission
            total_distance += distance
        else:
            st.error(f"Impossible de géocoder les villes pour le segment {i+1}.")

    st.success(f"Distance totale : {total_distance:.1f} km")
    st.success(f"Émissions totales estimées : {total_emission:.2f} kg CO₂e")
