
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import great_circle

# Liste de villes suggérées
suggested_cities = [
    "Paris", "Lyon", "Marseille", "Toulouse", "Nice",
    "Lille", "Nantes", "Strasbourg", "Bordeaux", "Montpellier",
    "Bruxelles", "Genève", "Londres", "Berlin", "Madrid",
    "Rome", "New York", "Tokyo", "Montréal", "Dakar"
]

st.title("Calculateur d'empreinte carbone logistique")
st.markdown("""
Saisissez ou sélectionnez les villes d'origine et de destination.
La distance sera calculée automatiquement à vol d'oiseau (great-circle).
""")

# Choix ou saisie libre pour origine
col1, col2 = st.columns(2)
with col1:
    origin_choice = st.selectbox("Ville d'origine (ou saisie libre ci-dessous)", [""] + suggested_cities)
    origin_custom = st.text_input("Ou entrez une ville d'origine personnalisée")
    origin = origin_custom if origin_custom else origin_choice

# Choix ou saisie libre pour destination
with col2:
    dest_choice = st.selectbox("Ville de destination (ou saisie libre ci-dessous)", [""] + suggested_cities)
    dest_custom = st.text_input("Ou entrez une ville de destination personnalisée")
    destination = dest_custom if dest_custom else dest_choice

# Mode de transport et poids
mode = st.selectbox("Mode de transport", ["Aérien", "Maritime", "Routier", "Ferroviaire"])
poids = st.number_input("Poids de la cargaison (en tonnes)", min_value=0.01, value=1.0)

# Facteurs d'émission (kg CO2e / tonne.km)
factors = {
    "Aérien": 0.5,
    "Maritime": 0.015,
    "Routier": 0.1,
    "Ferroviaire": 0.03
}

# Calcul
if st.button("Calculer l'empreinte carbone"):
    if origin and destination:
        geolocator = Nominatim(user_agent="co2_app")
        try:
            loc1 = geolocator.geocode(origin)
            loc2 = geolocator.geocode(destination)
            if loc1 and loc2:
                coord1 = (loc1.latitude, loc1.longitude)
                coord2 = (loc2.latitude, loc2.longitude)
                distance = great_circle(coord1, coord2).km
                emission = distance * poids * factors[mode]
                st.success(f"Distance estimée : {distance:.1f} km")
                st.success(f"Émissions estimées : {emission:.2f} kg CO₂e")
            else:
                st.error("Impossible de localiser une ou les deux villes.")
        except Exception as e:
            st.error(f"Erreur lors de la géolocalisation : {e}")
    else:
        st.warning("Veuillez saisir les deux villes.")
