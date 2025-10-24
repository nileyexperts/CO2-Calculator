# co2_calculator_app.py
# ------------------------------------------------------------
# Calculateur CO2 multimodal - NILEY EXPERTS
# - Géocodage OpenCage
# - Distance routière via OSRM + polyline sur la carte
# - Fond recentré, couleur d'origine + texte explicatif clair
# - Facteurs d'émission éditables, poids global ou par segment
# - Carte PyDeck (PathLayer pour routes OSRM, LineLayer en fallback)
# - Correctif: utilisation de st.rerun() (plus de st.experimental_rerun())
# - Nouveauté: reset_form() pour vider explicitement tous les champs
# ------------------------------------------------------------

import os
import time
import requests
import pandas as pd
import streamlit as st
import pydeck as pdk
from opencage.geocoder import OpenCageGeocode
from geopy.distance import great_circle

# =========================
# 🎯 Paramètres par défaut
# =========================
DEFAULT_EMISSION_FACTORS = {
    "🚛 Routier 🚛": 0.100,     # kg CO2e / t.km
    "✈️ Aérien ✈️": 0.500,
    "🚢 Maritime 🚢": 0.015,
    "🚂 Ferroviaire 🚂": 0.030,
}

BACKGROUND_URL = "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/background.png"
MAX_SEGMENTS = 10  # utilisé pour nettoyer toutes les clés potentielles

st.set_page_config(page_title="Calculateur CO₂ multimodal - NILEY EXPERTS",
                   page_icon="🌍", layout="centered")

# =========================
# 🎨 Styles (fond recentré, couleur d'origine)
# =========================
st.markdown(f"""
<style>
/* 🖼️ Image de fond : centrée, couleur d'origine (pas de voile) */
.stApp {{
  background-image: url("{BACKGROUND_URL}");
  background-size: 1200px auto;     /* largeur fixe confortable sur desktop */
  background-repeat: no-repeat;
  background-position: top center;   /* recentré */
  background-attachment: scroll;     /* moins envahissant au scroll */
}}

/* 🎯 Conteneur de contenu : fond blanc léger pour lisibilité */
.main .block-container {{
  background: rgba(255,255,255,0.95);
  border-radius: 10px;
  padding: 1.2rem 1.2rem 1.6rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  max-width: 960px;
}}

/* 🧩 Cartouches segments + contrôles (couleurs d'origine NILEY) */
.segment-box {{
  background-color: #DFEDF5;
  padding: 15px; border-radius: 10px; margin-bottom: 10px;
  border: 2px solid #BB9357;
}}
.stButton > button {{
  background-color: #BB9357; color: white; font-weight: bold;
  border-radius: 8px; padding: 10px 20px; border: none;
}}
.stTextInput > div > input, .stNumberInput > div > input {{
  background-color: #DFEDF5; border: 2px solid #BB9357; border-radius: 5px;
}}
.stSelectbox > div > div {{
  background-color: #DFEDF5 !important; border: 2px solid #BB9357; border-radius: 5px;
}}

/* 📱 Responsive : fond plus petit sur mobile */
@media (max-width: 768px) {{
  .stApp {{ background-size: 900px auto; }}
}}
</style>
""", unsafe_allow_html=True)

# =========================
# 🧰 Utilitaires
# =========================
def read_secret(key: str, default: str = "") -> str:
    if "secrets" in dir(st) and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

@st.cache_data(show_spinner=False, ttl=60*60)
def geocode_cached(query: str, limit: int = 5):
    if not query:
        return []
    try:
        time.sleep(0.1)  # évite de spammer en dev
        return geocoder.geocode(query, no_annotations=1, limit=limit) or []
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=24*60*60)
def coords_from_formatted(formatted: str):
    try:
        res = geocoder.geocode(formatted, no_annotations=1, limit=1)
        if res:
            g = res[0]["geometry"]
            return (g["lat"], g["lng"])  # (lat, lon)
    except Exception:
        pass
    return None

def compute_distance_km(coord1, coord2) -> float:
    return great_circle(coord1, coord2).km

def compute_emissions(distance_km: float, weight_tonnes: float, factor_kg_per_tkm: float) -> float:
    return distance_km * weight_tonnes * factor_kg_per_tkm

@st.cache_data(show_spinner=False, ttl=6*60*60)
def osrm_route(coord1, coord2, base_url: str, overview: str = "full"):
    """
    Distance routière (km) + géométrie (polyline GeoJSON) via OSRM /route/v1/driving.
    - coord1/coord2: (lat, lon)
    - base_url: ex. https://router.project-osrm.org
    Utilise: overview=full, geometries=geojson, alternatives=false, annotations=false
    """
    # OSRM attend lon,lat
    lon1, lat1 = coord1[1], coord1[0]
    lon2, lat2 = coord2[1], coord2[0]
    url = f"{base_url.rstrip('/')}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {
        "overview": overview,           # 'simplified' ou 'full'
        "alternatives": "false",
        "annotations": "false",
        "geometries": "geojson"
    }
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    routes = data.get("routes", [])
    if not routes:
        raise ValueError("Aucune route retournée par OSRM")
    route = routes[0]
    meters = float(route.get("distance", 0.0))
    distance_km = meters / 1000.0
    geom = route.get("geometry", {})  # GeoJSON LineString
    coords = geom.get("coordinates", [])  # [[lon, lat], ...]
    return {"distance_km": distance_km, "coords": coords}

def reset_form(max_segments: int = MAX_SEGMENTS):
    """
    Vide explicitement tous les champs et l'état :
    - supprime les clés de widgets (origin/dest input & select, mode, weight)
    - réinitialise la structure 'segments'
    - purge le cache de données
    - relance l'app
    """
    # Supprimer les clés de widgets potentielles
    widget_keys = []
    for i in range(max_segments):
        widget_keys.extend([
            f"origin_input_{i}", f"origin_select_{i}",
            f"dest_input_{i}",   f"dest_select_{i}",
            f"mode_{i}",         f"weight_{i}",
        ])
    for k in widget_keys:
        if k in st.session_state:
            del st.session_state[k]

    # Réinitialiser les données d'app
    for k in ["segments", "osrm_base_url", "weight_0"]:
        if k in st.session_state:
            del st.session_state[k]

    # Purger le cache (géocodage, routes, etc.)
    st.cache_data.clear()

    # Relancer l'app
    st.rerun()

# =========================
# 🔐 API OpenCage
# =========================
API_KEY = read_secret("OPENCAGE_KEY")
if not API_KEY:
    st.error("Clé API OpenCage absente. Ajoutez `OPENCAGE_KEY` à `st.secrets` ou à vos variables d’environnement.")
    st.stop()
geocoder = OpenCageGeocode(API_KEY)

# =========================
# 🏷️ En-tête & Texte explicatif (clair)
# =========================
st.markdown("""
<div style='background-color:#002E49;padding:20px;border-radius:10px'>
  <h1 style='color:white;text-align:center;margin:0'>
    Calculateur d'empreinte carbone multimodal - NILEY EXPERTS
  </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#f2f7fb; font-weight:500; margin-top:6px;">
  Ajoutez plusieurs segments (origine → destination), choisissez le mode et le poids.
  Le mode <strong>Routier</strong> utilise <strong>OSRM</strong> (distance réelle + tracé).
</div>
""", unsafe_allow_html=True)

# =========================
# 🔄 Reset (utilise reset_form)
# =========================
col_r, col_dummy = st.columns([1,4])
with col_r:
    if st.button("🔄 Réinitialiser le formulaire"):
        reset_form()

# =========================
# ⚙️ Paramètres
# =========================

# (Paramètres masqués) Facteurs par défaut et OSRM
factors = DEFAULT_EMISSION_FACTORS.copy()
unit = "kg"  # poids saisi en kg par défaut
osrm_default = st.session_state.get("osrm_base_url", "https://router.project-osrm.org")
osrm_base_url = osrm_default
st.session_state["osrm_base_url"] = osrm_base_url

