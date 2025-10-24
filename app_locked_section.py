import os
import time
import math
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
    "🚛 Routier 🚛": 0.100,  # kg CO2e / t.km
    "✈️ Aérien ✈️": 0.500,
    "🚢 Maritime 🚢": 0.015,
    "🚂 Ferroviaire 🚂": 0.030,
}
BACKGROUND_URL = "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/background.png"
MAX_SEGMENTS = 10

st.set_page_config(
    page_title="Calculateur CO₂ multimodal - NILEY EXPERTS",
    page_icon="🌍",
    layout="centered"
)

# =========================
# 🎨 Styles (placeholder)
# =========================
st.markdown(""" """, unsafe_allow_html=True)

# =========================
# 🧠 Utilitaires
# =========================
def read_secret(key: str, default: str = "") -> str:
    if "secrets" in dir(st) and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

@st.cache_data(show_spinner=False, ttl=60*60)
def geocode_cached(query: str, limit: int = 5):
    if not query: return []
    try:
        time.sleep(0.1)
        return geocoder.geocode(query, no_annotations=1, limit=limit) or []
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=24*60*60)
def coords_from_formatted(formatted: str):
    try:
        res = geocoder.geocode(formatted, no_annotations=1, limit=1)
        if res:
            g = res[0]["geometry"]
            return (g["lat"], g["lng"])
    except Exception:
        pass
    return None

def compute_distance_km(coord1, coord2) -> float:
    return great_circle(coord1, coord2).km

def compute_emissions(distance_km: float, weight_tonnes: float, factor_kg_per_tkm: float) -> float:
    return distance_km * weight_tonnes * factor_kg_per_tkm

@st.cache_data(show_spinner=False, ttl=6*60*60)
def osrm_route(coord1, coord2, base_url: str, overview: str = "full"):
    lon1, lat1 = coord1[1], coord1[0]
    lon2, lat2 = coord2[1], coord2[0]
    url = f"{base_url.rstrip('/')}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {
        "overview": overview,
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
    geom = route.get("geometry", {})
    coords = geom.get("coordinates", [])
    return {"distance_km": distance_km, "coords": coords}

def reset_form(max_segments: int = MAX_SEGMENTS):
    widget_keys = []
    for i in range(max_segments):
        widget_keys.extend([
            f"origin_input_{i}", f"origin_select_{i}",
            f"dest_input_{i}", f"dest_select_{i}",
            f"mode_{i}", f"weight_{i}",
        ])
    for k in widget_keys:
        if k in st.session_state:
            del st.session_state[k]
    for k in ["segments", "osrm_base_url", "weight_0"]:
        if k in st.session_state:
            del st.session_state[k]
    st.cache_data.clear()
    st.rerun()

# =========================
# 🔐 API OpenCage
# =========================
API_KEY = read_secret("OPENCAGE_KEY")
if not API_KEY:
    st.error("Clé API OpenCage absente.")
    st.stop()
geocoder = OpenCageGeocode(API_KEY)

# =========================
# 🏷️ En-tête
# =========================
st.markdown("""
    <div style='display: flex; align-items: center;'>
        <img src='https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/NILEY-EXPERTS-logo-removebg-preview.png' style='height: 60px; marginteur d'empreinte carbone multimodal - NILEY EXPERTS</h2>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
Ajoutez plusieurs segments (origine → destination), choisissez le mode et le poids.
Le mode Routier utilise OSRM (distance réelle + tracé).
""", unsafe_allow_html=True)

# =========================
# 🔄 Reset
# =========================
col_r, col_dummy = st.columns([1, 4])
with col_r:
    if st.button("🔄 Réinitialiser le formulaire"):
        reset_form()

# =========================
# 🔐 Section protégée
# =========================
with st.expander("⚙️ Paramètres, facteurs d'émission & OSRM"):
    password = st.text_input("Mot de passe requis pour accéder aux paramètres", type="password")
    if password == "Niley2019!":
        default_mode = "Envoi unique (même poids sur tous les segments)"
        weight_mode = st.radio("Mode de gestion du poids :", [default_mode, "Poids par segment"], horizontal=False)

        factors = {}
        for mode_name, val in DEFAULT_EMISSION_FACTORS.items():
            factors[mode_name] = st.number_input(
                f"Facteur {mode_name} (kg CO₂e / tonne.km)",
                min_value=0.0, value=float(val), step=0.001, format="%.3f", key=f"factor_{mode_name}"
            )

        unit = st.radio("Unité de saisie du poids", ["kg", "tonnes"], index=0, horizontal=True)

        osrm_help = (
            "**OSRM** – pour test : `https://router.project-osrm.org` (serveur démo, non garanti). "
            "En production, utilisez un serveur auto‑hébergé ou un provider."
        )
        st.markdown(osrm_help)

        osrm_default = st.session_state.get("osrm_base_url", "https://router.project-osrm.org")
        osrm_base_url = st.text_input("Endpoint OSRM", value=osrm_default, help="Ex: https://router.project-osrm.org ou votre propre serveur OSRM")
        st.session_state["osrm_base_url"] = osrm_base_url
    else:
        st.warning("Mot de passe incorrect ou non saisi.")

# ✅ Le reste du code (segments, calcul, carte, export CSV) reste inchangé
