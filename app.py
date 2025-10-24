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
    "🚛 Routier 🚛": 0.100,
    "✈️ Aérien ✈️": 0.500,
    "🚢 Maritime 🚢": 0.015,
    "🚂 Ferroviaire 🚂": 0.030,
}
MAX_SEGMENTS = 10

st.set_page_config(page_title="Calculateur CO₂ multimodal - NILEY EXPERTS", page_icon="🌍", layout="centered")

# =========================
# 🧠 Utilitaires
# =========================
def read_secret(key: str, default: str = "") -> str:
    if "secrets" in dir(st) and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

@st.cache_data(ttl=3600)
def geocode_cached(query: str, limit: int = 5):
    if not query: return []
    try:
        time.sleep(0.1)
        return geocoder.geocode(query, no_annotations=1, limit=limit) or []
    except Exception:
        return []

@st.cache_data(ttl=86400)
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

@st.cache_data(ttl=21600)
def osrm_route(coord1, coord2, base_url: str, overview: str = "full"):
    lon1, lat1 = coord1[1], coord1[0]
    lon2, lat2 = coord2[1], coord2[0]
    url = f"{base_url.rstrip('/')}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": overview, "alternatives": "false", "annotations": "false", "geometries": "geojson"}
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
    for i in range(max_segments):
        for k in [f"origin_input_{i}", f"origin_select_{i}", f"dest_input_{i}", f"dest_select_{i}", f"mode_{i}", f"weight_{i}"]:
            st.session_state.pop(k, None)
    for k in ["segments", "osrm_base_url", "weight_0"]:
        st.session_state.pop(k, None)
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
        <img src='https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/NILEY-EXPERTS-logo-removebg-preview.png' stylee='margin: 0;'>Calculateur d'empreinte carbone multimodal - NILEY EXPERTS</h2>
    </div>
""", unsafe_allow_html=True)

st.markdown("Ajoutez plusieurs segments (origine → destination), choisissez le mode et le poids.", unsafe_allow_html=True)

# =========================
# 🔄 Reset + Mot de passe
# =========================
col_r, col_pwd = st.columns([1, 2])
with col_r:
    if st.button("🔄 Réinitialiser le formulaire"):
        reset_form()
with col_pwd:
    password = st.text_input("Mot de passe", type="password", placeholder="Entrez le mot de passe")

# =========================
# ⚙️ Paramètres
# =========================
with st.expander("⚙️ Paramètres, facteurs d'émission & OSRM"):
    default_mode = "Envoi unique (même poids sur tous les segments)"
    weight_mode = st.radio("Mode de gestion du poids :", [default_mode, "Poids par segment"], horizontal=False)

    factors = {}
    for mode_name, val in DEFAULT_EMISSION_FACTORS.items():
        factors[mode_name] = st.number_input(
            f"Facteur {mode_name} (kg CO₂e / tonne.km)",
            min_value=0.0, value=float(val), step=0.001, format="%.3f", key=f"factor_{mode_name}"
        )

    unit = st.radio("Unité de saisie du poids", ["kg", "tonnes"], index=0, horizontal=True)

    osrm_help = "**OSRM** – pour test : `https://router.project-osrm.org` (serveur démo, non garanti)."
    st.markdown(osrm_help)

    osrm_default = st.session_state.get("osrm_base_url", "https://router.project-osrm.org")
    osrm_base_url = st.text_input("Endpoint OSRM", value=osrm_default)
    st.session_state["osrm_base_url"] = osrm_base_url

# =========================
# 🧩 Segments
# =========================
def _default_segment(origin_raw="", origin_sel="", dest_raw="", dest_sel="", mode="🚛 Routier 🚛", weight=1000.0):
    return {"origin_raw": origin_raw, "origin_sel": origin_sel, "dest_raw": dest_raw, "dest_sel": dest_sel, "mode": mode, "weight": weight}

if "segments" not in st.session_state or not st.session_state.segments:
    st.session_state.segments = [_default_segment()]

segments_out = []
for i in range(len(st.session_state.segments)):
    st.subheader(f"Segment {i+1}")
    c1, c2 = st.columns(2)
    with c1:
        origin_raw = st.text_input(f"Origine {i+1}", value=st.session_state.segments[i]["origin_raw"], key=f"origin_input_{i}")
        origin_suggestions = geocode_cached(origin_raw, limit=5) if origin_raw else []
        origin_options = [r['formatted'] for r in origin_suggestions] if origin_suggestions else []
        origin_sel = st.selectbox("Suggestions origine", origin_options or ["—"], key=f"origin_select_{i}")
        if origin_sel == "—": origin_sel = ""
    with c2:
        dest_raw = st.text_input(f"Destination {i+1}", value=st.session_state.segments[i]["dest_raw"], key=f"dest_input_{i}")
        dest_suggestions = geocode_cached(dest_raw, limit=5) if dest_raw else []
        dest_options = [r['formatted'] for r in dest_suggestions] if dest_suggestions else []
        dest_sel = st.selectbox("Suggestions destination", dest_options or ["—"], key=f"dest_select_{i}")
        if dest_sel == "—": dest_sel = ""
    mode = st.selectbox(f"Mode {i+1}", list(DEFAULT_EMISSION_FACTORS.keys()), key=f"mode_{i}")
    weight_val = st.number_input(f"Poids {i+1}", min_value=0.001, value=float(st.session_state.segments[i]["weight"]), step=100.0, key=f"weight_{i}")

    st.session_state.segments[i] = {"origin_raw": origin_raw, "origin_sel": origin_sel, "dest_raw": dest_raw, "dest_sel": dest_sel, "mode": mode, "weight": weight_val}
    segments_out.append({"origin": origin_sel or origin_raw, "destination": dest_sel or dest_raw, "mode": mode, "weight": weight_val})

# =========================
# ✅ Calcul + Carte
# =========================
if st.button("Calculer l'empreinte carbone totale"):
    rows = []
    total_emissions = 0.0
    total_distance = 0.0
    points = []
    route_paths = []
    for idx, seg in enumerate(segments_out, start=1):
        if not seg["origin"] or not seg["destination"]:
            st.warning(f"Segment {idx} incomplet.")
            continue
        coord1 = coords_from_formatted(seg["origin"])
        coord2 = coords_from_formatted(seg["destination"])
        if not coord1 or not coord2:
            st.error(f"Segment {idx} : lieu introuvable.")
            continue
        route_coords = None
        if seg["mode"].startswith("🚛"):
            try:
                r = osrm_route(coord1, coord2, st.session_state.get("osrm_base_url", "https://router.project-osrm.org"))
                distance_km = r["distance_km"]
                route_coords = r["coords"]
            except Exception:
                distance_km = compute_distance_km(coord1, coord2)
        else:
            distance_km = compute_distance_km(coord1, coord2)
        weight_tonnes = seg["weight"] / 1000.0
        factor = DEFAULT_EMISSION_FACTORS.get(seg["mode"], 0.0)
        emissions = compute_emissions(distance_km, weight_tonnes, factor)
        total_distance += distance_km
        total_emissions += emissions
        rows.append({"Segment": idx, "Origine": seg["origin"], "Destination": seg["destination"], "Mode": seg["mode"], "Distance (km)": round(distance_km, 1), "Poids (kg)": seg["weight"], "Émissions (kg CO₂e)": round(emissions, 2)})

        # Points pour la carte
        points.append({"position": [coord1[1], coord1[0]], "name": f"Origine {idx}"})
        points.append({"position": [coord2[1], coord2[0]], "name": f"Destination {idx}"})
        if route_coords:
            route_paths.append({"path": route_coords, "name": f"Segment {idx}"})

    if rows:
        st.success(f"Distance totale : {total_distance:.1f} km • Émissions totales : {total_emissions:.2f} kg CO₂e")
        df = pd.DataFrame(rows)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger CSV", data=csv, file_name="resultats_co2.csv", mime="text/csv")

        # Carte PyDeck
        layers = []
        if route_paths:
            layers.append(pdk.Layer("PathLayer", data=route_paths, get_path="path", get_color=[187, 147, 87, 220], width_scale=1, width_min_pixels=4))
        if points:
            layers.append(pdk.Layer("ScatterplotLayer", data=points, get_position="position", get_fill_color=[0, 122, 255, 220], get_radius=20000))
        view_state = pdk.ViewState(latitude=48.8566, longitude=2.3522, zoom=3)
        st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=view_state, layers=layers))
``
