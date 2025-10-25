# -*- coding: utf-8 -*-
# Calculateur CO2 multimodal - NILEY EXPERTS
# Version : fond WEB (#DFEDF5) + logo fixe + login avec bouton + Natural Earth (Cartopy)
# + ratio carte PDF conservé + ROUTAGE MARITIME SIMPLIFIÉ (Distance Tools, clé codée en dur)

import os
import time
import math
import unicodedata
import requests
import pandas as pd
import streamlit as st
import pydeck as pdk
from opencage.geocoder import OpenCageGeocode
from geopy.distance import great_circle
from io import BytesIO
from datetime import datetime
# PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage

from PIL import Image as PILImage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import tempfile
import numpy as np

# -- Cache Cartopy (Natural Earth)
os.environ.setdefault("CARTOPY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "cartopy_cache"))

# =========================
# Paramètres & Config page
# =========================
st.set_page_config(
    page_title="Calculateur CO2 multimodal - NILEY EXPERTS",
    page_icon="🌍",
    layout="centered"
)

# Ajout du fond de page couleur #DFEDF5
st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: #DFEDF5 !important;
        }
        [data-testid="stSidebar"] {
            background: #DFEDF5 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

DEFAULT_EMISSION_FACTORS = {
    "Routier": 0.100,
    "Aerien": 0.500,
    "Maritime": 0.015,
    "Ferroviaire": 0.030,
}
MAX_SEGMENTS = 10
LOGO_URL = "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/NILEY-EXPERTS-logo-removebg-preview.png"

# =========================
# Utilitaires communs
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
    params = {"overview": overview, "alternatives": "false", "annotations": "false", "geometries": "geojson"}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    routes = data.get("routes", [])
    if not routes:
        raise ValueError("Aucune route retournee par OSRM")
    route = routes[0]
    meters = float(route.get("distance", 0.0))
    distance_km = meters / 1000.0
    geom = route.get("geometry", {})
    coords = geom.get("coordinates", [])
    return {"distance_km": distance_km, "coords": coords}

def _normalize_no_diacritics(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()

def mode_to_category(mode_str: str) -> str:
    s = _normalize_no_diacritics(mode_str)
    if "routier" in s or "road" in s or "truck" in s:
        return "routier"
    if "aerien" in s or "air" in s or "plane" in s:
        return "aerien"
    if "maritime" in s or "mer" in s or "bateau" in s or "sea" in s or "ship" in s:
        return "maritime"
    if "ferroviaire" in s or "rail" in s or "train" in s:
        return "ferroviaire"
    return "routier"

ICON_URLS = {
    "routier": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/truck.png",
    "aerien": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/plane.png",
    "maritime": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/ship.png",
    "ferroviaire": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/train.png",
}

# =========================
# Routage maritime simplifié (Distance Tools)
# =========================
API_DISTANCE_TOOLS_KEY = "DEMO_KEY"  # Remplacez par votre clé réelle
DISTANCE_TOOLS_BASE = "https://api.distance.tools"

def sea_route_maritime(coord1, coord2):
    """Retourne la distance maritime en km via Distance Tools, fallback sur great-circle."""
    try:
        lat1, lon1 = coord1[0], coord1[1]
        lat2, lon2 = coord2[0], coord2[1]
        payload = {
            "route": [
                {"name": f"{lat1},{lon1}"},
                {"name": f"{lat2},{lon2}"}
            ]
        }
        url = f"{DISTANCE_TOOLS_BASE}/api/v2/distance/route/maritime"
        headers = {
            "X-Billing-Token": API_DISTANCE_TOOLS_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        r = requests.post(url, json=payload, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        distance_km = None
        route_obj = data.get("route", {})
        if isinstance(route_obj, dict):
            maritime = route_obj.get("maritime")
            if isinstance(maritime, dict) and "distance" in maritime:
                distance_km = float(maritime.get("distance"))
            elif "distance" in route_obj:
                distance_km = float(route_obj.get("distance"))
        if distance_km is None:
            steps = data.get("steps") or []
            if steps and isinstance(steps, list):
                distance_km = sum(float(s.get("distance", 0.0)) for s in steps)
        if distance_km is None:
            raise ValueError("Distance non trouvée")
        return distance_km
    except Exception:
        return great_circle(coord1, coord2).km

# =========================
# Vérification du mot de passe
# =========================
PASSWORD_KEY = "APP_PASSWORD"
if PASSWORD_KEY not in st.secrets:
    st.error("Mot de passe non configuré. Ajoutez APP_PASSWORD dans .streamlit/secrets.toml.")
    st.stop()

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

st.markdown("## Accès sécurisé")
with st.form("login_form", clear_on_submit=False):
    password_input = st.text_input("Entrez le mot de passe :", type="password", placeholder="Votre mot de passe")
    submitted = st.form_submit_button("Valider")

if not st.session_state.auth_ok:
    if submitted:
        if password_input == st.secrets[PASSWORD_KEY]:
            st.session_state.auth_ok = True
            st.success("Accès autorisé. Bienvenue !")
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")
    else:
        st.info("Veuillez saisir le mot de passe puis cliquer sur Valider.")
    st.stop()
else:
    st.success("Accès autorisé. Bienvenue !")

# =========================
# API OpenCage
# =========================
API_KEY = read_secret("OPENCAGE_KEY")
if not API_KEY:
    st.error("Clé API OpenCage absente.")
    st.stop()
geocoder = OpenCageGeocode(API_KEY)

# =========================
# Interface principale
# =========================
st.markdown("## Calculateur d'empreinte carbone multimodal - NILEY EXPERTS")
st.markdown("Ajoutez plusieurs segments (origine -> destination), choisissez le mode et le poids.")

dossier_transport = st.text_input("N° dossier Transport (obligatoire) *", value=st.session_state.get("dossier_transport", ""), placeholder="ex : TR-2025-001")
st.session_state["dossier_transport"] = (dossier_transport or "").strip()

default_mode_label = "Envoi unique (même poids sur tous les segments)"
weight_mode = st.radio("Mode de gestion du poids :", [default_mode_label, "Poids par segment"], horizontal=False)

factors = {}
for mode_name, val in DEFAULT_EMISSION_FACTORS.items():
    factors[mode_name] = st.number_input(f"Facteur {mode_name} (kg CO2e / tonne.km)", min_value=0.0, value=float(val), step=0.001, format="%.3f", key=f"factor_{mode_name}")

unit = st.radio("Unité de saisie du poids", ["kg", "tonnes"], index=0, horizontal=True)

osrm_base_url = st.text_input("Endpoint OSRM", value=st.session_state.get("osrm_base_url", "https://router.project-osrm.org"))
st.session_state["osrm_base_url"] = osrm_base_url

# Segments
def _default_segment(origin_raw="", origin_sel="", dest_raw="", dest_sel="", mode=None, weight=1000.0):
    if mode is None:
        mode = list(DEFAULT_EMISSION_FACTORS.keys())[0]
    return {"origin_raw": origin_raw, "origin_sel": origin_sel, "dest_raw": dest_raw, "dest_sel": dest_sel, "mode": mode, "weight": weight}

if "segments" not in st.session_state or not st.session_state.segments:
    st.session_state.segments = [_default_segment()]

for i in range(1, len(st.session_state.segments)):
    prev = st.session_state.segments[i - 1]
    cur = st.session_state.segments[i]
    if prev.get("dest_sel") and not cur.get("origin_raw") and not cur.get("origin_sel"):
        cur["origin_raw"] = prev["dest_sel"]
        cur["origin_sel"] = prev["dest_sel"]

segments_out = []
for i in range(len(st.session_state.segments)):
    st.markdown(f"##### Segment {i+1}")
    c1, c2 = st.columns(2)
    with c1:
        origin_raw = st.text_input(f"Origine du segment {i+1}", value=st.session_state.segments[i]["origin_raw"], key=f"origin_input_{i}")
        origin_suggestions = geocode_cached(origin_raw, limit=5) if origin_raw else []
        origin_options = [r['formatted'] for r in origin_suggestions] if origin_suggestions else []
        origin_sel = st.selectbox("Suggestions pour l'origine", origin_options or ["-"], index=0, key=f"origin_select_{i}")
        if origin_sel == "-":
            origin_sel = ""
    with c2:
        dest_raw = st.text_input(f"Destination du segment {i+1}", value=st.session_state.segments[i]["dest_raw"], key=f"dest_input_{i}")
        dest_suggestions = geocode_cached(dest_raw, limit=5) if dest_raw else []
        dest_options = [r['formatted'] for r in dest_suggestions] if dest_suggestions else []
        dest_sel = st.selectbox("Suggestions pour la destination", dest_options or ["-"], index=0, key=f"dest_select_{i}")
        if dest_sel == "-":
            dest_sel = ""

    mode = st.selectbox(f"Mode de transport du segment {i+1}", list(DEFAULT_EMISSION_FACTORS.keys()), index=list(DEFAULT_EMISSION_FACTORS.keys()).index(st.session_state.segments[i]["mode"]) if st.session_state.segments[i]["mode"] in DEFAULT_EMISSION_FACTORS else 0, key=f"mode_{i}")

    if weight_mode == "Poids par segment":
        default_weight = st.session_state.segments[i]["weight"]
        weight_val = st.number_input(f"Poids transporté pour le segment {i+1}", min_value=0.001, value=float(default_weight), step=100.0 if unit == "kg" else 0.1, key=f"weight_{i}")
    else:
        default_weight = st.session_state.segments[0]["weight"]
        if i == 0:
            weight_val = st.number_input("Poids transporté (appliqué à tous les segments)", min_value=0.001, value=float(default_weight), step=100.0 if unit == "kg" else 0.1, key="weight_0")
        else:
            weight_val = st.session_state.get("weight_0", default_weight)

    st.session_state.segments[i] = {"origin_raw": origin_raw, "origin_sel": origin_sel, "dest_raw": dest_raw, "dest_sel": dest_sel, "mode": mode, "weight": weight_val}
    segments_out.append({"origin": origin_sel or origin_raw or "", "destination": dest_sel or dest_raw or "", "mode": mode, "weight": weight_val})

bc1, bc2, _ = st.columns([2, 2, 6])
with bc1:
    can_add = len(st.session_state.segments) < MAX_SEGMENTS
    if st.button("Ajouter un segment après le dernier", disabled=not can_add):
        last = st.session_state.segments[-1]
        new_seg = _default_segment(origin_raw=last.get("dest_sel") or last.get("dest_raw") or "", origin_sel=last.get("dest_sel") or "", mode=last.get("mode", list(DEFAULT_EMISSION_FACTORS.keys())[0]), weight=last.get("weight", 1000.0))
        st.session_state.segments.append(new_seg)
        st.rerun()
with bc2:
    if st.button("Supprimer le dernier segment", disabled=len(st.session_state.segments) <= 1):
        st.session_state.segments.pop()
        st.rerun()

# =========================
# Calcul + Carte
# =========================
can_calculate = bool(st.session_state.get("dossier_transport"))
if not can_calculate:
    st.warning("Veuillez renseigner le N° dossier Transport avant de lancer le calcul.")

if st.button("Calculer l'empreinte carbone totale", disabled=not can_calculate):
    rows = []
    total_emissions = 0.0
    total_distance = 0.0

    with st.spinner("Calcul en cours..."):
        for idx, seg in enumerate(segments_out, start=1):
            if not seg["origin"] or not seg["destination"]:
                st.warning(f"Segment {idx} : origine/destination manquante(s).")
                continue

            coord1 = coords_from_formatted(seg["origin"])
            coord2 = coords_from_formatted(seg["destination"])
            if not coord1 or not coord2:
                st.error(f"Segment {idx} : lieu introuvable ou ambigu.")
                continue

            route_coords = None
            mode_norm = _normalize_no_diacritics(seg["mode"])

            if "routier" in mode_norm:
                try:
                    r = osrm_route(coord1, coord2, st.session_state["osrm_base_url"], overview="full")
                    distance_km = r["distance_km"]
                    route_coords = r["coords"]
                except Exception:
                    distance_km = compute_distance_km(coord1, coord2)

            elif "maritime" in mode_norm:
                distance_km = sea_route_maritime(coord1, coord2)

            else:
                distance_km = compute_distance_km(coord1, coord2)

            weight_tonnes = seg["weight"] if unit == "tonnes" else seg["weight"]/1000.0
            factor = float(factors.get(seg["mode"], DEFAULT_EMISSION_FACTORS.get(seg["mode"], 0.0)))
            emissions = compute_emissions(distance_km, weight_tonnes, factor)

            total_distance += distance_km
            total_emissions += emissions

            rows.append({"Segment": idx, "Origine": seg["origin"], "Destination": seg["destination"], "Mode": seg["mode"], "Distance (km)": round(distance_km, 1), f"Poids ({unit})": round(seg["weight"], 3 if unit=="tonnes" else 1), "Facteur (kg CO2e/t.km)": factor, "Emissions (kg CO2e)": round(emissions, 2), "lat_o": coord1[0], "lon_o": coord1[1], "lat_d": coord2[0], "lon_d": coord2[1], "route_coords": route_coords})

    if rows:
        df = pd.DataFrame(rows)
        st.success(f"{len(rows)} segment(s) calculé(s) • Distance totale : {total_distance:.1f} km • Emissions totales
