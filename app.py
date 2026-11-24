
# -*- coding: utf-8 -*-
# Calculateur CO2 multimodal - NILEY EXPERTS
# Application Streamlit avec export PDF mono-page.

import os
import io
import re
import time
import math
import unicodedata
import tempfile
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from PIL import Image as PILImage

import streamlit as st
import pydeck as pdk
from geopy.distance import great_circle
from opencage.geocoder import OpenCageGeocode

# PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.utils import ImageReader

# Correctif : échappement XML pour Paragraph (ReportLab)
from xml.sax.saxutils import escape as xml_escape

# Matplotlib pour le rendu de la carte du PDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Cache Cartopy
os.environ.setdefault("CARTOPY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "cartopy_cache"))

# --------------------------
# Configuration page
# --------------------------
st.set_page_config(page_title="Calculateur GES multimodal - NILEY EXPERTS", page_icon="🌍", layout="centered")

# --------------------------
# Style global : fond + conteneurs arrondis + saisies mises en évidence
# --------------------------
st.markdown(
    """
    <style>
    /* ---------- Fond de l'app ---------- */
    [data-testid="stAppViewContainer"] {
        background-color: #DFEDF5 !important;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }
    .stApp { /* fallback */
        background-color: #DFEDF5 !important;
    }
    .block-container { /* garder le fond global visible sous les blocs */
        background: transparent !important;
    }

    /* ---------- Cartes/sections : bordures arrondies ---------- */
    [data-testid="stAppViewContainer"] .block-container > div {
        border-radius: 12px;
        background: #FFFFFF;
        border: 1px solid #CFD8E3;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.06);
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }
    [data-testid="stVerticalBlock"] > div[style*="border"] {
        border-radius: 12px !important;
        border: 1px solid #CFD8E3 !important;
        background: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.06) !important;
        padding: 0.75rem 1rem !important;
    }

    /* Tableaux, onglets, et composants courants */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #CFD8E3;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.06);
        background: #FFFFFF;
    }
    [data-testid="stTabs"] {
        border-radius: 12px;
        border: 1px solid #CFD8E3;
        background: #FFFFFF;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.06);
        padding: 0.25rem 0.25rem 0.75rem 0.25rem;
    }

    /* Pydeck et Matplotlib : cadre doux autour des canvases */
    .stDeckGlJson {
        border-radius: 12px;
        border: 1px solid #CFD8E3;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.06);
        background: #FFFFFF;
        padding: 0.25rem;
    }
    [data-testid="stImage"] canvas,
    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    /* Boutons : cohérence visuelle */
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid #CFD8E3 !important;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06) !important;
    }

    /* ==============================
       Champs de saisie Origine/Destination
       ============================== */
    /* Le label texte au-dessus du champ */
    label:has(+ div [data-testid="stTextInput"] input[aria-label*="Origine — Adresse"]),
    label:has(+ div [data-testid="stTextInput"] input[aria-label*="Destination — Adresse"]) {
        color: #0F2A3A;
        font-weight: 600;
    }

    /* Boîte wrapper des champs Origine */
    [data-testid="stTextInput"]:has(input[aria-label*="Origine — Adresse"]) {
        border: 1px solid #8CB3CC;
        border-radius: 10px;
        background: #CFE3EE;  /* plus foncé que #DFEDF5 */
        padding: 0.5rem 0.65rem;
        box-shadow: 0 1px 4px rgba(16,24,40,0.06);
    }

    /* Boîte wrapper des champs Destination */
    [data-testid="stTextInput"]:has(input[aria-label*="Destination — Adresse"]) {
        border: 1px solid #8CB3CC;
        border-radius: 10px;
        background: #CFE3EE;
        padding: 0.5rem 0.65rem;
        box-shadow: 0 1px 4px rgba(16,24,40,0.06);
    }

    /* L'INPUT interne : transparent pour laisser voir le fond du wrapper */
    [data-testid="stTextInput"] input[aria-label*="Origine — Adresse"],
    [data-testid="stTextInput"] input[aria-label*="Destination — Adresse"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #0F2A3A;
    }

    /* Placeholder lisible */
    [data-testid="stTextInput"] input[aria-label*="Origine — Adresse"]::placeholder,
    [data-testid="stTextInput"] input[aria-label*="Destination — Adresse"]::placeholder {
        color: #3E6074;
        opacity: 0.9;
    }

    /* Focus (clavier/souris) : cadre + halo d’accessibilité */
    [data-testid="stTextInput"]:has(input[aria-label*="Origine — Adresse"]:focus-within),
    [data-testid="stTextInput"]:has(input[aria-label*="Destination — Adresse"]:focus-within) {
        border-color: #3B82B3 !important;
        box-shadow: 0 0 0 3px rgba(59,130,179,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True
)

LOGO_URL = "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/NILEY-EXPERTS-logo-removebg-preview.png"

# --------------------------
# Constantes
# --------------------------
PDF_THEME_DEFAULT = "terrain"
NE_SCALE_DEFAULT = "50m"

PDF_BASEMAP_LABELS = [
    "Identique a la carte Web (Carto)",
    "Auto (Stamen -> OSM -> NaturalEarth)",
    "Stamen Terrain (internet)",
    "OSM (internet)",
    "Natural Earth (vectoriel, offline)"
]
PDF_BASEMAP_MODES = {
    "Identique a la carte Web (Carto)": "carto_web",
    "Auto (Stamen -> OSM -> NaturalEarth)": "auto",
    "Stamen Terrain (internet)": "stamen",
    "OSM (internet)": "osm",
    "Natural Earth (vectoriel, offline)": "naturalearth",
}

MODE_LABELS = {
    "Routier": "Routier",
    "Maritime": "Maritime",
    "Ferroviaire": "Rail (ferroviaire)",
    "Aerien": "Aérien",
}

GES_GUIDE_URL = "https://www.ecologie.gouv.fr/politiques-publiques/information-ges-prestations-transport"

# Guide ministériel « Information GES des prestations de transport ».
# Tableau 10 : g CO2e/t.km, par densité croissante.
RAIL_FACTORS = {
    "Mixte France métropolitaine (90 % électrique / 10 % gazole)": (5.36, 4.21, 3.64),
    "Électrique – France métropolitaine": (1.99, 1.55, 1.33),
    "Électrique – Europe hors France": (17.4, 13.5, 11.7),
    "Diesel – gazole non routier": (36.2, 28.1, 24.2),
}
RAIL_DENSITY_LABELS = ("< 250", "250 à < 400", "≥ 400")

# Tableau 16 : g CO2e/t.km.
SEA_FACTORS = {
    "Porte-conteneurs de 1 900 à 3 849 EVP": 20.4,
    "Porte-conteneurs de moins de 1 200 EVP": 33.1,
    "Porte-conteneurs de 1 200 à 1 899 EVP": 21.9,
    "Porte-conteneurs de 3 850 à 7 499 EVP": 13.7,
    "Porte-conteneurs de 7 500 EVP et plus": 10.2,
    "Ro-Ro": 103.0,
}
# Tableaux 17 et 18 : litres de gazole par km et charge moyenne en tonnes.
ROAD_PROFILES = {
    "Articulé 40 t – longue distance": (0.342, 12.50),
    "Articulé 40 t – régional": (0.338, 12.50),
    "Articulé 40 t – porte-conteneur": (0.373, 12.50),
    "Articulé 40 t – messagerie": (0.342, 6.00),
    "Porteur 19 t – messagerie": (0.270, 2.50),
    "Utilitaire 3,5 t – express colis": (0.160, 0.46),
}
ROAD_DIESEL_KG_CO2E_PER_L = 3.16
# Tableau 7 : repères historiques DGAC (2016) pour avions tout-cargo.
AIR_CARGO_PROFILES = (
    (1000, 1.508, "100–250 t MMD, ≤ 1 000 km"),
    (4000, 1.065, "100–250 t MMD, 1 000–4 000 km"),
    (7000, 0.562, "> 250 t MMD, 4 000–7 000 km"),
    (math.inf, 0.537, "> 250 t MMD, > 7 000 km"),
)

MAX_SEGMENTS = 50

ICON_URLS = {
    "routier": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/truck.png",
    "aerien":  "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/plane.png",
    "maritime":"https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/ship.png",
    "ferroviaire":"https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/train.png",
}

# --------------------------
# Helpers segments
# --------------------------
def _default_segment(mode=None, weight=1000.0):
    if mode is None:
        mode = next(iter(MODE_LABELS))
    return {
        "origin": {"query":"", "display":"", "iata":"", "coord":None},
        "dest":   {"query":"", "display":"", "iata":"", "coord":None},
        "mode": mode,
        "weight": weight,
    }

def add_segment_end():
    if "segments" not in st.session_state or not st.session_state.segments:
        st.session_state.segments = [_default_segment()]
        return
    if len(st.session_state.segments) >= MAX_SEGMENTS:
        st.warning("Nombre maximum de segments atteint.")
        return

    prev = st.session_state.segments[-1]
    new_seg = _default_segment()

    prev_dest = prev.get("dest", {})
    # Forcer le chaînage même si l'utilisateur a modifié l'origine
    if prev_dest.get("display"):
        new_seg["origin"].update({
            "display": prev_dest.get("display", ""),
            "coord": prev_dest.get("coord"),
            "iata": prev_dest.get("iata", ""),
            "query": prev_dest.get("display", "")
        })
        # Mise à jour explicite de l'état de session pour refléter le chaînage
        j = len(st.session_state.segments)
        st.session_state[f"origin_query_{j}"] = prev_dest.get("display", "")
        st.session_state[f"origin_display_{j}"] = prev_dest.get("display", "")
        st.session_state[f"origin_coord_{j}"] = prev_dest.get("coord")
        st.session_state[f"origin_iata_{j}"] = prev_dest.get("iata", "")
        st.session_state[f"origin_autofill_{j}"] = True
        st.session_state[f"chain_src_signature_{j}"] = _normalize_signature(prev_dest.get("display"), prev_dest.get("coord"))
        st.session_state[f"origin_user_edited_{j}"] = False

    if "weight_0" in st.session_state:
        try:
            new_seg["weight"] = float(st.session_state["weight_0"])
        except Exception:
            pass

    st.session_state.segments.append(new_seg)

def clear_segment_inputs(index):
    for prefix in ("origin", "dest"):
        for suffix in ("query", "choice", "coord", "display", "iata", "unlo", "autofill", "user_edited"):
            st.session_state.pop(f"{prefix}_{suffix}_{index}", None)
    for prefix in (
        "mode_select", "chain_src_signature", "distance_km", "emission_method",
        "carrier_total", "emission_source", "specific_factor", "road_profile",
        "sea_profile", "rail_traction", "rail_volume", "additional_emissions",
    ):
        st.session_state.pop(f"{prefix}_{index}", None)


def remove_last_segment():
    segments = st.session_state.get("segments", [])
    last_index = max(0, len(segments) - 1)
    if last_index == 0:
        st.session_state.segments = [_default_segment()]
    else:
        segments.pop()
    clear_segment_inputs(last_index)


def reset_segments():
    try:
        st.session_state.segments = [_default_segment()]
        for i in range(MAX_SEGMENTS):
            clear_segment_inputs(i)
        st.session_state.pop("calculation_result", None)
        st.session_state.pop("weight_0", None)
        st.session_state.pop("dossier_transport", None)
    finally:
        st.rerun()

# --------------------------
# Utilitaires & APIs
# --------------------------
def read_secret(key: str, default: str = "") -> str:
    if "secrets" in dir(st) and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

@st.cache_data(show_spinner=False, ttl=60*60)
def geocode_cached(query: str, limit: int = 5):
    if not query or geocoder is None:
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
def osrm_route(coord1, coord2, base_url: str = "https://router.project-osrm.org", overview: str = "full"):
    lon1, lat1 = coord1[1], coord1[0]
    lon2, lat2 = coord2[1], coord2[0]
    url = f"{base_url.rstrip('/')}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "full", "alternatives": "false", "annotations": "false", "geometries": "geojson"}
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
    normalized = _normalize_no_diacritics(mode_str)
    if any(word in normalized for word in ("ferroviaire", "rail", "train")):
        return "ferroviaire"
    if any(word in normalized for word in ("aerien", "air", "plane")):
        return "aerien"
    if any(word in normalized for word in ("maritime", "mer", "bateau", "sea", "ship")):
        return "maritime"
    return "routier"


def segment_distance(segment):
    coord_o, coord_d = segment["coord_o"], segment["coord_d"]
    if segment["distance_km"] is not None:
        return segment["distance_km"], "Distance renseignée", None
    if segment["carrier_total"] is not None:
        return None, "Distance non renseignée", None
    mode = segment["mode"]
    if mode in ("Ferroviaire", "Maritime"):
        network = "ferroviaire" if mode == "Ferroviaire" else "maritime"
        raise ValueError(f"renseignez le kilométrage de l’itinéraire {network}.")
    if not coord_o or not coord_d:
        raise ValueError("sélectionnez les deux lieux ou renseignez la distance.")
    if mode == "Routier":
        try:
            route = osrm_route(coord_o, coord_d)
            if route["distance_km"] > 0:
                return route["distance_km"], "Distance routière estimée (OSRM)", route["coords"]
        except (requests.RequestException, ValueError, TypeError, KeyError):
            pass
        return compute_distance_km(coord_o, coord_d), "Vol d’oiseau (itinéraire routier indisponible)", None
    return compute_distance_km(coord_o, coord_d), "Distance orthodromique", None


def emission_factor(segment, distance_km):
    mode = segment["mode"]
    if segment["method"] == "Facteur spécifique":
        if segment["specific_factor"] is None:
            raise ValueError("renseignez le facteur d’émission spécifique.")
        return segment["specific_factor"], "Méthode spécifique", segment["source"]
    if mode == "Ferroviaire":
        if segment["profile"] is None or segment["volume"] is None:
            raise ValueError("sélectionnez la traction et renseignez le volume du fret en m³.")
        density = segment["weight"] / segment["volume"]
        density_index = 0 if density < 250 else 1 if density < 400 else 2
        factor = RAIL_FACTORS[segment["profile"]][density_index] / 1000
        profile = f"{segment['profile']}, {RAIL_DENSITY_LABELS[density_index]} kg/m³"
        return factor, "Niveau 1", f"Guide GES, tab. 10 – {profile}"
    if mode == "Routier":
        consumption, average_load = ROAD_PROFILES[segment["profile"]]
        factor = consumption * ROAD_DIESEL_KG_CO2E_PER_L / average_load
        return factor, "Niveau 1", f"Guide GES, tab. 17–18 – {segment['profile']}"
    if mode == "Maritime":
        factor = SEA_FACTORS[segment["profile"]] / 1000
        return factor, "Niveau 1", f"Guide GES, tab. 16 – {segment['profile']}"
    for maximum_km, factor, profile in AIR_CARGO_PROFILES:
        if distance_km <= maximum_km:
            return factor, "Estimation tout-cargo", f"Guide GES, tab. 7 (DGAC 2016) – {profile}"


def calculate_segment(segment, index):
    if not segment["origin_display"] or not segment["destination_display"]:
        raise ValueError("renseignez l’origine et la destination.")
    if segment["method"] != "Estimation par profil" and not segment["source"]:
        raise ValueError("renseignez la source ou la référence du document.")
    if segment["method"] == "Total du transporteur" and segment["carrier_total"] is None:
        raise ValueError("renseignez le total en kg CO₂e du transporteur.")

    distance, distance_source, route_coords = segment_distance(segment)
    tonnes_km = (segment["weight"] / 1000) * distance if distance is not None else None
    factor = None
    if segment["carrier_total"] is not None:
        emissions = segment["carrier_total"]
        method, source = "Total du transporteur", segment["source"]
        additional = 0.0
    else:
        if distance <= 0:
            raise ValueError("renseignez une distance supérieure à zéro.")
        factor, method, source = emission_factor(segment, distance)
        additional = segment["additional_emissions"]
        emissions = compute_emissions(distance, segment["weight"] / 1000, factor) + additional

    coord_o = segment["coord_o"] or (None, None)
    coord_d = segment["coord_d"] or (None, None)
    return {
        "Segment": index,
        "Origine": segment["origin_display"],
        "Destination": segment["destination_display"],
        "Mode": MODE_LABELS[segment["mode"]],
        "Distance (km)": distance,
        "Source distance": distance_source,
        "Poids (kg)": segment["weight"],
        "Activité (t.km)": tonnes_km,
        "Facteur (kg CO2e/t.km)": factor,
        "Fuites frigorigènes (kg CO2e)": additional if factor is not None else None,
        "Emissions (kg CO2e)": emissions,
        "Méthode": method,
        "Source / hypothèse": source,
        "lat_o": coord_o[0], "lon_o": coord_o[1],
        "lat_d": coord_d[0], "lon_d": coord_d[1],
        "route_coords": route_coords,
    }


def distance_summary(total_distance, rows):
    missing = sum(row["Distance (km)"] is None for row in rows)
    if missing == len(rows):
        return "Non renseignée"
    suffix = f" (partielle, {missing} sans distance)" if missing else ""
    return f"{total_distance:.1f} km{suffix}"


# --------------------------
# PDF helpers
# --------------------------
def _compute_extent_from_coords(all_lats, all_lons, margin_ratio=0.12, min_span_deg=1e-3):
    if not all_lats or not all_lons:
        return (-10, 30, 30, 60)
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    span_lat = max(max_lat - min_lat, min_span_deg)
    span_lon = max(max_lon - min_lon, min_span_deg)
    min_lat -= span_lat * margin_ratio; max_lat += span_lat * margin_ratio
    min_lon -= span_lon * margin_ratio; max_lon += span_lon * margin_ratio
    return (min_lon, max_lon, min_lat, max_lat)

def _fit_extent_to_aspect(min_lon, max_lon, min_lat, max_lat, target_aspect_w_over_h):
    span_lon = max(1e-6, max_lon - min_lon)
    span_lat = max(1e-6, max_lat - min_lat)
    mid_lat = (min_lat + max_lat) / 2.0
    cos_mid = max(0.05, math.cos(math.radians(mid_lat)))
    aspect_geo = (span_lon * cos_mid) / span_lat
    aspect_target = max(1e-6, float(target_aspect_w_over_h))
    if aspect_geo < aspect_target:
        needed_lon = (aspect_target * span_lat) / cos_mid
        extra = (needed_lon - span_lon) / 2.0
        min_lon -= extra; max_lon += extra
    else:
        needed_lat = (span_lon * cos_mid) / aspect_target
        extra = (needed_lat - span_lat) / 2.0
        min_lat -= extra; max_lat += extra
    min_lon = max(-180.0, min_lon); max_lon = min(180.0, max_lon)
    min_lat = max(-90.0, min_lat);  max_lat = min(90.0,  max_lat)
    return (min_lon, max_lon, min_lat, max_lat)

def _pdf_add_mode_icon(ax, lon, lat, cat_key, size_px, transform=None):
    try:
        url = ICON_URLS.get(cat_key)
        if not url:
            return
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return
        pil = PILImage.open(io.BytesIO(resp.content)).convert('RGBA')
        w = max(1, pil.width)
        zoom = max(0.1, float(size_px) / float(w))
        imgbox = OffsetImage(pil, zoom=zoom)
        ab = AnnotationBbox(imgbox, (lon, lat), frameon=False)
        if transform is not None:
            ab.set_transform(transform)
        ab.set_zorder(5)
        ax.add_artist(ab)
    except Exception:
        pass

# --------------------------
# Tuiles Carto XYZ (même style que la carte web)
# --------------------------
def _carto_tiler_from_web_style(web_style_label: str):
    try:
        import cartopy.io.img_tiles as cimgt
    except Exception:
        return None

    url_by_label = {
        "Carto Positron (clair)": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "Carto Dark Matter (sombre)": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "Carto Voyager": "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
    }
    subdomains = ["a", "b", "c", "d"]

    template = url_by_label.get(web_style_label)
    if not template:
        template = url_by_label["Carto Positron (clair)"]

    class CartoTiles(cimgt.GoogleTiles):
        def _image_url(self, tile):
            z, x, y = tile
            s = subdomains[(x + y) % len(subdomains)]
            return template.format(s=s, z=z, x=x, y=y)

    return CartoTiles()

# --------------------------
# Chaînage robuste: helpers
# --------------------------
def _normalize_signature(display: str, coord) -> tuple:
    d = (display or "").strip()
    if coord and isinstance(coord, (list, tuple)) and len(coord) == 2:
        return (d, round(float(coord[0]), 6), round(float(coord[1]), 6))
    return (d, None, None)

def _is_location_empty(loc: dict) -> bool:
    if not loc:
        return True
    return not (loc.get("display") or loc.get("query") or "").strip()

# --------------------------
# Rapport PDF mono-page
# --------------------------
def generate_pdf_report(
    dossier_val, total_distance, total_emissions, unit, rows,
    pdf_basemap_choice_label, ne_scale='50m', pdf_theme='terrain',
    pdf_icon_size_px=24, web_map_style_label=None, detail_params=None
):
    from reportlab.pdfgen import canvas as pdfcanvas

    if detail_params is None:
        detail_params = {"dpi": 220, "max_zoom": 9}

    PAGE_W, PAGE_H = landscape(A4)
    M = 1.0 * cm
    AVAIL_W = PAGE_W - 2*M
    AVAIL_H = PAGE_H - 2*M

    buffer = BytesIO()
    c = pdfcanvas.Canvas(buffer, pagesize=landscape(A4))

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=14, textColor=colors.HexColor('#1f4788'), spaceAfter=0, alignment=1)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=10.5, textColor=colors.HexColor('#2c5aa0'), spaceAfter=0, spaceBefore=0, alignment=0)
    normal_style = styles['Normal']; normal_style.fontSize = 8

    y = PAGE_H - M

    # Logo + titre (anti-chevauchement)
    logo_h = 1.5 * cm
    logo_w = 3.0 * cm
    logo_drawn = False
    try:
        resp = requests.get(LOGO_URL, timeout=10)
        if resp.ok:
            img = ImageReader(io.BytesIO(resp.content))
            c.drawImage(img, M, y - logo_h, width=logo_w, height=logo_h, preserveAspectRatio=True, mask='auto')
            logo_drawn = True
    except Exception:
        pass

    title_para = Paragraph("RAPPORT D’ÉMISSIONS DE GES DU TRANSPORT", title_style)
    title_box_w = AVAIL_W - (logo_w + 0.5*cm if logo_drawn else 0)
    title_w, title_h = title_para.wrap(title_box_w, AVAIL_H)
    if logo_drawn:
        title_x = M + logo_w + 0.5*cm
        title_y = y - (logo_h/2.0) - (title_h/2.0)
    else:
        title_x = M
        title_y = y - title_h
    title_para.drawOn(c, title_x, title_y)

    header_block_h = max(logo_h if logo_drawn else 0, title_h)
    y = y - header_block_h - 0.35*cm

    # Résumé
    info_summary_data = [
        ["N° dossier Transport", dossier_val, "Distance totale", distance_summary(total_distance, rows)],
        ["Date du rapport:", datetime.now().strftime("%d/%m/%Y %H:%M"), "Emissions totales:", f"{total_emissions:.2f} kg CO2e"],
        ["Nombre de segments", str(len(rows)), "Périmètre GES", "Amont de l’énergie et fonctionnement"],
    ]
    sum_col_w = [4.5*cm, 5.5*cm, 4.5*cm, 5.5*cm]
    info_tbl = Table(info_summary_data, colWidths=sum_col_w)
    info_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f7')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#fff4e6')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    iw, ih = info_tbl.wrap(AVAIL_W, AVAIL_H)
    info_tbl.drawOn(c, M, y - ih)
    y = y - ih - 0.25*cm
    method_note = Paragraph(
        f"Somme des segments. Pour les estimations, masse brute (t) × distance (km) × facteur, "
        f"avec fuites frigorigènes saisies le cas échéant. "
        f'<link href="{GES_GUIDE_URL}" color="#1f4788">Guide ministériel Information GES</link>.',
        normal_style,
    )
    _, note_height = method_note.wrap(AVAIL_W, AVAIL_H)
    method_note.drawOn(c, M, y - note_height)
    y -= note_height + 0.2*cm

    # Carte (mono-page)
    footer_h = 0.6*cm
    min_table_h = 5.5*cm
    max_map_h = 7.5*cm
    remaining_h = (y - M) - footer_h

    map_h = max(4.0*cm, min(max_map_h, remaining_h * 0.42))
    table_h_avail = remaining_h - map_h - 0.25*cm
    if table_h_avail < min_table_h:
        delta = (min_table_h - table_h_avail)
        map_h = max(4.0*cm, map_h - delta)
        table_h_avail = (y - M) - footer_h - map_h - 0.25*cm

    dpi = int(detail_params.get("dpi", 220))

    def _choose_zoom(min_lon, max_lon, min_lat, max_lat):
        span_lon = max_lon - min_lon
        span_lat = max_lat - min_lat
        span = max(span_lon, span_lat)
        if span <= 0.5:   z = 9
        elif span <= 1.0: z = 8
        elif span <= 2.0: z = 7
        elif span <= 5.0: z = 6
        elif span <= 12.0:z = 5
        elif span <= 24.0:z = 4
        else:             z = 3
        return min(z, int(detail_params.get("max_zoom", 9)))

    map_buffer = None
    mapped_rows = [row for row in rows if row["lat_o"] is not None and row["lat_d"] is not None]
    try:
        all_lats = [r["lat_o"] for r in mapped_rows] + [r["lat_d"] for r in mapped_rows]
        all_lons = [r["lon_o"] for r in mapped_rows] + [r["lon_d"] for r in mapped_rows]
        min_lon, max_lon, min_lat, max_lat = _compute_extent_from_coords(all_lats, all_lons)

        fig_w_in = AVAIL_W / 72.0
        fig_h_in = map_h / 72.0

        min_lon, max_lon, min_lat, max_lat = _fit_extent_to_aspect(
            min_lon, max_lon, min_lat, max_lat, target_aspect_w_over_h=(AVAIL_W / max(1, map_h))
        )

        use_cartopy = True
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.io.img_tiles import Stamen, OSM
        except Exception:
            use_cartopy = False

        mode_label = pdf_basemap_choice_label
        mode = PDF_BASEMAP_MODES.get(mode_label, "auto")

        if use_cartopy:
            fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
            ax = None
            raster_ok = False
            try:
                if mode == "carto_web" and web_map_style_label:
                    tiler = _carto_tiler_from_web_style(web_map_style_label)
                    if tiler is not None:
                        ax = plt.axes(projection=tiler.crs)
                        ax.set_extent((min_lon, max_lon, min_lat, max_lat), crs=ccrs.PlateCarree())
                        zoom = _choose_zoom(min_lon, max_lon, min_lat, max_lat)
                        ax.add_image(tiler, zoom)
                        raster_ok = True

                if not raster_ok and mode in ("auto","stamen"):
                    tiler = Stamen('terrain-background')
                    ax = plt.axes(projection=tiler.crs)
                    ax.set_extent((min_lon, max_lon, min_lat, max_lat), crs=ccrs.PlateCarree())
                    zoom = _choose_zoom(min_lon, max_lon, min_lat, max_lat)
                    ax.add_image(tiler, zoom)
                    raster_ok = True

                if not raster_ok and mode in ("auto","osm"):
                    tiler = OSM()
                    ax = plt.axes(projection=tiler.crs)
                    ax.set_extent((min_lon, max_lon, min_lat, max_lat), crs=ccrs.PlateCarree())
                    zoom = _choose_zoom(min_lon, max_lon, min_lat, max_lat)
                    ax.add_image(tiler, zoom)
                    raster_ok = True
            except Exception:
                raster_ok = False

            if not raster_ok:
                ax = plt.axes(projection=ccrs.PlateCarree())
                colors_cfg = {'ocean':'#EAF4FF','land':'#F7F5F2','lakes_fc':'#EAF4FF','lakes_ec':'#B3D4F5',
                              'coast':'#818892','borders0':'#8F98A3'}
                ax.add_feature(cfeature.OCEAN.with_scale(ne_scale), facecolor=colors_cfg['ocean'], edgecolor='none', zorder=0)
                ax.add_feature(cfeature.LAND.with_scale(ne_scale),  facecolor=colors_cfg['land'],  edgecolor='none', zorder=0)
                ax.add_feature(cfeature.LAKES.with_scale(ne_scale), facecolor=colors_cfg['lakes_fc'], edgecolor=colors_cfg['lakes_ec'], linewidth=0.3, zorder=1)
                ax.add_feature(cfeature.COASTLINE.with_scale(ne_scale), edgecolor=colors_cfg['coast'], linewidth=0.4, zorder=2)
                ax.add_feature(cfeature.BORDERS.with_scale(ne_scale),   edgecolor=colors_cfg['borders0'], linewidth=0.5, zorder=2)
                try:
                    admin1 = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor='#777777', facecolor='none')
                    ax.add_feature(admin1, linewidth=0.4, zorder=2)
                    roads = cfeature.NaturalEarthFeature('cultural', 'roads', '10m', edgecolor='#B07020', facecolor='none')
                    ax.add_feature(roads, linewidth=0.35, zorder=3)
                    urban = cfeature.NaturalEarthFeature('cultural', 'urban_areas', '10m', edgecolor='none', facecolor='#E0D5CC')
                    ax.add_feature(urban, zorder=1)
                except Exception:
                    pass
                try:
                    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.25, color='#DADDE2', alpha=0.7, linestyle='--')
                except Exception:
                    pass
                ax.set_extent((min_lon, max_lon, min_lat, max_lat), crs=ccrs.PlateCarree())
            else:
                try:
                    ax.add_feature(cfeature.COASTLINE.with_scale(ne_scale), edgecolor='#555', linewidth=0.4, zorder=2)
                except Exception:
                    pass
                try:
                    ax.add_feature(cfeature.BORDERS.with_scale(ne_scale), edgecolor='#666', linewidth=0.4, zorder=2)
                except Exception:
                    pass
                try:
                    ax.add_feature(cfeature.LAKES.with_scale(ne_scale), facecolor='#87B9FF', edgecolor='#6FA8FF', linewidth=0.2, zorder=1)
                except Exception:
                    pass
                try:
                    ax.add_feature(cfeature.RIVERS.with_scale(ne_scale), edgecolor='#6FA8FF', linewidth=0.25, zorder=1)
                except Exception:
                    pass

            mode_colors = {"routier":"#0066CC","aerien":"#CC0000","maritime":"#009900","ferroviaire":"#9900CC"}
            for r in mapped_rows:
                cat = mode_to_category(r["Mode"]); color = mode_colors.get(cat, "#666666")
                ax.plot([r["lon_o"], r["lon_d"]], [r["lat_o"], r["lat_d"]],
                        color=color, linewidth=2.0, alpha=0.9, transform=ccrs.PlateCarree(), zorder=3)
                ax.scatter([r["lon_o"]], [r["lat_o"]], s=22, c="#0A84FF", edgecolors='white', linewidths=0.8,
                           transform=ccrs.PlateCarree(), zorder=4)
                ax.scatter([r["lon_d"]], [r["lat_d"]], s=22, c="#FF3B30", edgecolors='white', linewidths=0.8,
                           transform=ccrs.PlateCarree(), zorder=4)
                mid_lon = (r["lon_o"] + r["lon_d"]) / 2.0
                mid_lat = (r["lat_o"] + r["lat_d"]) / 2.0
                _pdf_add_mode_icon(ax, mid_lon, mid_lat, cat, pdf_icon_size_px, transform=ccrs.PlateCarree())

            map_buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(map_buffer, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            map_buffer.seek(0)

        else:
            # Fallback sans cartopy
            fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
            ax.set_facecolor('#F7F8FA')
            ax.set_xlim(min_lon, max_lon); ax.set_ylim(min_lat, max_lat)
            lat_step = (max_lat-min_lat)/6.0 if (max_lat-min_lat)>0 else 1
            lon_step = (max_lon-min_lon)/8.0 if (max_lon-min_lon)>0 else 1
            for yy in np.arange(min_lat, max_lat+1e-9, lat_step):
                ax.plot([min_lon, max_lon],[yy,yy], color='#E6E9EF', lw=0.6)
            for xx in np.arange(min_lon, max_lon+1e-9, lon_step):
                ax.plot([xx,xx],[min_lat,max_lat], color='#E6E9EF', lw=0.6)
            mode_colors = {"routier":"#0066CC","aerien":"#CC0000","maritime":"#009900","ferroviaire":"#9900CC"}
            for r in mapped_rows:
                cat = mode_to_category(r["Mode"]); color = mode_colors.get(cat, "#666666")
                ax.plot([r["lon_o"], r["lon_d"]], [r["lat_o"], r["lat_d"]], color=color, lw=2.0, alpha=0.9)
                ax.scatter([r["lon_o"]],[r["lat_o"]], s=22, c="#0A84FF", edgecolor='white', lw=0.8)
                ax.scatter([r["lon_d"]],[r["lat_d"]], s=22, c="#FF3B30", edgecolor='white', lw=0.8)
            map_buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(map_buffer, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            map_buffer.seek(0)

    except Exception:
        map_buffer = None

    if map_buffer:
        img = ImageReader(map_buffer)
        c.drawImage(img, M, y - map_h, width=AVAIL_W, height=map_h, preserveAspectRatio=True, mask='auto')
        y = y - map_h - 0.25*cm

    else:
        table_h_avail = (y - M) - footer_h - 0.25*cm

    heading_para = Paragraph("Détail des segments", heading_style)
    hw, hh = heading_para.wrap(AVAIL_W, AVAIL_H)
    heading_para.drawOn(c, M, y - hh)
    y = y - hh - 0.10*cm

    headers = [
        "Seg.", "Origine", "Destination", "Mode", "Dist.\n(km)", f"Poids\n({unit})",
        "Facteur\n(kg CO2e/t.km)", "Émissions\n(kg CO2e)", "Méthode et sources",
    ]
    col_widths = [width * cm for width in (1.0, 3.7, 3.7, 2.35, 1.45, 1.65, 2.35, 2.35, 9.1)]

    def cell(text):
        return Paragraph(xml_escape(str(text)), normal_style)

    data_rows = []
    for row in rows:
        distance = row["Distance (km)"]
        factor = row["Facteur (kg CO2e/t.km)"]
        additional = row["Fuites frigorigènes (kg CO2e)"]
        method_details = f"{row['Méthode']} · {row['Source / hypothèse']} · {row['Source distance']}"
        if additional:
            method_details += f" · Fuites frigorigènes : {additional:g} kg CO2e"
        data_rows.append([
            str(row["Segment"]),
            cell(row["Origine"]),
            cell(row["Destination"]),
            cell(row["Mode"]),
            "—" if distance is None else f"{distance:.3f}".rstrip("0").rstrip("."),
            f"{row[f'Poids ({unit})']:g}",
            "—" if factor is None else f"{factor:.8f}".rstrip("0").rstrip("."),
            f"{row['Emissions (kg CO2e)']:.6f}".rstrip("0").rstrip("."),
            cell(method_details),
        ])

    total_row = [
        "TOTAL", "", "", "", f"{total_distance:.1f}" if total_distance else "—",
        "", "", f"{total_emissions:.2f}", cell(distance_summary(total_distance, rows)),
    ]

    def build_table(font_size, rows_limit=None, show_notice=False, hidden_count=0):
        body = [headers]
        body += (data_rows if rows_limit is None else data_rows[:rows_limit])
        body.append(total_row)
        if show_notice and hidden_count > 0:
            notice = Paragraph(
                f"... {hidden_count} segment(s) supplémentaire(s) dans le détail CSV ...",
                ParagraphStyle('Notice', parent=styles['Normal'], fontSize=max(7, font_size-1), textColor=colors.grey, alignment=1)
            )
            body.append([notice] + [""]*(len(headers)-1))
        tbl = Table(body, colWidths=col_widths, repeatRows=1)
        total_row_offset = 1 if (show_notice and hidden_count>0) else 0
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), font_size),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, -1 - total_row_offset), (-1, -1 - total_row_offset), colors.HexColor('#fff4e6')),
        ]
        if show_notice and hidden_count>0:
            style.append(('SPAN', (0, -1), (-1, -1)))
        tbl.setStyle(TableStyle(style))
        return tbl

    avail_w, avail_h = AVAIL_W, table_h_avail
    final_tbl = None

    for fs in (8, 7, 6):
        test_tbl = build_table(fs)
        tw, th = test_tbl.wrap(avail_w, avail_h)
        if th <= avail_h:
            final_tbl = test_tbl
            break

    if final_tbl is None:
        max_data = len(data_rows)
        low, high = 0, max_data
        best_tbl = None; best = 0
        while low <= high:
            mid = (low + high)//2
            test_tbl = build_table(6, rows_limit=mid, show_notice=True, hidden_count=(max_data - mid))
            tw, th = test_tbl.wrap(avail_w, avail_h)
            if th <= avail_h:
                best = mid; best_tbl = test_tbl; low = mid + 1
            else:
                high = mid - 1
        final_tbl = best_tbl

    if final_tbl is not None:
        tw, th = final_tbl.wrap(avail_w, avail_h)
        final_tbl.drawOn(c, M, y - th)
        y = y - th - 0.10*cm

    footer_para = Paragraph(
        f"Document genere le {datetime.now().strftime('%d/%m/%Y %H:%M')} — Calculateur CO2 multimodal - NILEY EXPERTS",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=7, textColor=colors.grey, alignment=1)
    )
    fw, fh = footer_para.wrap(AVAIL_W, 0.8*cm)
    footer_para.drawOn(c, M + (AVAIL_W - fw)/2.0, M - 0.2*cm)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# --------------------------
# Auth
# --------------------------
PASSWORD_KEY = "APP_PASSWORD"
if PASSWORD_KEY not in st.secrets:
    st.error("Mot de passe non configure. Ajoutez APP_PASSWORD dans .streamlit/secrets.toml.")
    st.stop()

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.markdown("## Acces securise")
    with st.form("login_form", clear_on_submit=True):
        password_input = st.text_input("Entrez le mot de passe pour acceder a l'application :", type="password", placeholder="Votre mot de passe", key="__pwd__")
        submitted = st.form_submit_button("Valider")
        if submitted:
            if password_input == st.secrets[PASSWORD_KEY]:
                st.session_state.auth_ok = True
                try:
                    del st.session_state["__pwd__"]
                except KeyError:
                    pass
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
        else:
            st.info("Veuillez saisir le mot de passe puis cliquer sur Valider.")
    st.stop()
else:
    st.success("Acces autorise.")

# --------------------------
# API OpenCage
# --------------------------
API_KEY = read_secret("OPENCAGE_KEY")
geocoder = OpenCageGeocode(API_KEY) if API_KEY else None
if geocoder is None:
    st.caption("Géocodage OpenCage indisponible. Vous pouvez saisir les lieux et les distances manuellement.")

# --------------------------
# Donnees Aeroports / IATA
# --------------------------
@st.cache_data(show_spinner=False, ttl=7*24*60*60)
def load_airports_iata(path: str = "airport-codes.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.warning(f"Impossible de charger '{path}': {e}")
        return pd.DataFrame(columns=["iata_code","name","municipality","iso_country","lat","lon","label","type"])

    required = {"iata_code","name","coordinates"}
    missing = required - set(df.columns)
    if missing:
        st.error("Colonnes manquantes dans airport-codes.csv")
        return pd.DataFrame(columns=["iata_code","name","municipality","iso_country","lat","lon","label","type"])

    df = df[df["iata_code"].astype(str).str.len()==3].copy()
    df["iata_code"] = df["iata_code"].astype(str).str.upper()

    if "type" in df.columns:
        df = df[df["type"].isin(["large_airport","medium_airport"])].copy()

    coord_series = df["coordinates"].astype(str).str.replace('"','').str.strip()
    parts = coord_series.str.split(",", n=1, expand=True)
    if parts.shape[1] < 2:
        parts = pd.DataFrame({0:coord_series, 1:None})
    df["lat"] = pd.to_numeric(parts[0].astype(str).str.strip(), errors="coerce")
    df["lon"] = pd.to_numeric(parts[1].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["lat","lon"]).copy()

    for col in ["municipality","iso_country","name"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["name","municipality","iso_country"]:
        df[col] = df[col].astype(str).replace({"nan":""}).fillna("").str.strip()

    def _label(r):
        base = f"{(r['iata_code'] or '').strip()} - {(r['name'] or 'Sans nom').strip()}"
        extra = " · ".join([p for p in [r['municipality'], r['iso_country']] if p])
        return f"{base} {extra}" if extra else base

    df["label"] = df.apply(_label, axis=1)
    cols = ["iata_code","name","municipality","iso_country","lat","lon","label"]
    if "type" in df.columns:
        cols.append("type")
    return df[cols]

@st.cache_data(show_spinner=False, ttl=24*60*60)
def search_airports(query: str, limit: int = 20) -> pd.DataFrame:
    df = load_airports_iata()
    q = (query or "").strip()
    if df.empty:
        return df
    if not q:
        return df.head(limit)
    if len(q) <= 3:
        res = df[df["iata_code"].str.startswith(q.upper())]
        if res.empty:
            res = df[
                df["name"].str.lower().str.contains(q.lower())
                | df["municipality"].astype(str).str.lower().str.contains(q.lower())
            ]
    else:
        res = df[
            df["name"].str.lower().str.contains(q.lower())
            | df["municipality"].astype(str).str.lower().str.contains(q.lower())
        ]
    return res.head(limit)

# --------------------------
# Ports
# --------------------------
@st.cache_data(show_spinner=False, ttl=7*24*60*60)
def load_ports_csv(path: str = "ports.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.warning(f"Impossible de charger '{path}': {e}")
        return pd.DataFrame(columns=["unlocode", "name", "country", "lat", "lon", "label"])

    cols = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    col_unlo = pick("unlocode", "locode", "unlocode ")
    col_name = pick("name", "port_name")
    col_ctry = pick("country", "iso_country", "country_code")
    col_lat  = pick("lat", "latitude")
    col_lon  = pick("lon", "lng", "long", "longitude")

    if col_lat is None or col_lon is None:
        st.error("Colonnes lat/lon manquantes dans ports.csv")
        return pd.DataFrame(columns=["unlocode", "name", "country", "lat", "lon", "label"])

    out = pd.DataFrame()
    out["unlocode"] = df[col_unlo] if col_unlo else ""
    out["name"]     = df[col_name] if col_name else ""
    out["country"]  = df[col_ctry] if col_ctry else ""
    out["lat"] = pd.to_numeric(df[col_lat], errors="coerce")
    out["lon"] = pd.to_numeric(df[col_lon], errors="coerce")
    out = out.dropna(subset=["lat","lon"]).copy()

    out["unlocode"] = out["unlocode"].astype(str).str.upper().str.strip()
    out["name"]     = out["name"].astype(str).replace({"nan":""}).fillna("").str.strip()
    out["country"]  = out["country"].astype(str).replace({"nan":""}).fillna("").str.strip()

    def _label_port(r):
        base = (r["name"] or "Port sans nom").strip()
        extras = " · ".join([p for p in [r["unlocode"], r["country"]] if p])
        return f"{base} {extras}" if extras else base

    out["label"] = out.apply(_label_port, axis=1)
    return out[["unlocode","name","country","lat","lon","label"]]

@st.cache_data(show_spinner=False, ttl=24*60*60)
def search_ports(query: str, limit: int = 12) -> pd.DataFrame:
    df = load_ports_csv()
    q = (query or "").strip()
    if df.empty:
        return df
    if not q:
        return df.head(limit)

    if len(q) in (5,6):
        res = df[
            df["unlocode"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
            .str.startswith(re.sub(r"[^A-Za-z0-9]", "", q.upper()))
        ]
        if not res.empty:
            return res.head(limit)

    ql = q.lower()
    res = df[
        df["name"].str.lower().str.contains(ql, na=False)
        | df["country"].str.lower().str.contains(ql, na=False)
        | df["unlocode"].str.lower().str.contains(ql, na=False)
    ]
    return res.head(limit)

# --------------------------
# Champ unifié Adresse/Ville/Pays ou IATA
# --------------------------
def unified_location_input(side_key: str, seg_index: int, label_prefix: str, show_airports: bool = True, show_ports: bool = False):
    """
    Saisie unifiée. Mode IATA prioritaire: si la saisie est exactement 3 lettres, on force la recherche aéroports.
    Retourne dict: coord, display, iata, unlocode, query, choice
    """
    q_key   = f"{side_key}_query_{seg_index}"
    c_key   = f"{side_key}_choice_{seg_index}"
    crd_key = f"{side_key}_coord_{seg_index}"
    disp_key= f"{side_key}_display_{seg_index}"
    iata_key= f"{side_key}_iata_{seg_index}"
    unlo_key= f"{side_key}_unlo_{seg_index}"

    label = f"{label_prefix} — Adresse / Ville / Pays"
    query_val = st.text_input(label, value=st.session_state.get(q_key, ""), key=q_key)
    q_raw = (query_val or "").strip()
    is_iata_mode = show_airports and bool(re.fullmatch(r"[A-Za-z]{3}", q_raw))
    q_iata = q_raw.upper() if is_iata_mode else None

    airports = pd.DataFrame()
    ports = pd.DataFrame()
    oc_opts = []

    if q_raw:
        if show_airports or is_iata_mode:
            airports = search_airports(q_raw, limit=20)
            if is_iata_mode and not airports.empty:
                exact  = airports[airports["iata_code"].str.upper() == q_iata]
                others = airports[airports["iata_code"].str.upper() != q_iata]
                airports = pd.concat([exact, others], ignore_index=True)

        if show_ports and not is_iata_mode:
            ports = search_ports(q_raw, limit=12)

        if not is_iata_mode:
            oc = geocode_cached(q_raw, limit=5)
            oc_opts = [r['formatted'] for r in oc] if oc else []

    options = []
    airport_rows = []
    port_rows = []

    if (show_airports or is_iata_mode) and not airports.empty:
        for _, r in airports.iterrows():
            label_opt = f"✈️ {r['label']} (IATA {r['iata_code']})"
            options.append(label_opt); airport_rows.append(r)

    if show_ports and not is_iata_mode and not ports.empty:
        for _, r in ports.iterrows():
            suffix = f" (UN/LOCODE {r['unlocode']})" if r.get("unlocode") else ""
            label_opt = f"⚓ {r['label']}{suffix}"
            options.append(label_opt); port_rows.append(r)

    if not is_iata_mode and oc_opts:
        options += [f"📍 {o}" for o in oc_opts]

    if not options:
        options = ["— Aucun resultat —"]

    default_index = 0
    if is_iata_mode and airport_rows:
        default_index = 0

    sel = st.selectbox("Resultats", options, index=default_index, key=c_key)

    coord = None; display = ""; sel_iata = ""; sel_unlo = ""
    if sel != "— Aucun resultat —":
        if sel.startswith("✈️"):
            idx_in_all = options.index(sel)
            r = airport_rows[idx_in_all] if 0 <= idx_in_all < len(airport_rows) else airports.iloc[0]
            coord = (float(r["lat"]), float(r["lon"]))
            display = r["label"]
            sel_iata = r["iata_code"]
            sel_unlo = ""
        elif sel.startswith("⚓"):
            idx_global = options.index(sel)
            idx = idx_global - (len(airport_rows))
            r = port_rows[idx] if 0 <= idx < len(port_rows) else ports.iloc[0]
            coord = (float(r["lat"]), float(r["lon"]))
            display = r["label"]
            sel_unlo = str(r.get("unlocode") or "")
            sel_iata = ""
        else:
            formatted = sel[2:].strip() if sel.startswith("📍") else sel
            coord = coords_from_formatted(formatted)
            display = formatted
            sel_iata = ""; sel_unlo = ""

    st.session_state[crd_key]  = coord
    st.session_state[disp_key] = display
    st.session_state[iata_key] = sel_iata
    st.session_state[unlo_key] = sel_unlo

    return {"coord": coord, "display": display or q_raw, "iata": sel_iata, "unlocode": sel_unlo, "query": query_val, "choice": sel}

# --------------------------
# UI
# --------------------------
st.image(LOGO_URL, width=620)
st.markdown("### Calculateur d’émissions de GES du fret multimodal")

if "segments" not in st.session_state or not st.session_state.segments:
    st.session_state.segments = [_default_segment()]

# Auto-chainage au chargement si origine vide
for i in range(1, len(st.session_state.segments)):
    prev, cur = st.session_state.segments[i-1], st.session_state.segments[i]
    if prev.get("dest",{}).get("display") and _is_location_empty(cur.get("origin", {})):
        cur["origin"]["display"] = prev["dest"]["display"]
        cur["origin"]["coord"]   = prev["dest"]["coord"]
        cur["origin"]["iata"]    = prev["dest"]["iata"]
        cur["origin"]["query"]   = prev["dest"]["display"]
        st.session_state[f"origin_autofill_{i}"] = True
        st.session_state[f"chain_src_signature_{i}"] = _normalize_signature(prev["dest"].get("display"), prev["dest"].get("coord"))
        st.session_state[f"origin_user_edited_{i}"] = False

col_title, col_reset = st.columns([8, 2])
with col_title:
    st.subheader("Saisie des segments")
with col_reset:
    st.button("Reinitialiser", type="secondary", use_container_width=True, help="Effacer tous les segments", key="btn_reset_segments", on_click=reset_segments)

segments_out = []
for i in range(len(st.session_state.segments)):
    with st.container(border=True):
        hl, hr = st.columns([6, 4])
        with hl:
            st.markdown(f"##### Segment {i+1}")
        with hr:
            mode_options = list(MODE_LABELS)
            current_mode = st.session_state.segments[i].get("mode", mode_options[0])
            if current_mode not in mode_options:
                current_mode = mode_options[0]
            mode = st.selectbox("Mode de transport", options=mode_options, index=mode_options.index(current_mode), format_func=MODE_LABELS.get, key=f"mode_select_{i}")
            st.session_state.segments[i]["mode"] = mode

        c1, c2 = st.columns(2)
        with c1:
            if st.session_state.get(f"origin_autofill_{i}", False):
                st.markdown("**Origine** (repris du segment precedent)", unsafe_allow_html=True)
            else:
                st.markdown("**Origine**")
            o = unified_location_input("origin", i, "Origine", show_airports=(mode == "Aerien"), show_ports=(mode == "Maritime"))
        with c2:
            st.markdown("**Destination**")
            d = unified_location_input("dest", i, "Destination", show_airports=(mode == "Aerien"), show_ports=(mode == "Maritime"))

        # Si l'utilisateur modifie l'origine par rapport à la source de chainage, enlever le badge et verrouiller
        if st.session_state.get(f"origin_autofill_{i}", False) and i > 0:
            prev_sig = st.session_state.get(f"chain_src_signature_{i}")
            cur_sig  = _normalize_signature(o["display"], o["coord"])
            if prev_sig and cur_sig != prev_sig:
                st.session_state[f"origin_autofill_{i}"] = False
                st.session_state[f"origin_user_edited_{i}"] = (not _is_location_empty({"display": o["display"], "coord": o["coord"]}))

        if "weight_0" not in st.session_state:
            st.session_state["weight_0"] = st.session_state.segments[0]["weight"]

        if i == 0:
            weight_val = st.number_input("Masse brute de l’envoi (kg, emballages compris, pour tous les segments)", min_value=0.001, step=100.0, format="%.3f", key="weight_0")
        else:
            weight_val = st.session_state["weight_0"]
        st.session_state.segments[i]["weight"] = weight_val

        st.session_state.segments[i]["origin"] = {"query": o["query"], "display": o["display"], "iata": o["iata"], "coord": o["coord"]}
        st.session_state.segments[i]["dest"]   = {"query": d["query"], "display": d["display"], "iata": d["iata"], "coord": d["coord"]}

        # Chaine live D(i) -> O(i+1) robuste
        if i + 1 < len(st.session_state.segments):
            next_seg = st.session_state.segments[i + 1]
            dest_disp  = d["display"]; dest_coord = d["coord"]; dest_iata = d["iata"] or ""
            dest_valid = bool(dest_disp)
            next_origin      = next_seg.get("origin", {})
            next_empty       = _is_location_empty(next_origin)
            next_autofill    = bool(st.session_state.get(f"origin_autofill_{i+1}", False))
            next_user_locked = bool(st.session_state.get(f"origin_user_edited_{i+1}", False))
            next_prev_sig    = st.session_state.get(f"chain_src_signature_{i+1}")
            new_sig          = _normalize_signature(dest_disp, dest_coord)

            should_chain = False
            if dest_valid and not next_user_locked:
                if next_empty:
                    should_chain = True
                elif next_autofill and new_sig != next_prev_sig:
                    should_chain = True

            if should_chain:
                next_seg["origin"]["display"] = dest_disp
                next_seg["origin"]["coord"]   = dest_coord
                next_seg["origin"]["iata"]    = dest_iata
                next_seg["origin"]["query"]   = dest_disp

                j = i + 1
                st.session_state[f"origin_query_{j}"]   = dest_disp
                st.session_state[f"origin_display_{j}"] = dest_disp
                st.session_state[f"origin_coord_{j}"]   = dest_coord
                st.session_state[f"origin_iata_{j}"]    = dest_iata
                st.session_state[f"origin_autofill_{j}"]= True
                st.session_state[f"chain_src_signature_{j}"] = new_sig
                st.session_state[f"origin_user_edited_{j}"]  = False
                st.session_state.segments[j] = next_seg

        if _is_location_empty(st.session_state.segments[i]["origin"]):
            st.session_state[f"origin_user_edited_{i}"] = False

        distance_help = {
            "Ferroviaire": "Kilométrage sur le réseau ferré, fourni par l’opérateur ou un distancier ferroviaire.",
            "Maritime": "Kilométrage de la route maritime port à port. Convertissez les milles nautiques en km (× 1,852).",
            "Routier": "Laissez vide pour une estimation OSRM.",
            "Aerien": "Laissez vide pour calculer la distance orthodromique entre les aéroports.",
        }
        distance_km = st.number_input(
            "Distance du segment (km)", min_value=0.001, value=None, format="%.3f",
            key=f"distance_km_{i}", help=distance_help[mode],
        )
        method = st.selectbox(
            "Donnée d’émissions",
            ("Estimation par profil", "Total du transporteur", "Facteur spécifique"),
            key=f"emission_method_{i}",
        )
        source = ""
        carrier_total = specific_factor = profile = volume = None
        additional_emissions = 0.0
        if method == "Total du transporteur":
            carrier_total = st.number_input(
                "Total du transporteur (kg CO₂e pour cet envoi et ce segment)",
                min_value=0.0, value=None, format="%.6f", key=f"carrier_total_{i}",
                help="Reprenez le montant couvrant les phases amont et de fonctionnement.",
            )
        elif method == "Facteur spécifique":
            specific_factor = st.number_input(
                "Facteur (kg CO₂e/t.km)", min_value=0.0, value=None,
                format="%.8f", key=f"specific_factor_{i}",
                help="Facteur représentatif du segment couvrant l’amont et le fonctionnement.",
            )
        elif mode == "Ferroviaire":
            c_traction, c_volume = st.columns(2)
            with c_traction:
                profile = st.selectbox(
                    "Traction ferroviaire", list(RAIL_FACTORS), index=None,
                    key=f"rail_traction_{i}",
                    help="Pour un trajet en France métropolitaine de traction inconnue, choisissez le profil mixte France.",
                )
            with c_volume:
                volume = st.number_input(
                    "Volume brut du fret (m³)", min_value=0.001, value=None,
                    format="%.3f", key=f"rail_volume_{i}",
                )
        elif mode == "Routier":
            profile = st.selectbox("Profil routier – gazole", list(ROAD_PROFILES), key=f"road_profile_{i}")
        elif mode == "Maritime":
            profile = st.selectbox("Profil du navire", list(SEA_FACTORS), key=f"sea_profile_{i}")
        else:
            st.caption("Profil indicatif avion tout-cargo selon la distance, d’après les repères DGAC 2016.")

        if method != "Estimation par profil":
            source = st.text_input("Source / référence du document", key=f"emission_source_{i}").strip()
        if method != "Total du transporteur":
            with st.expander("Fret sous température dirigée"):
                additional_emissions = st.number_input(
                    "Émissions additionnelles de fuites frigorigènes (kg CO₂e)",
                    min_value=0.0, value=0.0, format="%.3f", key=f"additional_emissions_{i}",
                    help="Ajoutez les fuites attribuées à l’envoi si elles sont absentes du facteur. Utilisez un total transporteur ou un facteur qui couvre aussi l’énergie du groupe froid.",
                )

        segments_out.append({
            "origin_display": o["display"],
            "destination_display": d["display"],
            "mode": mode,
            "weight": weight_val,
            "coord_o": o["coord"],
            "coord_d": d["coord"],
            "distance_km": distance_km,
            "method": method,
            "source": source,
            "carrier_total": carrier_total,
            "specific_factor": specific_factor,
            "profile": profile,
            "volume": volume,
            "additional_emissions": additional_emissions,
        })

with st.container():
    col_add, col_del = st.columns([1, 1])
    with col_add:
        if st.button("Ajouter un segment", use_container_width=True, key="btn_add_bottom"):
            add_segment_end()
            st.rerun()
    with col_del:
        if st.button("Supprimer le dernier segment", use_container_width=True, key="btn_del_bottom"):
            remove_last_segment()
            st.rerun()

# Carte interactive
st.subheader("Carte interactive")
col_map1, col_map2, col_map3 = st.columns([3, 2, 3])
with col_map1:
    map_style_label = st.selectbox(
        "Fond de carte (Web)",
        options=["Carto Voyager", "Carto Positron (clair)", "Carto Dark Matter (sombre)"],
        index=0
    )
MAP_STYLES = {
    "Carto Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    "Carto Positron (clair)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "Carto Dark Matter (sombre)": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
}
with col_map2:
    focus_choices = ["— Tous —"] + [f"Segment {i+1}" for i in range(len(segments_out))]
    focus_sel = st.selectbox("Focus segment", options=focus_choices, index=0)
with col_map3:
    show_icons = st.checkbox("Afficher les icones mode", value=True)

# Calcul
st.subheader("Calcul")
dossier_transport = st.text_input("N° dossier Transport (obligatoire) *", value=st.session_state.get("dossier_transport",""), placeholder="ex : TR-2025-001")
st.session_state["dossier_transport"] = (dossier_transport or "").strip()

unit = "kg"
st.caption(
    "Calcul segment par segment en kg CO₂e, amont de l’énergie et fonctionnement. "
    f"[Guide ministériel Information GES]({GES_GUIDE_URL}). "
    "Si le transporteur fournit un total pour le segment, reprenez-le tel quel. "
    "Pour un transport combiné, saisissez aussi les pré- et post-acheminements."
)
dossier = st.session_state["dossier_transport"]
calculation_inputs = (dossier, segments_out)
can_calculate = bool(dossier)
if not can_calculate:
    st.warning("Veuillez renseigner le N° dossier Transport avant de lancer le calcul.")

if st.button("Calculer les émissions totales", disabled=not can_calculate):
    st.session_state.pop("calculation_result", None)
    rows = []
    with st.spinner("Calcul en cours…"):
        for idx, segment in enumerate(segments_out, start=1):
            try:
                rows.append(calculate_segment(segment, idx))
            except ValueError as exc:
                st.error(f"Segment {idx} — {exc}")
    if len(rows) == len(segments_out):
        st.session_state.calculation_result = (calculation_inputs, rows)
    else:
        st.info("Complétez les segments signalés pour obtenir le total et les exports.")

result = st.session_state.get("calculation_result")
if result and result[0] == calculation_inputs:
    rows = result[1]
    total_distance = math.fsum(row["Distance (km)"] for row in rows if row["Distance (km)"] is not None)
    total_emissions = math.fsum(row["Emissions (kg CO2e)"] for row in rows)
    if rows:
        df = pd.DataFrame(rows)
        st.success(f"{len(rows)} segment(s) · {distance_summary(total_distance, rows)} · {total_emissions:.2f} kg CO₂e")
        st.dataframe(
            df.drop(columns=["lat_o", "lon_o", "lat_d", "lon_d", "route_coords"]),
            column_config={
                "Distance (km)": st.column_config.NumberColumn(format="%.2f"),
                "Activité (t.km)": st.column_config.NumberColumn(format="%.3f"),
                "Facteur (kg CO2e/t.km)": st.column_config.NumberColumn(format="%.8f"),
                "Emissions (kg CO2e)": st.column_config.NumberColumn(format="%.6f"),
            },
            use_container_width=True,
        )
        mapped_rows = [row for row in rows if row["lat_o"] is not None and row["lat_d"] is not None]
        if mapped_rows:
            st.caption("Sur la carte, les lignes directes relient les points géocodés à titre indicatif.")
        route_paths = [
            {"path": r["route_coords"], "name": f"Segment {r['Segment']} - {r['Mode']}"}
            for r in mapped_rows if r.get("route_coords")
        ]

        layers = []
        if route_paths:
            layers.append(pdk.Layer(
                "PathLayer", data=route_paths, get_path="path", get_color=[187,147,87,220],
                width_scale=1, width_min_pixels=4, pickable=True
            ))

        straight_lines = []
        for r in mapped_rows:
            if not r.get("route_coords"):
                straight_lines.append({
                    "from":[r["lon_o"], r["lat_o"]],
                    "to":  [r["lon_d"], r["lat_d"]],
                    "name": f"Segment {r['Segment']} - {r['Mode']}",
                    "color": [133, 61, 171, 200] if mode_to_category(r["Mode"]) == "ferroviaire" else [120, 120, 120, 160],
                })
        if straight_lines:
            layers.append(pdk.Layer(
                "LineLayer", data=straight_lines, get_source_position="from", get_target_position="to",
                get_width=3, get_color="color", pickable=True
            ))

        points, labels = [], []
        for r in mapped_rows:
            points += [
                {"position":[r["lon_o"], r["lat_o"]], "name":f"S{r['Segment']} - Origine", "color":[0,122,255,220]},
                {"position":[r["lon_d"], r["lat_d"]], "name":f"S{r['Segment']} - Destination","color":[220,66,66,220]},
            ]
            labels += [
                {"position":[r["lon_o"], r["lat_o"]], "text":f"S{r['Segment']} O", "color":[0,122,255,255]},
                {"position":[r["lon_d"], r["lat_d"]], "text":f"S{r['Segment']} D", "color":[220,66,66,255]},
            ]
        if points:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=points, get_position="position", get_fill_color="color",
                get_radius=20000, radius_min_pixels=2, radius_max_pixels=60, pickable=True,
                stroked=True, get_line_color=[255,255,255], line_width_min_pixels=1
            ))
        if labels:
            layers.append(pdk.Layer(
                "TextLayer", data=labels, get_position="position", get_text="text", get_color="color",
                get_size=16, size_units="pixels", get_text_anchor="start", get_alignment_baseline="top", background=False
            ))

        if show_icons:
            icons = []
            for r in mapped_rows:
                cat = mode_to_category(r["Mode"])
                url = ICON_URLS.get(cat)
                if not url:
                    continue
                if r.get("route_coords"):
                    coords_poly = r["route_coords"]; mid_index = len(coords_poly)//2
                    lon_mid, lat_mid = coords_poly[mid_index][0], coords_poly[mid_index][1]
                else:
                    lon_mid = (r["lon_o"] + r["lon_d"]) / 2.0
                    lat_mid = (r["lat_o"] + r["lat_d"]) / 2.0
                icons.append({
                    "position":[lon_mid,lat_mid],
                    "name":f"S{r['Segment']} - {cat.capitalize()}",
                    "icon":{"url":url, "width":64, "height":64, "anchorY":64, "anchorX":32}
                })
            if icons:
                layers.append(pdk.Layer(
                    "IconLayer", data=icons, get_icon="icon", get_position="position",
                    get_size=28, size_units="pixels", pickable=True
                ))

        # Vue
        def compute_view_for_segment(r):
            mid_lat = (r["lat_o"] + r["lat_d"]) / 2.0
            mid_lon = (r["lon_o"] + r["lon_d"]) / 2.0
            span_deg = max(
                abs(r["lat_d"]-r["lat_o"]),
                abs(r["lon_d"]-r["lon_o"])*max(0.3, math.cos(math.radians(mid_lat)))
            )
            zoom = 6 if span_deg < 1.5 else (5 if span_deg < 4 else (4 if span_deg < 10 else 3))
            return pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=zoom)

        if focus_sel != "— Tous —":
            idx = int(focus_sel.split()[-1])
            sel_row = next((r for r in mapped_rows if r["Segment"]==idx), None)
            view = compute_view_for_segment(sel_row) if sel_row else pdk.ViewState(latitude=48.85, longitude=2.35, zoom=3)
        else:
            all_lats, all_lons = [], []
            for r in mapped_rows:
                all_lats += [r["lat_o"], r["lat_d"]]
                all_lons += [r["lon_o"], r["lon_d"]]
            if route_paths:
                for dct in route_paths:
                    if dct.get("path"):
                        all_lats += [pt[1] for pt in dct["path"]]
                        all_lons += [pt[0] for pt in dct["path"]]
            if not all_lats or not all_lons:
                view = pdk.ViewState(latitude=48.8534, longitude=2.3488, zoom=3)
            else:
                min_lat, max_lat = min(all_lats), max(all_lats)
                min_lon, max_lon = min(all_lons), max(all_lons)
                mid_lat = (min_lat+max_lat)/2; mid_lon = (min_lon+max_lon)/2
                span_lat = max(1e-6, max_lat-min_lat); span_lon = max(1e-6, max_lon-min_lon)
                span_lon_equiv = span_lon*max(0.1, math.cos(math.radians(mid_lat)))
                zoom_x = math.log2(360.0/max(1e-6, span_lon_equiv))
                zoom_y = math.log2(180.0/max(1e-6, span_lat))
                zoom = max(1.0, min(15.0, min(zoom_x, zoom_y)))
                view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=float(zoom))

        st.pydeck_chart(pdk.Deck(
            map_style=MAP_STYLES[map_style_label],
            initial_view_state=view,
            layers=layers,
            tooltip={"text": "{name}"}
        ))

        # Exports
        df_export = df.drop(columns=["lat_o","lon_o","lat_d","lon_d","route_coords"]).copy()
        dossier_val = st.session_state.get("dossier_transport","")
        df_export.insert(0, "N° dossier Transport", dossier_val)

        csv = df_export.to_csv(index=False).encode("utf-8")
        safe_suffix = "".join(c if (c.isalnum() or c in "-_") else "_" for c in dossier_val.strip())
        safe_suffix = f"_{safe_suffix}" if safe_suffix else ""
        filename_csv = f"resultats_co2_multimodal{safe_suffix}.csv"
        filename_pdf = f"rapport_co2_multimodal{safe_suffix}.pdf"

        st.subheader("Exporter")
        pdf_base_choice = st.selectbox(
            "Fond de carte du PDF",
            options=PDF_BASEMAP_LABELS,
            index=0,
            help="Identique a la carte Web utilise le style Carto (Voyager/Positron/Dark Matter)."
        )
        detail_levels = {
            "Standard (leger, rapide)": {"dpi": 180, "max_zoom": 7},
            "Detaille (equilibre)":     {"dpi": 220, "max_zoom": 9},
            "Ultra (fin mais plus lent)": {"dpi": 280, "max_zoom": 10},
        }
        quality_label = st.selectbox(
            "Qualite de rendu PDF",
            options=list(detail_levels.keys()),
            index=1,
            help="Ajuste la finesse du fond de carte: DPI et niveau de zoom."
        )
        detail_params = detail_levels[quality_label]

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Telecharger le detail (CSV)", data=csv, file_name=filename_csv, mime="text/csv", on_click="ignore")
        with c2:
            try:
                with st.spinner("Generation du PDF..."):
                    pdf_buffer = generate_pdf_report(
                        dossier_val=dossier_val,
                        total_distance=total_distance,
                        total_emissions=total_emissions,
                        unit=unit,
                        rows=rows,
                        pdf_basemap_choice_label=pdf_base_choice,
                        ne_scale=NE_SCALE_DEFAULT,
                        pdf_theme=PDF_THEME_DEFAULT,
                        pdf_icon_size_px=24,
                        web_map_style_label=map_style_label,
                        detail_params=detail_params
                    )
                st.download_button("Telecharger le rapport PDF", data=pdf_buffer, file_name=filename_pdf, mime="application/pdf", on_click="ignore")
            except Exception as e:
                st.error(f"Erreur lors de la generation du PDF : {e}")
                import traceback; st.code(traceback.format_exc())
