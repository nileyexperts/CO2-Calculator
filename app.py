
# -*- coding: utf-8 -*-
# Calculateur CO2 multimodal - NILEY EXPERTS
# Application Streamlit avec export PDF mono-page.

import os
import io
import re
import time
import math
import heapq
from collections import defaultdict
import unicodedata
import json
import base64
import zlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
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

try:
    import searoute
except ImportError:
    searoute = None

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
from matplotlib import patheffects
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

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
    "Porteur 12 t – marchandises diverses": (0.240, 1.80),
    "Utilitaire 3,5 t – express colis": (0.160, 0.46),
}
ROAD_DIESEL_KG_CO2E_PER_L = 3.16
# Representative payload thresholds (kg), deliberately rounded for vehicle selection.
# They select an emissions profile and do not represent a vehicle's certified payload.
ROAD_BY_PAYLOAD = (
    (1400, "Utilitaire 3,5 t – express colis"),
    (4500, "Porteur 12 t – marchandises diverses"),
    (9500, "Porteur 19 t – messagerie"),
    (25000, "Articulé 40 t – longue distance"),
)


def automatic_road_profile(weight_kg):
    return next(
        (profile for capacity, profile in ROAD_BY_PAYLOAD if weight_kg <= capacity),
        ROAD_BY_PAYLOAD[-1][1],
    )
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
def _default_segment(mode=None, weight=None):
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
            "country": prev_dest.get("country", ""),
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
        st.session_state.pop("generated_report", None)
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

def compute_distance_km(coord1, coord2) -> float:
    return great_circle(coord1, coord2).km

def compute_emissions(distance_km: float, weight_tonnes: float, factor_kg_per_tkm: float) -> float:
    return distance_km * weight_tonnes * factor_kg_per_tkm

@st.cache_data(show_spinner=False, ttl=6*60*60)
def osrm_route(coord1, coord2, base_url: str = "https://router.project-osrm.org"):
    lon1, lat1 = coord1[1], coord1[0]
    lon2, lat2 = coord2[1], coord2[0]
    url = f"{base_url.rstrip('/')}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "full", "alternatives": "false", "annotations": "false", "geometries": "geojson"}
    r = requests.get(url, params=params, timeout=(4, 12))
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


# SNCF Réseau and Trainline datasets are published under the ODbL.
SNCF_API = "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets"
TRAINLINE_STATIONS_URL = "https://raw.githubusercontent.com/trainline-eu/stations/master/stations.csv"
EU_RAIL_COUNTRIES = frozenset(
    "AL AT BA BE BG CH CZ DE DK EE ES FI FR GB GR HR HU IE IT LT LU LV ME MK NL NO PL PT RO RS SE SI SK TR XK".split()
)
RAIL_ELECTRIC_FR = "Électrique – France métropolitaine"
RAIL_ELECTRIC_EU = "Électrique – Europe hors France"
RAIL_DIESEL = "Diesel – gazole non routier"
RAIL_MIX_FR = next(iter(RAIL_FACTORS))


def sncf_export(dataset, extension, **params):
    response = requests.get(f"{SNCF_API}/{dataset}/exports/{extension}", params=params, timeout=(4, 20))
    response.raise_for_status()
    return response


@st.cache_data(show_spinner=False, ttl=3600)
def load_stations():
    stations = []
    try:
        document = sncf_export(
            "liste-des-gares", "geojson", select="code_uic,libelle,commune,fret,geo_shape",
        ).json()
        for feature in document["features"]:
            fields = feature["properties"]
            geometry = feature.get("geometry") or {}
            if geometry.get("type") != "Point" or not fields.get("libelle"):
                continue
            lon, lat = geometry["coordinates"][:2]
            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            stations.append({
                "name": fields["libelle"], "city": fields.get("commune") or "",
                "code": str(fields.get("code_uic") or ""), "country": "FR",
                "freight": fields.get("fret") == "O", "coord": (lat, lon),
            })
    except (requests.RequestException, ValueError, KeyError, TypeError):
        pass

    try:
        response = requests.get(TRAINLINE_STATIONS_URL, timeout=(4, 12))
        response.raise_for_status()
        catalog = pd.read_csv(io.BytesIO(response.content), sep=";", dtype=str).fillna("")
        catalog = catalog[
            (catalog["is_city"] != "t") & (catalog["is_airport"] != "t")
            & (catalog["latitude"] != "") & (catalog["longitude"] != "")
        ]
        has_french_catalog = bool(stations)
        for row in catalog.itertuples(index=False):
            if has_french_catalog and row.country == "FR":
                continue
            stations.append({
                "name": row.name, "city": "", "code": row.uic8_sncf or row.uic,
                "country": row.country, "freight": False,
                "coord": (float(row.latitude), float(row.longitude)),
            })
    except (requests.RequestException, ValueError, KeyError, TypeError):
        pass

    unique = {}
    for station in stations:
        name = _station_search_text(station["name"])
        city = _station_search_text(station["city"])
        key = (station["country"], station["code"] or name, city or tuple(round(v, 2) for v in station["coord"]))
        if key not in unique or station["freight"]:
            unique[key] = {**station, "search_name": name, "search_city": city}
    return list(unique.values())


def _station_search_text(value):
    return " ".join(re.findall(r"\w+", _normalize_no_diacritics(value), re.UNICODE))


def search_stations(query, limit=12):
    name, separator, code = query.partition(" | UIC ")
    query = _station_search_text(code if separator and code.isdigit() else name.split(" · ", 1)[0])
    if len(query) < 2:
        return []
    matches = []
    for station in load_stations():
        code, name, city = station["code"], station["search_name"], station["search_city"]
        if query.isdigit():
            if not code.startswith(query):
                continue
            score = 0 if code == query or (len(code) == 8 and code[:-1] == query) else 1
        elif query == name:
            score = 0
        elif query == city or name.startswith(query):
            score = 1
        elif all(part in f"{name} {city} {station['country'].lower()}" for part in query.split()):
            score = 2
        else:
            continue
        matches.append((score, not station["freight"], name, len(matches), station))
    return [match[-1] for match in sorted(matches)[:limit]]


def railway_km(point_a, point_b):
    """Short-arc distance between two nearby WGS84 vertices (lon, lat)."""
    lon_a, lat_a = point_a
    lon_b, lat_b = point_b
    x = (lon_b - lon_a) * math.cos(math.radians((lat_a + lat_b) / 2))
    return 111.195 * math.hypot(x, lat_b - lat_a)


def kilometre_marker(value):
    match = re.fullmatch(r"(\d+)([+-])(\d{3})", str(value).strip())
    if match is None:
        return None
    km, sign, metres = match.groups()
    return int(km) + (1 if sign == "+" else -1) * int(metres) / 1000


def build_rail_graph(active_features, electrified_features):
    electrified = defaultdict(list)
    for feature in electrified_features:
        props = feature["properties"]
        start, end = kilometre_marker(props.get("pkd")), kilometre_marker(props.get("pkf"))
        if start is not None and end is not None and props.get("elect"):
            key = (str(props.get("code_ligne")), str(props.get("rg_troncon")))
            electrified[key].append((min(start, end), max(start, end)))

    graph = defaultdict(list)
    endpoints = set()
    vertex_lines = defaultdict(set)
    for feature in active_features:
        props = feature["properties"]
        label = _normalize_no_diacritics(str(props.get("lib_ligne") or ""))
        if re.search(r"\blgv\b|ligne a grande vitesse", label):
            continue
        shape = feature.get("geometry") or {}
        kind = shape.get("type")
        geometry = shape.get("coordinates", [])
        lines = [geometry] if kind == "LineString" else geometry if kind == "MultiLineString" else []
        pk_start, pk_end = kilometre_marker(props.get("pkd")), kilometre_marker(props.get("pkf"))
        line_key = (str(props.get("code_ligne")), str(props.get("rg_troncon")))
        spans = electrified[line_key]
        polylines = [
            [tuple(round(x, 5) for x in point[:2]) for point in line]
            for line in lines if len(line) > 1
        ]
        begin = props.get("x_d_wgs84"), props.get("y_d_wgs84")
        if polylines and all(isinstance(value, (int, float)) for value in begin):
            if railway_km(begin, polylines[0][0]) > railway_km(begin, polylines[-1][-1]):
                polylines = [list(reversed(line)) for line in reversed(polylines)]
        lengths = [[railway_km(a, b) for a, b in zip(line, line[1:])] for line in polylines]
        total = sum(map(sum, lengths))
        if not total:
            continue
        accumulated = 0.0
        for line, distances in zip(polylines, lengths):
            endpoints.update((line[0], line[-1]))
            for a, b, km in zip(line, line[1:], distances):
                if not km:
                    continue
                electric_share = None
                if pk_start is not None and pk_end is not None:
                    start = pk_start + (pk_end - pk_start) * accumulated / total
                    end = pk_start + (pk_end - pk_start) * (accumulated + km) / total
                    lo, hi = sorted((start, end))
                    # Merge clipped ranges to avoid counting multiple voltage listings twice.
                    covered = 0.0
                    cursor = lo
                    for left, right in sorted(spans):
                        first, last = max(lo, left, cursor), min(hi, right)
                        if last > first:
                            covered += last - first
                            cursor = last
                    electric_share = covered / (hi - lo) if hi > lo else None
                accumulated += km
                graph[a].append((b, km, electric_share))
                graph[b].append((a, km, electric_share))
                vertex_lines[a].add(line_key)
                vertex_lines[b].add(line_key)

    # Some geometries stop a few metres apart at a connection.
    cell = 0.001
    grid = defaultdict(list)
    for point in graph:
        x, y = (math.floor(value / cell) for value in point)
        grid[x, y].append(point)
    for point in sorted(endpoints & graph.keys()):
        x, y = (math.floor(value / cell) for value in point)
        nearby = (
            other for i in range(x - 1, x + 2) for j in range(y - 1, y + 2)
            for other in grid[i, j] if vertex_lines[point].isdisjoint(vertex_lines[other])
        )
        km, closest = min(((railway_km(point, other), other) for other in nearby), default=(math.inf, None))
        if km <= 0.035:
            graph[point].append((closest, km, None))
            graph[closest].append((point, km, None))
    return graph


@st.cache_resource(show_spinner=False, ttl=24 * 3600)
def load_french_rail_graph():
    active = sncf_export("lignes-par-statut", "geojson", where="statut = 'Exploitée'").json()["features"]
    electric = sncf_export("liste-des-lignes-electrifiees", "geojson").json()["features"]
    if not active or not electric:
        raise ValueError("Données ferroviaires SNCF incomplètes.")
    graph = build_rail_graph(active, electric)
    nodes = list(graph)
    return graph, nodes, np.asarray(nodes)


def shortest_rail_path(graph, origin, destination):
    queue = [(0.0, origin)]
    distances = {origin: 0.0}
    previous = {}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances[node]:
            continue
        if node == destination:
            path, electric, unknown = [destination], 0.0, 0.0
            while node != origin:
                node, km, share = previous[node]
                path.append(node)
                if share is None:
                    unknown += km
                else:
                    electric += km * share
            return cost, electric, unknown, list(reversed(path))
        for next_node, km, share in graph[node]:
            new_cost = cost + km
            if new_cost < distances.get(next_node, math.inf):
                distances[next_node] = new_cost
                previous[next_node] = node, km, share
                heapq.heappush(queue, (new_cost, next_node))
    return None


@st.cache_data(show_spinner=False, ttl=24 * 3600, max_entries=300)
def french_rail_estimate(coord_o, coord_d):
    try:
        graph, nodes, coordinates = load_french_rail_graph()
        if not nodes:
            return None
        snapped = []
        for lat, lon in (coord_o, coord_d):
            delta = coordinates - (lon, lat)
            delta[:, 0] *= math.cos(math.radians(lat))
            index = int(np.argmin(np.sum(delta ** 2, axis=1)))
            node = nodes[index]
            if railway_km((lon, lat), node) > 8:
                return None
            snapped.append(node)
        result = shortest_rail_path(graph, *snapped)
        if not result:
            return None
        km, electric, unknown, path = result
        direct = compute_distance_km(coord_o, coord_d)
        if km <= 0 or km < direct * 0.8 or km > max(20.0, direct * 3):
            return None
        return {
            "km": km, "geometry": path,
            "known_share": max(0.0, 1 - unknown / km),
            "electric_share": electric / (km - unknown) if km > unknown else 0.0,
            "source": "SNCF Réseau",
        }
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


@st.cache_data(show_spinner=False, ttl=24 * 3600, max_entries=300)
def open_rail_estimate(coord_o, coord_d):
    endpoint = os.getenv("NILEY_RAIL_ROUTING_URL", "https://routing.openrailrouting.org/route")
    params = [
        ("profile", "non_tgv"), ("points_encoded", "false"), ("details", "electrified"),
        ("point", f"{coord_o[0]},{coord_o[1]}"), ("point", f"{coord_d[0]},{coord_d[1]}"),
    ]
    try:
        response = requests.get(endpoint, params=params, timeout=(4, 15))
        response.raise_for_status()
        path = response.json()["paths"][0]
        km = float(path["distance"]) / 1000
        points = [point[:2] for point in path["points"]["coordinates"]]
        if not math.isfinite(km) or km <= 0 or len(points) < 2:
            return None
        if railway_km(coord_o[::-1], points[0][:2]) > 8 or railway_km(coord_d[::-1], points[-1][:2]) > 8:
            return None
        electrical_km = diesel_km = 0.0
        for start, end, value in path.get("details", {}).get("electrified", []):
            part = math.fsum(railway_km(a[:2], b[:2]) for a, b in zip(points[start:end], points[start + 1:end + 1]))
            tag = str(value).lower()
            if tag in ("contact_line", "rail", "yes"):
                electrical_km += part
            elif tag == "no":
                diesel_km += part
        known_km = electrical_km + diesel_km
        return {
            "km": km, "geometry": points, "known_share": min(1.0, known_km / km),
            "electric_share": electrical_km / known_km if known_km else 0.0,
            "source": "OpenRailRouting / OpenStreetMap",
        }
    except (requests.RequestException, ValueError, KeyError, TypeError, IndexError):
        return None


def rail_profile_estimate(segment):
    countries = segment.get("country_o"), segment.get("country_d")
    french = countries == ("FR", "FR")
    european = all(country in EU_RAIL_COUNTRIES for country in countries)
    if french:
        default_share = 0.9
        default_label = "mix fret France (90 % électrique)"
    elif european:
        default_share = 0.8
        default_label = "mix fret européen IRG-Rail 2024 (80 % électrique)"
    else:
        default_share = 0.0
        default_label = "facteur diesel de repli"
    electric_profile = RAIL_ELECTRIC_FR if french else RAIL_ELECTRIC_EU
    coord_o, coord_d = segment["coord_o"], segment["coord_d"]

    route = None
    if coord_o and coord_d:
        if french:
            route = french_rail_estimate(coord_o, coord_d)
        if not route and segment["distance_km"] is None:
            route = open_rail_estimate(coord_o, coord_d)

    if not route or route["known_share"] == 0:
        return route, default_share, electric_profile, f"{default_label}, traction du trajet inconnue"
    known_share = route["known_share"]
    share = route["electric_share"] * known_share + default_share * (1 - known_share)
    traction = "électrique" if share >= 0.999 else "diesel" if share <= 0.001 else "mixte"
    label = f"traction {traction} estimée ({share:.0%} électrique, {route['source']})"
    if known_share < 0.99:
        label += f", {default_label} sur la part non qualifiée"
    if share > 0 and not european:
        label += ", facteur électrique européen utilisé comme approximation"
    return route, share, electric_profile, label


def segment_distance(segment, rail_estimate=None):
    coord_o, coord_d = segment["coord_o"], segment["coord_d"]
    if segment["distance_km"] is not None:
        return segment["distance_km"], "Distance renseignée", None
    if segment["carrier_total"] is not None:
        return None, "Distance non renseignée", None
    if not coord_o or not coord_d:
        raise ValueError("choisissez deux lieux dans les résultats ou renseignez la distance dans Données du transporteur.")
    mode = segment["mode"]
    if mode == "Routier":
        try:
            route = osrm_route(coord_o, coord_d)
            if route["distance_km"] > 0:
                return route["distance_km"], "Distance routière estimée (OSRM)", route["coords"]
        except (requests.RequestException, ValueError, TypeError, KeyError):
            pass
        raise ValueError("itinéraire routier indisponible. Vérifiez les lieux ou renseignez la distance du transporteur.")
    if mode == "Ferroviaire":
        if rail_estimate:
            return rail_estimate["km"], f"Itinéraire ferré calculé ({rail_estimate['source']})", rail_estimate["geometry"]
        raise ValueError("itinéraire ferré indisponible. Choisissez des gares raccordées ou renseignez la distance du transporteur.")
    if mode == "Maritime":
        if searoute is not None:
            try:
                route = searoute.searoute(coord_o[::-1], coord_d[::-1], units="km")
                distance = route["properties"]["length"]
                if distance > 0:
                    return distance, "Route maritime estimée (Searoute)", [point[:2] for point in route["geometry"]["coordinates"]]
            except (ValueError, KeyError, TypeError, RuntimeError):
                pass
        raise ValueError("route maritime indisponible. Indiquez la distance port à port fournie par le transporteur.")
    return compute_distance_km(coord_o, coord_d), "Distance orthodromique (great circle)", None


def emission_factor(segment, distance_km, rail_profile=None):
    mode = segment["mode"]
    if segment["method"] == "Facteur spécifique":
        if segment["specific_factor"] is None:
            raise ValueError("renseignez le facteur d’émission spécifique.")
        return segment["specific_factor"], "Méthode spécifique", segment["source"]
    if mode == "Ferroviaire":
        volume = segment["volume"]
        density = segment["weight"] / volume if volume else None
        density_index = 1
        if density is not None:
            density_index = 0 if density < 250 else 1 if density < 400 else 2
        density_label = f"{RAIL_DENSITY_LABELS[density_index]} kg/m³"
        if density is None:
            density_label += " (classe supposée)"
        if segment["profile"]:
            factor = RAIL_FACTORS[segment["profile"]][density_index]
            traction = segment["profile"]
        else:
            route, electric_share, electric_profile, traction = rail_profile
            if route is None and electric_share == 0.9 and electric_profile == RAIL_ELECTRIC_FR:
                factor = RAIL_FACTORS[RAIL_MIX_FR][density_index]
            else:
                electric = RAIL_FACTORS[electric_profile][density_index]
                diesel = RAIL_FACTORS[RAIL_DIESEL][density_index]
                factor = electric_share * electric + (1 - electric_share) * diesel
        source = f"Train de fret de référence (1 000 t) · {traction} · {density_label} · Guide GES, tab. 10"
        return factor / 1000, "Estimation ferroviaire", source
    if mode == "Routier":
        consumption, average_load = ROAD_PROFILES[segment["profile"]]
        factor = consumption * ROAD_DIESEL_KG_CO2E_PER_L / average_load
        profile = segment["profile"]
        if segment["weight"] > ROAD_BY_PAYLOAD[-1][0] and profile == ROAD_BY_PAYLOAD[-1][1]:
            profile += " (estimation pour charge répartissable, au-delà de 25 t)"
        return factor, "Niveau 1", f"Guide GES, tab. 17–18 – {profile}"
    if mode == "Maritime":
        factor = SEA_FACTORS[segment["profile"]] / 1000
        return factor, "Niveau 1", f"Guide GES, tab. 16 – {segment['profile']}"
    # The small-aircraft reference represents a narrow-body cargo up to ~24 t
    # of structural payload. Beyond these bands use the applicable DGAC reference.
    if segment["weight"] <= 24000 and distance_km <= 3704:
        factor = 1.914 if distance_km <= 1000 else 2.609
        return factor, "Estimation tout-cargo", "Guide GES, tab. 7 (DGAC 2016) – < 100 t MMD, cargo court/moyen-courrier"
    for maximum_km, factor, profile in AIR_CARGO_PROFILES:
        if distance_km <= maximum_km:
            return factor, "Estimation tout-cargo", f"Guide GES, tab. 7 (DGAC 2016) – {profile}"


def calculate_segment(segment, index):
    if not segment["origin_display"] or not segment["destination_display"]:
        raise ValueError("renseignez l’origine et la destination.")
    if segment["weight"] is None or segment["weight"] <= 0:
        raise ValueError("renseignez la masse brute de l’envoi.")
    if segment["method"] != "Estimation par profil" and not segment["source"]:
        raise ValueError("renseignez la source ou la référence du document.")
    if segment["method"] == "Total du transporteur" and segment["carrier_total"] is None:
        raise ValueError("renseignez le total en kg CO₂e du transporteur.")

    rail_profile = None
    if segment["mode"] == "Ferroviaire" and segment["carrier_total"] is None:
        if segment["distance_km"] is None or (
            segment["method"] == "Estimation par profil" and segment["profile"] is None
        ):
            rail_profile = rail_profile_estimate(segment)
    distance, distance_source, route_coords = segment_distance(
        segment, rail_profile[0] if rail_profile else None,
    )
    tonnes_km = (segment["weight"] / 1000) * distance if distance is not None else None
    factor = None
    if segment["carrier_total"] is not None:
        emissions = segment["carrier_total"]
        method, source = "Total du transporteur", segment["source"]
        additional = 0.0
    else:
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("renseignez une distance supérieure à zéro.")
        factor, method, source = emission_factor(segment, distance, rail_profile)
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
PDF_BASEMAP_MATCH = "Style de l’aperçu"
PDF_BASEMAP_LOCAL = "Fond clair"
MAP_THEME = {
    "Carto Voyager": ("rastertiles/voyager", "#dceaf3", "#f8f6ee", "#c7ccd0", "#5a7081"),
    "Carto Positron (clair)": ("light_all", "#e2eaf1", "#f8f9fa", "#cbd4dc", "#586a78"),
    "Carto Dark Matter (sombre)": ("dark_all", "#1c2a3a", "#2c3a48", "#485766", "#cbd6e2"),
}
ROUTE_COLORS = {
    "routier": "#2477c9", "ferroviaire": "#7549ad",
    "maritime": "#008275", "aerien": "#c73c3c",
}
ROUTE_LABELS = {"routier": "Route", "ferroviaire": "Rail", "maritime": "Mer", "aerien": "Air"}

# Natural Earth 1:50m country boundaries, public domain; simplified and packed.
# https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_50m_admin_0_countries.geojson
_WORLD_DATA = (
    'c-mB*OO9npa-O$RsPT=!@wfX>(m<ksCJDX(slh>XSE`v{RVJ8~34$DH%8BOMfl32u5w#PoME-xj`MFUIbe4r*KmK-ed)x1S|NHO%'
    '>&Kt|@b-s){_y=Huh*4Zt@ZJe=IctS*GKvL$M1jt`;zi{yyRT3M@iQZSkj|hD+}Dem8*72%Ldbm(!4sQsco<rRB5gPuerg?U=~<5'
    'nCA7!^GX8S{%lNDNV#vYtK)PD9M=Y~R#d7nTy-}#@_d5o?aHc<C{4TF-Zz-m++*lz2yFF{+iC(eoVJ+zQmF=+#u@^<z_CATRdvVQ'
    'BWs-sa}T{uD(rT0zpnFHPi`S?aIcHNJ)GV&hN&6%VQ}|0_TQ?Yxh>W0OY1PyZr3#Dq_k#%`mF1-xf_2;ZLz3J;LxHD)z$d+wp>rJ'
    '@|m_U8_XN5`mFCYD!WgCd2eiD9eUlnnO%3!yJTkE+urUx0QY*9%<}cIKg%B3oCh3xkh_AGdStb*Iy2@o^)~mZh`MvAx1n{=WH)%l'
    'UPS5A(DP)ol0cSkl|7uI1G6`FoOahz7aW+Y?P+FBGhbc3RnEiGa7xpOO6sC=`U<NPXZ`Q@{~o#_Z$JIvqwYt4Evx0FZtH9Bm$oi0'
    '$aA6JEq`9sn{xeTtKM(myKQ}L_PZnS?qIw-F7FS~^Ju+0hVKvOyCeJVAp4xQeNHRm9ebH_&1{u!P$_JnW|1o+Q)9Zy+V?%Od~pPH'
    'GZRbF)(w{3dM<3X_9GW&oF}DPlB^jQ4u-JIUTn3SYC5IY)X1(0xLc@PrnGUy-l+_z3WXMuO4$GqMJ?1VRG74@xivnk>aFuxVM!LB'
    '?T(w<&0KzecBzz+b**I67VeeO`cijjFNC*J0;zPyz~!I$2Du=~yLmS@l2#`*kQL+A)g9HJ2_-ZQGK0d{*t`doXVn<8Hfy%VqUsni'
    'q5U5nqr$2eVN}&ET8EW2vZ4tsJ}X%`xfd#^v{R}LSe&z>g<4ae<vrB>Svte<sb&k;#mKahv$AG$w=lixI$XNoT1gGPRd!RCf#Z2~'
    'RiNWbxPEqZp_{b9-SO&7SWwn{g}pVWeQWk~r;hQ|O_>d7*PB&O(cYk!nFQ@k*=0kImg;rgZtbUf+Nckx3G=DFr82%)IZ)ubTA+rN'
    'ggW%zt-VG?2Yioj*DD!lBdns9*WQB-F%e;h(zL{Sh&ISf!B=`0nRF>!u4}LP{>%%c_PMDVfAK|3>VG4wIL!xa^vkL1s+&0yDV@f6'
    'ri-_(&>I_C!I?U0StB<J*Y)(o_VKw6-rf(u!MYl%k?LMVJp!DL(s>Uk9bj|7nGY_mdS?oVqX?U`V!KoX1hc4Xy?dz+*8MNP{_@j@'
    'Pv1Xkd!1w7<MK%Db+?}P(irKsO@KT$GHQ>uao>jaPB)Pt`+}_v?Q50v&bPe(0X^-qyrI4NZR6h!?L}&PcQ&-~OA|qn+9gO|)IIAx'
    'wj`)*XfI%@`{Y&~i;YP7qw4%6Va=*5n1sga-1l@t*Q1U-F_(?K%zI7t7h@lvr2XGWX#ZOFn5zbqv|+Usm0c=JDpM;O(B7Q<sJodB'
    '?SFRP`IHrv?ar%KbVKQ}i;C*)_R8Mrq?dF<^8vD`Iy!q>hoJdj|DsIeV@=gmKzp;Q&i!74ENG9V?tiL~&0e*=E?J}B>%O7EngT7A'
    '4b^*=vGz@6+L^1wuf1q2TJg?UlJ;_H1ZjWK)mzbOjI1*4>R|Nyil<RDX6?+V`nKuUrulAyWdEvq8vQ6*D_W^q)5I5h1dT8D6{@}H'
    '3(cc%9O^#$(pZ*uFx3N9eX;LY9XfTRYR@(Lxtp&`qndkrnw~<4mc0(eeKh8)mhZF6VXvF=)GBGM8>=}5HI-@AIjfbm)*5P7=ZpAW'
    'C!=Ve_j98?RSor8MvL7jgEH4UU#WbJ_=;VpM!m-4{*}6|aQ61;Z&fqYrf<qZCvFdOm+8dqz1a11GWV!k(WtX-*uL_rS2ol*nff)_'
    'JJ<d1KK}Z@zkc`UZ$Ewi*og79%%M}4)9ctX*8{|r*Xyvq6A~4;7x>bRzJy~o!Y%yNR9Xd2fqS)7$`|GFY=M3^=G7aWG?R^llfX-$'
    'sFK8|y0M!BQC~w3UE+Ja-%ybo6zi~i-(VLR;8GFCr&J`vZbzUVI`GT}cgLPd*L!|tK%Jz;Xd4GeH|}bCWSd)|RNJzsD=K8aSc3{W'
    'b_~5=I*VHJt7{o1E7bMd@W`sJ!5n}(!Gj8#>=-SEvq`_s(5tVJW~|bfJlBGS(kbn|0_uu%ucYpErGhYbXE^f$)a2*B%NqqO!0`li'
    'sG0oK-D>*hnv+8hsMTmJsu;FZ8TtZKX&~3ycL+qb!5pVvU{fg-)EkKJT2;q#QCD}GYqnbuI9yj1`rU43vzCS~>gQCr+{JDut#*1#'
    'dN(f7UAqG93i%{qK9?pBT^A?0W8wM49#1Vn(m*`<<mJnB#PYFRY`~;-8XMEck-s}foz3FtvxR)-1FqNN@*Xql>hqh<H311;jYG?x'
    '*84n&l@vI+V}M+#>m05CX@TB(>#oE(+ml^vkQn5+zkT1oHFMmG#c|TvT#bt5KyA-{v13ig>hks4E9<)<=5_fJ>}#}`Qrmwo&wH=p'
    'oRn?sJv+!?pO4A%?^pdX*>=5jt|tm$dGhOu0k+-NV&x4rAO<t|ME@qfp^Z#KEOerRkq{L;ryM;c<ArwJyJ(P;)bu)CuEodhn1wgC'
    ';J4{j;^eGY8{s&>!r42e=`v;!nC})6Yp-?QAjm)mCD9KTT7;&+bdG!1P<iZp4l;x8Y9WK62N#ATF-r04AW+W;$8_+H6oYDkQztfb'
    'Q$Pi5ETV~1M+-bRm)`zV14l!VwS$E!L#Ktx$c749zPo_Xx+F15V9|X`L?h1YjI0^8YsQ_0VnV$zH!T!M()XcPBm6FK1(t^}4ZQ0c'
    '*0gei3)FO62P12`Ff)1ki@+5pG)*>I$1G4gWXq&oFrQCp(N2S){I)GkdxN_%u}~Y-ychdc)$wA+RU=tG@nRRR>m0-!hgNcnEBo&2'
    'fnuq(;Mo|UN?37$cdHoL%OjJh&Bs%GfA*OydQg3;3l%t?Vl4LCs&>VMY)*6DrC<-yxH*B9M9`iJz2BfZPTWVaH5*iKSt5Z9Zps_;'
    'W#jYBjcyQ3At}yg5Ex{VjDsT=pek$zN}oYHsNE>lI%Ffhb$=!jKf0*2VB{g8LJou~fyqc``5=_Nd{MWni<tv!<h+bP_B0DvQWRl^'
    '&fN!QcL5o^mwtC#E)xLiIBvIw16k1ZSvIz9^|UL@9kbBo&Dr=?E1BCkR7y4?1uT9B1m+rNJ)b+TSr`#RN-+k}3404y5DVF_ohaW8'
    'rkj*9u`t+8BPtVPL7Vlx+>>qAvoKjTbv@aa`1R0n#;#WoWZJeTxSsD|dpzr&ZX|5F{QQh@PHdl?W8T83aJrKGEAeoRs<HFD2v^+T'
    'JjeRfJ)c79(J1w=n~2KEs4gm;`jU8ErphEqj#28aT+>gw?$z{MUH2xbKYaS|uirlW`1bwdrPtR@SCxGl$LppD#Yvipe(7p*NxZgc'
    'u)3iHv>#Fl&b!D~q(`S7ZNGG=DM{W0ibRp%pu=k+cIlO8K!U#x-*rhG*ee^fDAj3Bl8AL(m87LpUBE;O<!)ElB1vX{cO2tr;3o-J'
    'dt<@rF`h_Gs(~0K%ed>tr)nBeQWr@-CFAH4D^7_+x?aj9T}RieyDl$5k6+{4Xi2COaoi@^%VSFXv!V8wnX)}|VRhBpzcS-%FRsJ`'
    '8`@Y(H1dY_Gbk2Bi=H2P$2PQYh<H~)<)N2U_4XGMAMS74hT4HV#{I2Mu|_){BD}n@x~F^GySBLO4ed>pEMH~i(P??Jdk=IVU9!36'
    'KK22w9_b!Iohyz<bt_0HD(!8aRWjBtwn<dZhW0%`Fx#cfsgkvJiIDX~Y-n#%r4OM-v2cR+!IX%p3u_g|CsbxgGLpjh#7Gv#QY68n'
    '3_+6wf?X_X<!tI*LU!EsU?Sh%(30dq5HbzwiDUMulfaAKfH?a41;H1mT8Pm{T@Q#xg@F1%S@S61g&N<c7?pUQe9n8LOsBK@X28TS'
    'J`qVR#wU^XKDyimpd$Ro8Q<Hhakfp+1h9`GXMQ7epZI;8Sew0h4~|0L9ZEpjrKU-YwTpg~aJ3J~S+gNT@q!vMY(dDafd-;oL9?GS'
    'J+AD?K+>3j(xX%$sa_<;<_if91ZBxa__aPX3-vZ1%`Hosw+!vmdl{OK4qDQv#NE2R&dC}&<&;V0+gYtinf6OOfWH+B*g1hxN_Otg'
    '+UIp2;m6D7i#DV?KJ;wpN!!pyg5sTC26trv?bQ&*-OyZ7miJz}OqB0XAA3o4gVl=G69NAaO(H}uXy4oAU-!P!hw6zF)~|a*CI1~d'
    'ed1U)w4WuA^bqR`;vQx1t{Bh_?UOGN)P{DS0qs$(L3=zI6%cYzr-d<Wj7pQRu3E1FA%$doz<ws}-jKc!)>YFUs!;B_jk+TIq{vYP'
    'F|^5xRRgOJ({)}#Gi|iX^Dh;p8@oK5WVTR+VgVc9@3|H#ZVE=cw!U7*)e;sxB%a++)?-H7;?oLc`<FEs#oiy`YSp7V8iY>B_dLaq'
    'xk&`{D(z>{5)Ad~rv=`r!SgI_(gr0`O>DfB&wF`v4;BxZ=GmoONor<;qGwWbX+Fyi>kPse$c9i(3~2Mr(SrTWKGQQeFEz%-Q7Rr+'
    'NLYuyoH0o0xJwHcMPu+ydA^<&9`O=QbMm@g6GF85t`Cb80(t0_+F*uEQM(-mxKZI+r><F4<FgbTvyhxhxsTN(TIZQBrq(uSNrdSD'
    '_}hLH6%90=O&1@0dUZb$H)whrv}gKt@Fq<-Lmn>^&XWq|%)$&r<FjDGd6*!*(+z^)e4qTnah5?k8E)_5jYG<i@(qG}5%Naez3S-R'
    'x}K25PUx}Z0(2?i7Ic$Piy_IFmI)Y9X^eR@n7fR}eE;olb<Y;4Ic<a-^@J2yNJ$5<&geSXpph<}$O19yAwnD16XZ6~USt%JUAKEy'
    '69P3LY4|1&s7x^+i@Jz$&{vlbWbz0G(R5R*Z=b%sef{>m80XiuX%|$i?X?WOl-1(jE>Y?W`WRUspi<X4*vF9C2<f}kP7;FjfeQJ+'
    '(o8k6OMRqr*B>aP#8#;e8hg}99gTeJ<q0>G6sS!hF?&aS31Qhgx;~C`?`UtOEmEEIgwOa*-}iw&SFh2AF;9_3eZ%vZU2Hfk12T1}'
    'k5FOIUuR4MEx+Xs1C_PjHNhE}Z?j?Aw<2HM(V4$6)2?4p<x~HREyg<C$3Q(3HUXN`bRPpMBH0Aw0=AI|b?8I%rs@ys_s6?7Pah~V'
    'd*K^2l!%Udm_+Dq?(!ZsE9#@)^SHnL&3G-9o>)xCZb&dT6&@oR)8RRxwnYTqXLkK`_YtryBv|LW{j--g@$flf|5{~zQJFUHB?cb|'
    '^ZG>BD$f4hw`c6x<|2dsIotg?Ut~rx7Vq0#|9qb~$0$Q5woXXAHtha0a0>0~yZUD?oRv26^L(%_aX&g|CC24CHhO7~ci*12So`X8'
    'jS)a){?fmxeBEbF+D6p@_T9v!qO#X_JNMmL=#<la{sS(Q54Vdk1A8}Y`7m8SzKymd(Up~$zR_DM<1rL|dLAqf>ynkJPNPl%rq22H'
    'v_;#CbGNX%TC58+h=p%`u|`PhIy6i|@6e%aHe7%^b^7C7o2L)Veo%+R_Y+m+v{{wj=S}fqOvU|!PE5$WJB&T{R{9o;EbgL>M35;y'
    '+dt^}FTVZjAAbGiA3y&5_3hL5k6vG|{orP?S+5%CCVDMT7?S)|(UC(>LKQnPL=ja8;e?)eAL1cidhd4;w0TIqzyl6b&+-s|Tt!hr'
    '?^NmMhKPtN(sGC(sbZFf=&~vT%<YMxfsEU{mZ1GKSLsoPw0c#qA4KPdidbk`z8+AYRrJ@UDX(Z*!YlWdXqj&(y<K9&RkknI$nct)'
    'hbZ`}Ne@u~RqXeW2}IS<hU9ovMAeW~u}W4tL<m$-6+@<#6~;b`dzMmQz#oema?6r!4N)gmWYLg_vx;JdV6Gyopo6Op=+NU|FIHnW'
    'RWH_F`el(Q1D>8TbzZLmE2DlTjL<)B`GPO++6C#6%?8N=Ld!lb&r)R?Ju_GH^@1Vtw_;71$KK?9Py|UpVW^9l(%BHH*cam~lx2?;'
    'Uajf~bje+qQJ>6huYI44MJ$;NF{-X(mp-&!V)k^BH`F$yy}UNX({(e7Y&v$?6r`4iqzbdTy?|ukB1&mPn}-#p1hKrKBJqkO4PAm3'
    '*+?$2o6<%U4ZIH|5*pQzZH%sD$5c>P6bxK6+%8E_k=;la-&bH-r+8-}^DYjsNMqV16DmSjPy$85%dVHK2<dk5kwt=4-ufcO9gJOM'
    'RtC*eq%+ynQW3Q4b-pMP1WAe_dq8NKqDL0WrN{zz7cqh+F_FsDUX$e%kW>Oih@%Uedv2o1i#Dob3MBTc7uIhdsV>XQBFYuQ+B+>q'
    'ri$?HlHu0`lg?euD%%}Z(U{1xs%X)!!_PhM+T<!Dy)H(giqGjf0##Rhm+7u*e8wO;f8r{usOBcMdX;%zldc`y)D{>?W*WlO6(fd)'
    'C58+eG?~q4?0T}_#7?8>lvD}Qo2=ikglRK(+fy_tb*c=k_OFsNRedWQMjtv&RaVk%(nnQ2Y-09KH|=>9YT3`I`q<cPKf1y%dyq|5'
    'p<t<|H@M2CEy{N1O?CrSJi8%1@Oz|^y66;(@UK;(4e@=o>NF*-fz%wE*b4ihE<5llo3&mg?b8#Agtbbb(;GH6MP<5WGU4r_Vyh5d'
    '?^u#-@}_t8k}~0g&BH#R+pDZ@k=TGwp+&%=+zRkyXVos<vf0KB>Ex(4-CA%CZ(h|C(`AlW#dG(Clu<&RRX7{MC1825S>i+p>#;Ku'
    ')(H953=Tt5DBOPDvi-6RA?zwa>d-){`28W?uqK2)@hgud?_CkcHS{A`hS#v@u#L+Y3X#$@{u*Mq0(0x!UWkvt^_R+0+?>1R=*KvW'
    '*Qyz>4NdTrf%VjP)H{#)-+uY_r*Cin^8Mqbwb#8YvV?dkd4ow>3o~0UO(LY1{m&*Hnhe!8EFn2oJTC=-ZR6Kc5p;{>+OPyxP3LOE'
    'B7GZT*8_`KDpEM^zBAvdq&51gV(IoV1r{&d5Xu!D!40dIs4wKC`dzS?kyC#Hi<N4_V-RWBAiOtB`FvNtsE^1VThmgO;mX`KM~qmD'
    '?9w(Y!fnGEKjJ#uNj4aqBEjB<8FOY$AlOUxw<5BOC>?#P_QkDh43u}5YmS)Z?{QMDmUkVskLCpAuwmxzxm=C<Vn&OMFK$?5d4p-R'
    'Uq*unE(l>7>Vyq@tul2gYgm9;bB(+p4{4n-UtZ^n@yvR|I=Fl97@H2e_R)3Py}zh?_P#Sm#@5)Q#26aunb?cw^`YEnMr7TF+q&1M'
    'x=oSgt+Fo&DVs#|T4$e0*8ifsdylj}g*}z_c*7b=$;^?RwSPmC&E(!|i4Ee+%S!QvJ8hmja_-iz|MvCm>(Af)w_ktx;n!cj{QAq!'
    '-~I66yWf5M<Co8$_8;;i*eu&?-q%O(+Uq7rnn;R}Z%8lQ!h}^etP@D~a`|!pY9647zsA~SKhm;x$%6JaN=_<h*(wHd;*{})zi+bd'
    '5E-x`5!!~NWy1a;em!+Xl*@9ZN$e}x{Dy>XDrTVmHR_6w#*nB<8h|XIP2}bkNT^tL7ZU<cz0Lv&_qe~k7xRDo_~Ga8e)sm}=YM$n'
    '@yG9FclnycZfrFAdTr)iltV0?tcI`(>Y8>l8zbZ31@pDp9ZXqHJr8!HR^dl1I&(qDpI{(l;F%C2i-by`?}7bjd~px)|HKeKzJ2<%'
    'p&!0|_vfF#{P4Y$Ij>`57V+@ybuW=tDF$1cI8xZI3S$p9R^gwgmhP_LVfs(RBGm#Zi6(19o)MrcociLn5s_|C*1L&D#k-zT>}b$A'
    'McunAjpuTsGDx|ouTf#<T4idGkOtp%?@ybJf@4+zYG@us!;~-6O4L1{N@!XazlcMw*c@XggYL_v&0~qWzukhr`}pC@pR@t{|GtDT'
    'fBWg<&sKhKA{P^c@w!i}j1zijzY*8z{7J_qlbT|g^m>A>5UO4ojNw9K-6G<msE=3L1}U|o(RI%_|Km5zTk9UqRj__5uSGBAeg7*)'
    'IfV_uhDjvW0d@Uq#VWS}Y99)VMW>}xSGTN4C|LC(;?wbL_YCE)-+uc1_TAf08%2Hm<J-r5aKFc2YcCJ-R(Uwd);}F$VE=|^Nkxc('
    'qQi0={ekCLwRh2F$*1LH_cy2Czx@2=>*w$O>f;|jYhU{7e(<px0wc;~w#B|+t;QqVZP1FoB0!W0x&|o6y6s@m{2?(zp=Nc&KwID0'
    '2$a}Zt-4fPQj}s4!qeWdoaDuiQpUj5pXeYkfklRR5AxU|RP48~MQFo!M5WU=M+OKwRw$-V-S`xZ3W7~Uoe2^GDPlBuD*X1DIVXIL'
    'y^<M^zV+I~O0KY&mIuc^qRql-eC=9Cz=qpLv!J?2lhH61&?(tWgncE8?17ry&F*$Q<b(sBX)=Ak1b6!rsV#(8MV=(BTR4aHct;>s'
    'WL4XG@i?NL)2>d{Mij+fNU&7N?kB*EQI-Zg8}I7J_Z!QJIYf|F^@858-XQy#+;z@3Rr6dv)V0G5B7_td)E53RwTkV%F4u|r=X=Sj'
    'VMBq(15*#d8Q4PqT+D(}2WFLW$$nvG-?bC%w#?tGWy`c{STdiEPE4>^Te}*b`tf}iligaezo%^R?E}~)U}}x^yX%}jC3kEU@1G{B'
    'qSf&4(_S=$?@ex(rqQ1D@9$}wxiyJP?r*f`%}SW1^=lRspZaIavPHp<0JsT8VIZZV&i3P7o%m+xX>W-=8=?+c;w%s8{&zVZj$Q0D'
    'O4MilyU!$n2WPH#Y-Fq~(Sz0PV|mv(eM%eLi#;u|wZUvePOn7*5F9L}@aqLt*3i-E#SXU!k4m+lr<m+wrO1?U%>H8?;Y!Jdr|KZc'
    'x7EUdMF82Y7S(Yt$$cf5d5=M7uvnoXd0!9T&I>_}l^Bu|_F`XBEnUZKW5cxBqP*Sn5_O1T_T4%>*Ajb;X-`q$P*mysdv4Mo^efhQ'
    'q;q$`>T!3v9{;vbj38UUqzbs4yXrGxk45&aEi20gW55A2>_@zg(pa5-rOE1-l|~)mQH(jpf8Rmjut9$fJjd%UYvQbOD*z?F3FVID'
    'Tn%C%;c(WhxM&D~tf**mq-w+WS6*ICB(=(g3r`(AhgRk_@V1qCyS~IdIE^;>akWssXgiu+Oq;Bt8m1_MtA$6@5pJJ`CI0H3i#&ds'
    'omOGvo0X`oGB&p<tF2~-<Xv7Zd=i|2t?()uXS$x{Rv1Zx)5c{bmxNS36kdg~xt&J)Zl^IX*BM$jD<&mGIO`+3vnFe9+SfW;87sTN'
    'HbP0OdmeS9`Dw%WQ(kXva29tt>+ki{J2-lM*6FcU`gW{momj)c*|35Q!JWOoWeL{C^!{=N0+v5wDEq{ATK=GNiS>+gW&Uh_|9SS#'
    '+o-G28aS-z`Do3KyAmh0##t^p>aDT&MN+L+?VTw`ydhypFyajZzl2c1vY3SPY_lgc0*2N~)De!x>%Yhbsx`)4B*ruUpK!8mo&7G7'
    'IPGylHwgW<HvO&Ex_LZe&|3?4WbI!IS|rm~Hmtb}*U?^L5~g<4ug5Rj!q!bAB-0_LF)j&Bi)j~~zh2#@#IL;t46S3@NcYwXeF6r='
    '-3YqH&aSe)YJCN)eZ5Tc$XX3~li+&JYbidXwCOw9qr4p)Q3_|*b1!-ehdr?U?X-`!1kc0kSr7{c+&DMc?F;rg>-jt|_hEuxi|iK<'
    '4JGURW@UJh1!aT2-F5Oi`pd#?yZ=%r9K{hTdwoP#?X9LSJ?U*>bWucJwB@~Mh|s<MvYgo481NLF3=7A5A0=OI;kDkdO7*!CDY5B3'
    'w|OBvZ?1D$^iDgPqxW4$s@Z9Nzs@0c7d#Q)Q%7%9o73abS)JZrk<)G3o!M^-c(6z$dFqQ~IP1{q*RIl?ehH&dgi71;eU!YTO`qYx'
    'T%vvU!h0%vkv4r#J4yqe6TMLl+Zpw%G4vQ`zxPO%5SCY}e!ppCx9Po*4*tx2ayC3p>7LFyOJC3y#sWneunpU-TjPv9l>u9eZ~_aO'
    '8RL|)sP8c^;UB--QNF?vTzwgbEY6ANE~mY_9nO;1d9sE+W9q&w<3Q{-{mv^k$Dz;5TKT5oCF5_H{Uy|uZP^GXaSoRa3^FPs!=+e#'
    '6m>dQTM32Bq5mai-`|?mZSAJt#Zz^R**1@mp7kYkIeb99*4=(Zxn({;7D7Rr)Cx^+zC~8R{W4Qtp_?!$sSR})=e#!a0EVSfytt<o'
    'S{4&a!J=`@k$1lJI!H&;F^LmAo^=X}HybbY%St$4jnhU}lqb)tqp~x9BedHSG(i>nh>^KqssApAP!a!)0l8q&w3zX4m@&A0J->Ik'
    'ltMTc2wpdgvhwm?NF39X*AUfZ@;VD;VCFAadx|ltcrxH}*Fjn>BQEs|&JP}L1g!Dfpv(0w?$8~%Vb-U5Z19;jTp93e-XkQVS8*Mb'
    'nZ;wys43kJ=g?<Q;ty1nf=9b~(nb3o|1z(6+2HNEdCk>zPr*@D*Pt!liz=I7l%Bxz+>55`6f|{Rs7R6TH0e=c+yskmTc&F@)wVF~'
    'X%~3F^p@1RZNcL-Xb&5=G`YVt8?7{Hw%RJ@V?`Sow=@|vwq>Iv=!9isr^L5zC(&lL5g>0zvk~Cc&U~8`%59l$Lt*MbCrygsz){f-'
    '^KFvF#X2{umZM+ErLP)rKX<B0&5Dl<>%bXf3|jhrE$<6FP3#wZ|Mc3zXMlU_LS^lKF<D4H0~jcz2$M0?e(`?kY@`-@{SF?gU*lU;'
    '=&?sq^Jedh?pHGH3lErbtUvc<opp$N*Aje^c^h~@<Bf)b7o%D6Ipg4{mc3T4lR5Lr<>TPACizdlf`_rp!t^fp-9~EJykLEnhrt|s'
    'XP%*0aeudS_PTiAO=m-U1RY;=#)3u%|Brg}OK~=e-?HF^P#~BOp-uA-@Jn$H;cb=;^Tamq_cZz{=S^?WgwyL*W|$7pj_TKSw{PP}'
    'Oskp=8e7m~t|NKKF#?3f2p$Z-3CmfU?yrp}5yJ$n*x;vwKWXrpzRMjR&-WKvBKWHg{}!-mS{uZ|-<F7SrEh(HL`EMTua76cH{cO`'
    'e#B;loDsG-Xm41kpuL-9W35A1Gzmk4R)>rW`C*&wmLKkA60PpF5`Jzy3|gV*!T*rCty}h_Kt#;};MfN1I5#YO*n9^qld^%aWmU!%'
    'I!3K|%#yvdW}aE-CU~4KORDAwT!`peYlgKx=6Ci;RHr%+*rJY&ZFQDfaZonoZI(&MxTQ54-9ooSo4UF;To{|%Vf;M|e&MtH?WlMA'
    'k|(xiBVCz0+^}VPj5Ay2XTxOr<xT0Cd0D1{i3>Vi!THDqyL?uf1C4cgp0w&vamYKOeRv<s2fI|Stu7n=>V2tRPl>KtOD<20!Q&TC'
    '-fUPbd%WNoyKMOOK6)K8`=0N$?boxsEw622-`jI9I0Fav+QNHImz#%;eQbxH-LS?mqn!FK*Io8m;yGz;hTN4pmdCQcE6z;4$9ci4'
    'mN4$sS@qkxIOl$s6|bsw-|-3m>oY0O*>l$>yrFTa?6uqQeHq9aTHH<5F|HfPUyKt5G7jVU0Uu#`(7*)BavwR@wlE@=sao)5Z6K$w'
    '`EK7lCL*imFknqSR&5#&4|p-Z-O?ck4`Lh=xOCE2kGp@}?WpPCznbYKzZc+>X&Lr}7w{Wrz9w)&pNB7l^=i|ym<j)DS<D1)53jB6'
    '?+m=8eVKum;O}bS3V6VX`*-t<ko!@NI6wGG9BhLh#O<O@=uh8khj{DIEuMSNtcmeVdXBA2(aE5I;XMNnKvVhr&vp3pF5jxQ>=1y='
    'WM!A@&YFJf$Z%F$cfZSbC!-Tdfgkja``Zj{6SxSTsJ}0FJ9QiyxJ%X<v{qN7AdL0`&jdZt(M6dy>-kBi^<qBJ_+;I>n_fN6%kZ;)'
    'rC-mtLRVM@Trm+0`Sf+uE%;uBuB9s+AQzPKoA&GG*SvT2$*1&c`q1U(BeKh(ms|%r%5@5Sjd4qX26LUlnsnny*4c9Ig6=5hZW_`~'
    '5YTDMZ7z5vJCQu;(hbkMj>47%@79IxgAVL^5`uTbXA5MBGm7X;uVI6?3I~r2Y*{{jmH}gM==Y+P`Rof?G+nn&AbBr@7JDiYa8x{6'
    'jdJK)k3l+-6nnAo<c#Q)7?Gx1mvhQ7b?hFq_9zX@{<V*3#J6~u#&}f6dy4KNS<iyLBFC7;zhm{Kb=zahcFK{1J%^XIl@BVsQip%f'
    'z85lO!u6T@l>2OGUJkD3GnxG+%BZb7Y+sk0y%+0ADmoqSQ5A1ntlyAKuRA0lDyUOOCVQtHImz*;a*w3<iFY5}P`veg8+TLuuk}G)'
    '-kArMDQ)PISo&US;4$=Km)}#3(`Q&lWCRYNeD_;DP7He;%ad*HHSmw-RiiJhxx;z7&FZz4Y==bxR-M3~I=6L2J9|9!;6@oQdkos<'
    'UPsVBsvkI3d+jKqU9mj6LymR~USkYWv-Rq8rC^NbTP@{XNB5{!!?YQ9pZf34Ms4XnKQZ5mX?1P4ZYt_`kF(tRD@Sq-c!pj}ea$G('
    '_d0YN_PR~2p>+XRg!dLvUM24ex=u3B9=c<R-Ma&)b_KEMt%d{GdUVV1v1sqjdyQFn&`apLoA*fce{Z3$tC$iSmY&<X1N`OorHt{&'
    'pg7(7-)<TnYhwm@^oKq0m1H<kN2GbTt%C1<_xL3%W|0ogH-FJ(chGy_UfuofyD42-IQJP>+csj?nU8P)-1ko7^qFy;9C*ImdS&U2'
    'dJnpqIw{_}b<a0pj(^4bo}cB+EA9*9jQ3jSFxpSnRR`_Ln)uA8r#@S#^UPiEd+$EmGEjY%YdkgHeH;azZ+DOTB0Sms-btU*9r3oy'
    '05%*myS!Dpbtupn-O-G@T$Xp6(XZMGK81A%ekXMslEXu_dz><E?bhjZ8MK~#xfp}*;KhD*e>?lLFV2$Rg+#gE0&MhRoHEGmJ$MA7'
    'Y6Kl6TsHzw>RpR+@kf1}eVli>_?>QDtl#mdQ)tugCb-(pYS1dTqi0z*i9c^#I6P`=!%4A=#P62*`(B&g#y9miTO!vzb@X=38a55n'
    'jxy&?P4A=F(H^uYILbPL{i%%WeI})!>edT%=`Z^-EhpJaUx7dMj)YF$j$Lu~hFrFcPsE3g%k#2Xy=ef6K8NWaR0UyLhFEi07KiA6'
    'Sk{EtJ=(`?;D#q|WVT_MI6Ym5e;<~|V$U+J7QB43W7OTRaV$pp*OT`jmw79xuZCrMh%AR`ttLOa5ppw$if~qgcWZX263l$98y<|='
    '-%8bw&1uL|h{1roWytt69ofrB5jR&Byr!3;j^#yC592<7{73CrUL-ndSoYpTBMsApP4;u+vOJCYreiVF9x3$ta)Pj29ew-=xuu@f'
    'Q(vB(XI=%@zL102$TsAJ)CoBuj)0aE+Vw1KDTbXZaIzL`{#S<cH;hw;<y1HTUgkGNE|1IntXK;Sb0in#JlM%}QEu7k?CX46B|92N'
    'CZE*4<*tKoi**)5`(<52!<mFLTpM$@enDPR-MmuO^FJ)ltc7t}wx4OoGJLH`U)HmTJnbR3hyJmA64#U8z%ic7d{eg0tDW*98Gi6+'
    '1$!aii(*-{b*U@eI;o=T9lX-M?zfJt$nP%X<eR+^|7acYKriX~=J&0SM%H(9spjQFH#(m%QC{+V!qvJ<bddpPWU$s_;*@2*r*+38'
    '^uq1vBT7RyYdb@9)+=lVvE{RVvm!U}*b032aIo}x-TGq7a=ktoTMg@@iD~P=0wrT>V%@G-g0QVA@XldRX`6(wolsw<D@wBbGi)cY'
    'Pq6*Qz<$#97}!Z#uZ@hzHfrTU7Mk*H?_pY^T#Ny_y<FHoTmjo|<k)D*wIS^KFb8Wv2N#XQ@$oxy^Pq_fb|l7w$i8dvDq5FnyezW4'
    '1)VDLwgnxprd!ay;*&Alp<H!>7t%SaJ@|9RmJZKs{|~)A=_xnt@8Forv5_g|mX8+84sF+fjGur*f+YuyeMTmF>XhfV(avc%VL$10'
    'PR!fq2YGC@KH75H%X{6{ZHFV8E##}-chl7cy;Td_m1zIAuTAOR%WRt)pGD-$0ZXUVZG&RF-8h$$+4VTD{d(HO6lCjhj&(ll2#+lI'
    ')QNs?n<MOlT#s`aYiPM?j9qudX%h&nf7TY6Q`jXTyK1&0NbDuJs-S-}4^cu^?Y@vD+x{gXHw~L3<j12t-i__K_mW%M9;q;=h+>be'
    'vuzAyE_v*mkwu%fFX(G}*xMxP_<Y!QD9#P`%>6!3VMkPuwc19gtXua{L@zZ59=XSM@2_Nbac<-j-prd7WaxJ}_rl}Wn$^Hl7iZ3P'
    'MHySJdE>0ulfpiPO?EB!K8QyR+ux#CoTvU7&#`H?{=bGD!FA+eVGEFL3ox$Rrn<MMe*1iT+8XF{XF6{vf$}IHcE-g909m5-`__-!'
    'ZUMb{b~*Ai>+}aULSYXO?ce<&*Yns1HWMDB%Ztm$^SvC{SVZ5*@2qpxSfB8^-&&leiBu$*zUsTa-q|j^TLbZPrU%0=06I%95D&r)'
    'VzW?h*!BXpPdVQnn^g8c>@RIQctrq@>zw}<=gm4t=^%#X^XN?+0OoGzwsl2zAGYK;N?8Yh9>}&w=y8wy9s9XyJ>vQH9wYk0`L6#6'
    'eu9nL(UD^77(q!~|9p?_R_NCnx;E3>4Semk&}la7f=~VPedSJCE{`4CfqTC7DEJZh4JPwd_(R;^-fi*5h8`F^(Zb#}KfeKQMq4;+'
    'N1gcgw#jEMmNUzP_~`hs)jZ`|OW*0xH=Tez^=Z#9Ie)9c7msH<(f;WVdq?Anj-7JIfTWv0G(gVp%A@}MJrVxh4|4YdgO4mP$$sN4'
    '1Bm*!Kfpw{SPsvAbbPj7H7|Kt|7iZp_NdpbH}QKD?bwE4p_>c;0oEkV$Gg7H*d_NtSs%9Ot%fW~d*!4;y+McM+b$hL=Xc-UZRrg^'
    '-+-eJtoep)iMg9E$Ih{aTs*$L&j9)$%Z?OtH0}_x%A@OHe}gwa^^fm?98?4U4C_nay@?$dxQ<vMfv-mHw;lc{_>r(z0~w|HY7#Dl'
    '&SvN>=;o2`zE0m4_$JwhAopYY4T}_bh1iKfFT>`9z54y_SqGUGHK&)^FE#NX?H^(9*K+(CTI3cx_3yFhJX*F6yV-IJ|1_0%_pkeq'
    '!yiL;%sFfZwn~=8-|gJ%TgevVb<tiQ*g0lmt&{W3ysLlD3p;!3<`f<zXnrE{g>3_@UC<bJ{o9`<zw==cj!gEoz#cU5ZVbtBT5{5_'
    'Q~&7?qcJ<lm&ArGISZS_r+2yN8a$%~Yy2td&l{GCyyeS%N1LY)ZFM6S<c5hC>ot}KU9%2l$c+}GErM^Ud_}px4KG~Xghmj)fmW7)'
    '-U9=Ugh!vQhcl`Sy*${2JwJ<9Fe)W!$3Q=BP!`LL0F*;Y*Cw*^SvNk#UYZc}0~SzD;z}-~RUvbR1iUxsy0^cY^w;*&DDXPY?N$8F'
    '*p3~avf%abV+*>mavn;Q!{51FxYcsqA{2es$7dtj!I!TG&&xYz`)6oZ)5nIxw%PP;zyuU=oqXLqY<tJsHv4j{+opPitu=lcVb^WD'
    '0)LBth3Q_IM1&tmX{?Fo0^W?Za+@vseZprw@AXADA8RTa!8Qj4{}uaQn1>&c&uroUVSD$4T~vDd33{yXmB)tOzBmD^9rHSF-*(mZ'
    '|0vPsb?hff%{R`od?EY+D)#*W_jGV%srZOEC>iG<`~hrdf5m<ycW94yb3bb#w&&P!L;Fnk&AQtDtMG=+EA~fyTQ<WQ+g_|x>=dqm'
    '@k43PwU0A`%;TJwdi<Vz9!H#O_#5+UZl{LbeZTj>W9ZLuo;vu~gipY_eWPr{9^Yca9(#;Uy?ta#ClHU%SBD<;`XV2izC+-;Q^&pn'
    'K9AUIn1?;<dGn9>yN2&Z#`ebi9r7991$d_LbHNrd{9oX+Tlj{+Z*|v6x2;zpfKc$awwp3vnc>@_4z^6?o&QVl$)=0a4cG8V(cQQ1'
    'uMjfmz}QeN^LEfBtKIx%$Rdyvm^KSIWLfy{*x$i6TILtwolSFkO^)0I`I6^l{xW!H+rq8e=jFgpdcGiA3jZK%GF$M`y6Y``TlBu!'
    'j>~OYFBa{r`jq2XMo`}Ei*zmsxq;V>W6xgSd%v)qsy80!;fEty;=st}e3r|2;$mz%!%rH&njStj$Q@GPC2S2N6XUqc<pXf!HrRKD'
    '+@^B=EJvx94+aigcVE?ZV7`mUcH&;LM&bXH$MLZ%*u(_hKl6R^e);&0G-N|F>~^6^ZI_zuUt+sp?6wX(4_<1>kHSV(WYe()%h<!_'
    'bZmrM1f$IDzNe6E_qY9N)0&N=g2$yic-eW|$s$9sPAuT#J6UfVw2b{)*e1t(56vc)6o&N0u{B{x1?$ch^C=P^AG%be`6<Uv%t;Qr'
    '^azeAqnM&9H$Nb=FpQ3AUv;e0lxtQi)_aOYZcf<P2kl!Yh|Hu|zjnrX``hY17t2V&ktO_O^-h=2qw74>EA)KoBkTb59u?EvB1>Bd'
    'o0yWGu-5X7LpaOXDTzx*ZnyBaZD9){wdeiqT<>7A!dZW>dsJ-uGMK;h!4he-V!b1_fW>-8scmuQd~y8UdoD5eLz9AIviXr>dKKOP'
    'pH1RpZy4Ufe$uD@`JVUIscnZ)ZA+ZFfqz+cGG6Ei!u)SIKludS|6E*7-bwpVW8)os0{(Z|L1&>&vUw!r1u^b3hMSge_^q3MHypfx'
    'vRCpx4}8zkFkdjp-QYT!H^IO+ydv)<<>Sd;jQZp$^ajM7IS)<+eUIU${e|OfOEs|3PCbI^(#LQ)JPG{8wC}xx>aJ#DpLVXhZ?XWy'
    '+>i^V;rh|nq4M=)lU{8sMo8=x1wsl20Uk>{i|XDS#Q7{*5#7OnHYmJAu=*hiH<%AJS=w=dFh}YUK1aNfO4s2-PGiwFf)Pp9_W4TS'
    '2-pUWhoZ?0UQSX24IIJA9W;1zPliZXUQG<C8hJPGcAo+=GEAl;AWLDiD%+V9hON!Xp{9-dn=?L9t~@jGEm;dw)K{t-3t1#%%b_?I'
    'laRtOhKy7zoC*|?CE6sEa5z_p)<~;I6l8sC<vJ<N?EHzWE3N2D;c{)9de7_3lVlWj>S_u}lan6^>DscLx(Jh)ZBz)KlI=8ee|y>z'
    '!pot6COeWHbFlf&S48|y%Cr;4$rU&q$A=tB?M_Cl8DqNccAOc>&KJ5g+0JEmzxSM!U5Lyd30nGwop{poppJ!K@(0hM;3pk;vT=iw'
    'h;7)$%}NNK%6M0917*$Uz@%WZQH6McY=jVPj_dJmY|nro<aH`12uMz0?9kQjx{cE#Cpi8!*~ko?Y|bI5EA1Vu#8cj{)BY_SBV$xJ'
    'Qz+wf7-ULTpDjSZ*drSu<5!t2)Ic{7W4hb9_nzElVO*|EQ?cKYk>_xB>ZB~Q!MBvwISe??w{A;Ol9ZX3L>UUO+nn1%d2X?cBT)#E'
    'wO)pE<dCw0Fs6tz9rYMW+Tn~95J;IuR7#=?LQ~>$kHJC3qro5wzx~X%2acS|1fgWLq|eZuuDT`4`B0Uvwv+vY%(dkP$-5$>p6aAR'
    'N(zBTrZPdOv=Aamp*j%<kn^QRLKcQjJ|V^DCN$SecRF1X1Pq4WoU(<vjk^L7n+9o*ooPlHZjo+x-CL|DUZ4Xix<W$u*BW7(6uWe;'
    '$U9N1GsP$Zb9Dk61xl{O;S$ourS^q@ytO*Nj4~i&KZz+~MO_U=RVxdn2drZ7*@@p;o0mjmXwHZ+&S#<{PpRT469KEseV~$qzRRe9'
    '5JE2wK^&iv!MS+@=(yY<t1i@Cog+y;f>psboWY|O)<M=p6$+>etyYM?F1O2CmJ@%dbQ)~;XG1NJ=SsfYl}ML;k!qQMWs;s(ta7`I'
    'Mk<71fTVv>DhqOPzYtZX7AGoEfZakSn2X_%Y=uD4m>eTrt;A78Fk7=*Pz2ZP7J7by$$5@X-F7m!RA{L9ESLMUi(^l4SJKKjD(j-O'
    'C5Qff!T5)Gz;TBVsFFgZCJ!ei2P7A$D7|sXuDc7}i%g21sLT2iZI|R`^2x0bTM-f4OXA?-1{I5z1T<Xq0JXl!(@YVpR#|0efm|~L'
    'vmMTAGfo!{bI#lnk;exh({xrIMerr1lyOvYCZReVUHJupKc4R|qmU;{=oIp}VxBe%QI}bVLg;f{$U(+v5>$;s3(i4A1a5GY3|%X@'
    'b&!b*`z=W2>XbT?;t|<gbuR}rk9N^r46(b#Ig6CilDN5Ufk*=zEQ-4={o0VpdvR`{N;K|5-L7EW%uRM_1z(^cs)PX50#!kHXGj)O'
    '2s|;ALaPvLKS1HLA>)J?jxbfhKx>HfAu6v*hXekquG-~9H3t!|fT~cb%XrxUS$7sJl7<`!3JyY}5e<B}T?GqgH(jBv(=p1U(TFR~'
    ')8W!`j|}d8NT+s#iq6fGB+(vWyfp?b33pS-yNrGhowzxpIs7=KOHsmEGs#xUaASK3%3$9_Nft7<3`vFyQpF)hjp8g<673YHxuP-R'
    ';td&opOFDd2!Y<Bz|Rkvrk+wA0=%tIdK8q^QOag9vIFL`d~RX5g(*-SOVWppo51UAfNaJKj$}h7F$LS#A*<<v+tv_wTp+v&Dn@jQ'
    '7j!Ea)d|!=B%hdsO=VsEH0GRKowMW3n$rQ&W9ZA9MWFfyxByw-r5n)PLU&wx^~PuA?#+#5664h~V~B3gVzF7BQ=KX0ctV$S?ht6('
    '6)u;m0?1LC^SgAN>m*%w!HI83?ocKd-67<ersK+cI9(gSJr-YIurV0YmK8)ZXt~PF&mH2xh{ufsBNjIfj5ygcx$_b>6lYEw90D2}'
    'oOKpenMtCz!O`H7NqteUdme)$=hEkiB8?qReM58;fl^LgpUK5|S=83<@u~%4IC~bjP<qV*T$&rHyhPcYxv2%j-#%w}X%7i-%J8CQ'
    '!0Yb9V-VgMUI(o@3JR~;@Jf<Apn28R?(;mvmDVZ2Uc`+#r(FjrB=3;wa$uz9i55h)kzycrb9ffibz_aj`k6f8lYAMRXpdu0K(Nus'
    '|Co(}b_>lIalh3C$?1@vZb8vC#Ij*Ds@sxP!52~$j5PpbZbP8Ag~mMjDbQzK;PZeZYxYU)zS4|=kpVVO=pr{IGuE4hc-s@g8`;Pa'
    'rGU|(KLe)*s2L;u9#WbW0(uz*wbIU-nNg6?w=^FDt#9*fQT__E4-%Q33S`aXjg_*bnCtLozPt(w4_!UT64AU+eCBmfyVbaFw-a?0'
    '*z+z`h3PxLH-HH2L%hcZ+gT_q)`k*p9Z;CI#?4Y=ychBSDrhctsYb>u6wd`A_XhPcR0b)D6KcB`gh_>9^FwMrM{ggx?#S?8&=F&X'
    'x4}JBhjbsJt5G%xTn}^&8`PO3`r;6^jWzHKy5xeKGfFu|F6CY*t`$*FH>h{BKx>FlDVgDj+Adf&jHRJVkfYM2(n9I(82We0ilX2-'
    'IOS`A#%P1WzsM7(EH_Hl(U`JkD9RzBy6xf6=2O1sD1SCc$F<#{P{kC#Nt88}>a#=_d$|f^^OPK=z~oOErxXN*Qxs`|%A6tui?TPX'
    'bdMJ=%t@L8DBUTlvLK3>B0EcBye<{mo+4Qb^yCzs>L~Y9LJ;E1ZBQo)VapV)nIXTXIP;9Z^c1U+dHq?3%zH6KRc3<cO>q*Lgk%A>'
    '9Fxz}t0OrDnarZJQJ3VVA{qi5fXc58{V+uvXX1-Zq!TdEQ%=emR(T?iAUmB)u~nNgeaX1BO)(%Du6Zh0L`JzbWjdKfnW>B28!mb5'
    'MMhi4sz(LU+!<T3DfS^Z@1}B+xBUDpHb6ORnT1w8h9kp`Oaz1}W84g$oDwr<1PN0-O~yQUib2YFeoYBeGX^YET+t1xR8QT^<jNIl'
    'n+m#@J15HE1eBysul5Ed%;P1SGEmCoM4i%|W^~O{E>)ROi&LjKHa2VLQQpUA%it79T_xQf?m|0HK)p%Uc=XnjT=MeY*q}BRD@9lE'
    'iBf!;JUt;D0v|bPA@kxoDL7=-Wqz4r&oR8;pdM`UJx<9GGSN$?i~%xE6qD@k_;gHZc(eR5HmGSvDIg1_?ala3O}zveiR+Z0x8ON4'
    'b%JiNKb+xVLd9b)J0(3TMA?}<Gst)+Gy#0p)H8@L!3K4Yu?mX{Yzq&d>ylImcabT|px{k2A+#_co4g|aOf*2jO@7Kytia$+*?>d~'
    'G69G|YX1vv_jAy3p_I9Cl)>xR;7*y{p3hdk+XZaTx|~#51eA*#AdOtXns|zUFF5&3k-i0+xGCDWU=}<@F&C_YC&=7uCf2@S^SpGW'
    '-R{z$3K7_r+Hoq4rS)_sjdd~ki!Rs;HzqR4`zG7Af*76(aiXNbhyx>DG)|PL_CgG~xqKcx$a2n(#+#A9Pca6W_*PR#EHB;`S=ePF'
    '8%?nmd3halH;H95#j<2Vb4>+}$V6ZaFu8(0<E$&<Q6{v+)VrDCe5QDhOt6g@<TYYZGn~&<_nC-a6PfYy21x9Rzs~|s!rCt_Vuov8'
    ';8b{=0E;W=IwdOPGqMGGMX#_fZegL6)$taxBtm!#WDdKt61Q82Yb7RBC{6U1?@|WoC#eAzZGc|sl~`1lc3}-sK}#yS8(eX_S7uT5'
    'zOBqO2jJCXhnB=k8D4M6Qa9Sov@7Ci74}W2Z5bQs5G+aOd<a%yQS}Zj`OaiwWUK-{TY!@zM+>Vx>#8x`p;cL5ot-6*^bBXYDo0L;'
    'e~IL~K@L|3tY^m6!!fu5;jPBxnX#zKjCWU5;AS<JBpGaRxRz+xOmwX!VmcH2WQiy@+FtT@^4q<{Fy+Z`gwHaflhvUV^ad>Pn;G7J'
    '^_!Gf;(1LIEa5&=3adj1R6c_cNYvGMF;-YIsmRzJEpe7nDnr1GWzrG{l`&>m;x{ua;u6Ztm^Cb+Oh0OC@Bng3P;0>qF1gcX_^dT}'
    '0EIFaso0V6FZmwi!TFWcJ!7J`glRLL0!wjdGY&CsjK_VUAtN-F2MH6j#1dvWlqIyCv9Vdh^aFMEl3|6jB%{f2UrS<@jQ!LCRS^1r'
    'iEPfK&|T<{FpOKEAY$V#9gytMWS3;C**QxuZn8s|Uc#{1;e)QYwbDXe@{;wf+%PvE)%TUS`l!UO)XwSNFb58e(j_W6I|Sk-!Km+$'
    'a#W`{wHp?dXiesoIiYB<F1a~pyFy=u3yKEnDzuNb?UJQ@cF4r5g7`4axGD$;Z7Ohah?}eWC1YN2uZVAz#agh(CJUNk)p)?HPa?sX'
    'wS4kl4bDLs#<SQA%HhHsqC|1qALL;G79)XamNd=}e8~jsT`ZX9B<XCh?b@&`70pR)7+#BYPLji53nf@wCrQFKtmK0M!+KRTe+h=G'
    'qRmZ0&BaMX61<aL2D8ZGQZbW5DO()=B1^7b>@tw$HC=2Phw{$saLST%7l)h3O7dxGKth3890noF5~0|IBa6`~Hl59)ic6pG>SGF#'
    'Wu9l*$cB-$xL?t##U`Rz{9AGOtgLM)Hl55FaSUfK%N4IUq<GFqcUbc*_lsha%A85xi0MuAwW3aPl1A!8nnsONG83CjdCWE`%^4Aw'
    '*Gf6XE%1#tZ917r&*+@MS!V9V;iGdVT_jr2oY%Acf&@6$K~fHfjLw;Uqpw>qruxXsXunu0&6o>PT(6ZZg)CzgEa{`iBb}7Rr7E8x'
    '8CAK2mE47_BUK%SH_I)jUYObBN@COMa_R9d%#N~5?W)5!<x0X)wBOk(3-4EyNBbx^Tuz!v&TXzvK53Y=u-uD`lEWr8S>7czEw>|O'
    'jm?j8CH-J=xzeQCl_ASRwjw1^eJSutjPr5=MqrXQx{fqNrMoXBH>^I}GB>h*Q82nu+DBQ_TEJ2x`wXD_v3X>cZm&8tVwTI+T`tvD'
    '+#|WM-|$|7+Vi;A#OCW+>`}FwL-ghGh|H)qS&#a$7urX=QD+e-)u{_y?htlfe}@^#Vzg@99hp?u5ql)%tHXe2X+^8w6wX_X^C{Q4'
    'Y7^?LJbv}ui$Q&#f7+b$E*~DJT%Ky&--1rPc^wk!bC+u`Z0?@XwmHvrmM*lSG7+ryYdiO-I$c558&vN*xE|GRa#`rRI^;WdvmV@{'
    'w7A1(z~6S)>8y|Mb?adfAXw$u^|e+`FOf-4>>S=XV-?`D=ke#5ugYV6q{$qCH#%_OK#U6lcWHeBH|qR2#C=xyxO(uI?t=5jgO}>T'
    'U5)bogagm$zRVcE&g!f;?j;^`;NEI?D$jjJzp}WPfqSKyM}xB($KeT@7M1H<H4h>7z4Fd4A1?@Iz3dg;$KsNi>MT@%u;pq@=yw^f'
    'Rvamp5NJgWvOMMrU*Ks?_yWzUm|Ng5UoaW!?K11J4k+mz(X2P9O9Rqf=*p*=qU#*{Ny^)4&P`c&Nkl&vg9MxmvRGg-b<+TK`0;6)'
    'GN!1QBTrBdo{m!iY%3!2$%R93J{5YbB5$7svk$HBlp!`zGbGND*$2$GIH6kF5&hU^z!fkFn2IA>@xPkfq0Ar)nRa1NxqV>T$st9z'
    'KiO~8<};%ehaFBN-nH~DV_zUC@gi+GdUXY|s3F3=FkwIltd5C|Hb^*&d9U)z2_0L`ENqN^R-8nbM#%j2fGVxl31VbYu22LLBt)uO'
    'Y+g`VD=V=rjgHY#*dTGZiM=wo%eW1XOy+bHFC<sCQa`dQcj}V1TjMBC?YvA#NW(2vKRKD}(fWt_{z}eJe)RkADj?a`+fCB*4U!EP'
    '#Xuo-uSZ7SZjTo;24wkX7Lf>TV$4H6KG6zQFS)=h^BNo7s8rWJKvK2oupSE@;gm4CiN=o67SolCF)R=adJ$~6FrL%Z#L%>!&!QFE'
    'l%Qh0M&yzYTNQ$HM}A;5FQ;yN>U`EV?<gcF9@unXvSITMSBS;1ZHdjrsDy6wa+YJ8uT-bNe%6hyWKUU|HT;r(As{}bHbX#{g!ZhV'
    't^$+*-fE@OLNtgL4|}T6`-Cn%-P)k2;QApQuIE-CFkiRKSh6PLkW=^mbUp>k>j<HR4Krc7?g~*KA&*7mXaewanHOSji_-x?6%r|C'
    'Bb)P?TUiNdR0G8#w17KCoQ=)NN{CYsIVAW`E#CcEv?9f*#nelo!!nQC!crA?hmaox_wTy#sTkx8F|rg%;((Nb&83yK;$dNydSny+'
    '><mD2Mftm=M>|pgF~*43*FaH$*hqYM#=~o2GYiLZw-60B_-n?}V0n>eSgykWDgq_IQ}>L8Eelb|Eh7-!le>E@$SJPVD-ksA&%$mq'
    'OE?_dW4<_h*wkhTVT+x+?>g~KLX7J`m`TFKCoIktsw7FWSRB~$u7AFlj=$K^gM^c0NQA3J!Iup8M{y9&yZ$|9apT2QnxwurEN!_R'
    '8M3T4V=hV87mvet-`;Z(M^!J6xwYpvGo5b|%0$fk)K)(0a)0}qdD)R9329p0sX4Y#a%x^0Z|#qF_0POa83#oLq0xggyn^six#)|w'
    '4wqP+G$`tr`#SBMb_)bfGiMBaVrq`}Jg7NS>%mE+^n~NvdyIN>enT-{<w+R^RVl9UaIbI>hobJ!a_G9?4xprh^3+2XoCQ?wp+fZn'
    'o;opxgZazTun)UEV}5m<n!)i7^N)%F5mq7<J09E3NiU07>aK9mJ!Tv>Ktrxz3pE)AO%oSUAy6hlYhsq7pUoilsVF`f)%}zpDOaof'
    'qVD-rcf6uTLko#gf6Amj^CsLCPJL;|;hU}Ay{w5u0kp2iv%;yr*CS&dFlEP>iIh4OK<SL9F&>o9QTIFq33P&r6>%!+i>uxmzEk0J'
    'CFymw?pgwJY=o!WXfi^kyF%2Ti2s<cY{D$<o(hv_Vc}gN>RZEPy@)|{C^7~Ll@9tLD}|!&-Bk&R<y72p%lV@e9g%vCRwlyzDD`L2'
    'io8GlWX*OzWV4~=LBlP=GQbep1IB0fg_<pl!!6#a_6kS5Q|%S*OFwcT2G_s8*O3o)7kbZ`B(i*4s=cl6I?qciHZjCtu&Tk%iWzX`'
    'v}mi($GiH^G3)Zya<i^8*xm)Q2o_$Kw6({(`iBnfP@EWl>srfM>wp>RRBUxvd`I0o&-1UpeE#yMj~|o^{I9?L@bcm7ixN~jnfYJ8'
    'efr^_KmPF_;>Yimss8oqwy|jKb(6As>LTrOCia`7%ir)4EON?TjhXlX&Up5$?|Bq_@ErlA;G0jFnwNqPXyZ{S5LKtVyUqeuAI!19'
    ';W^4lao;3=wh;A{>mZW?a>Ey}*4E}vH%$AwLj^zI2GYPmd&5Fo1pIp;WX}tZ-ujll27OiPOzMm3IP)^xJQuZzDQm+xs2~_=!kRbC'
    '+&NXGvwSJA#P%{QqMcqE*7Sw^nhk#Kxt?dU|K`J&FCV{t{`lp?cYoNx+ow+-|LNP_`VZgz`2YU<|NQ*NUw?V~^xdC*`r+e;Pv1XA'
    'dM)DU#l5_4o=hf;&UlTVm-wm2$e3~|2qsoOGDM6r1<b^nM|UY81(Ni)uIE;sdA>h>`ugGJKmGd4kH7w}PapPaYp<Kja3H5w(Q68D'
    '<+EnjY@DW>y)g~iM@LKC`xA`1C%6@yk8^O@s>)$5;em<P#VZu0u6xG#XSx=D`SI%?-#&i%q%`Dtz81M$?rOQd>bS{cBGOrB9^32~'
    'L?(f=gK}F@gzWcywbRYN`}C)8AAkJuO^L_L>*fQ<mp)%5NL4hiNIZ_So>PGrl78caA}TLoVbmp~OW2;?b$^?P{&#Obe*P9I)!Td>'
    'vLe;NO0Nn_E@$dkZ%m9E{GD_BHMlyGSG<r~DsDBAOkV7d-Sk|(IQPU2yIeIHV}qt}4#%c&gD=h>)Zj{;Co#UocBaA3HhKG~zRUal'
    'EkaI0sJu9*WW#ZRq!};v5Op33d&oLRq$9{Uhhvmu3qTmx7w1m0_iMu(8JF-E$N6lM3%)oX6Yl3XtUK#mV~%L!d}NL=)0XEo`1S(k'
    'rehV*puR|u@<Ix)7sn|*>woql@c;E(uZ}H8tnczSg?5M6o&GB0V2AKPh(4D0gB^`;$xe@Q!k)Iy;ny-gP}r_Cg>l$0dw?H`^F82h'
    '`r@1r2hQFPx5<8pG;_WRd*3?$ocHYXi(k|g=Z3z#jtEokxyM>0A77k-h~s<U)$VcN=IFEFyb$*4eSYh<{F1c{#`Zb)IpIZiq@ib>'
    'vlq$Zyq@C^n6cnSM*e8hw7shzeL_bQV)(`Rq0jjm>)rYC9?ZwP`oF!`|KaV||NGP1Ctd5AUZq=+E9<LbamWkZmwFO(XxZFxw|s+g'
    'b;Za@V^8>hK>4N>Mtc4%=y}=pB+l264OzlgVn~~vn897)4DsK8`0~@YAHX@izA9#vY;W`HCgT)B<6yT+Dbh0u77MQHT>)PS4f?F`'
    '+o}KE#}8lrwEz3=g^Qj#6vIZNn@o^P!rhT(3CYzZi@QOgQ9R~^rW07~t{dAef<Iwy-{s1Of?1_aV0}C*ocdXR4u09wNpP5C!boIX'
    'yDR*5PJi?5i>4v!OgkG>+As7_urPVyzD@-~I8z!4*IvYA2#B}@L(xm^tU9pyu`}7Jd)U7@?@Bn@;!THaiR&|jzz#Uyd#w&Rz2!75'
    '&#k9d3>WOpb}F3N{qFOpuRm|1<n60=D8KIZ#000lUm{6X&JGFRBy?^8;z231-~>TPy#Itm=B|7C_|G4{{QTixcKiADl9iTpC9he&'
    '(etz~lwXMujqU<t4b(jW@#}cp^={(%+lMdv?Tpryr>)R>?F6tot&8$ULI*#9VVG<gDtP$(`hLSjk^SyX^hJG@M?Qi681u8$O2j_l'
    'USc*u@yUD4^s^3QlScSOLiz@sL(G*bd`~8Goxi)4)B9h2{+G9JKYe)l+fN^vQ&Pxcg#&!8GA`L1<DvCO#8Dus2{zTlz2UVX(=+dj'
    '6sHXrZlW*q){{F>hwep#@f#=}Iv_<$#3@j$U%cM=is*68@!Ev*h;AaPk9qMrOTXjj6~wibyyJyvG(^rL4?MB;9Kk|TJ>p}LvqqF2'
    'Da>CS(?axxBn^Z%@H!z%BSH<f^u+B!Bt<k{JV_lx0WKt+C@4~zw!*F}QV(A!ib6muoo}fJ_h3jTM)4|?)Z>MyMYz67flFfdC|k6m'
    'e)R@Bd!5cLBQO`CNGs+=h;=Q|fwz_R9JiW&YeZfucMK*qYW{}2REBbkD4oP4g)4I8Y|7S&k1aWOtJ9RugP031WW@$zor%d*i4H0;'
    '3URi{Ur4kr%(p5z*+n&xITGuxN=|iYjw)Hs#X}L(3yUz$2D&@qqL2%(5qWr-VvGEWjfl#;wZtPbPj#~Gk;@xZ0};Z=L`eKhOyG$c'
    '2i@;@ukdDM8iZ5qCnqBj#c+ae#CC!vAPb}^8se?@Xql`m&GDI_$ZAAgY>uU;w=k86oJ2e@jDGwDoGW=Vp;Cxy#@vYA<}vG6C^)U!'
    'fDUXkvc5A9Jh+HjizT1qp_z{A2O`fm>nA*dYo_yfLen}kC^z)vog?BOQA3FpXo!fRS_9(DxlEDBDAN@(ZO&;&a7#W#9#}G|O3(H}'
    'd>rV1Za0!M;#;99sE*Ed&MtBD9Nl-E)6Oi3C~i3oG2$F&bH)l0vkdo9qR?XQK~72Z%S3)+FNwk4;RK1IFRCE3s|Cr&h}|dN6LGV^'
    'op^c?zn-&O#)-UDd~$NE0a+ij&pj{5bs%EXMogtgb5<ixu|>(;TEYzI>PyP_Am;B&=H1wDD6x*fN<uMuWrcqR*j+#Ew^fLyN=zwa'
    'C2zQgV>v$v(VPQgf{6)1rt(eSsJ^fsBXj7O>L936OOY~z(DPET5u09I218EgbI}vZ<^ds3SRmy^i7=ynEOvQuh#x;UQnu>>(qJDj'
    'Lp`1*epl#B`T|f38(m5gOgd(DAqlZ|2}$v;2}h$x=(0DIO3o@@k(oh1I>d*5g_bA*CEDziRYBT&ML0w;P5eEV>C9a|o3x_6*8usJ'
    '*mLChAd(#16sHA$10u8Gk07$RnJ?0<B-00PDI#4WC0@pKg3`fOSnGo;kOCWm-_YwpjzjJ{Tn%^{2YGNZW@o$w<Vz=G1zZ-RA`c%?'
    'tq63Z$Fq@NkD2r8A9})sB+=wh@DPKFcakV^glHgMC2uTQg^MW~onc7{&bMxfSc0bPG@pR3$g@F~Sgd7Ey$}%*+1w_n<fq42c11=+'
    'QWPRnA|P?Ml`J3J=@(Ao#k@``?tCX&PlQ^}=X-)ulS)0(Unn@dlZ-sb4HT_7DBv^n`#j0lNjvFOTJ<EVCY-y&n2r-nnKUa-3w{Sx'
    'c6#vW_dp(f(DbT`Xum{aOd>Da0U&;gIw8ePJu-Td&jKSRZIQExLTea`da+Ns!0<(2WqjMAX{02$e~lh4@{mFkjmVrzY|Cl-hfLPc'
    'PV`!X5<-GPoJPKI#%swpi0h7X2pAj(3-T18JqGt8m2u8)B&#uk7(4krwJAgvC%ZQJ32;vF_vKK^u-N{Bvl!n7dkDsF#Edy{n8{#G'
    'Cb08sND9a<jC8w{h{tJ)1xm!REP12#5{yG@G}C2Qyc4IDXp_kEgoEKNgZFk;U*sb*-Z!F^V(HNDnDRt*hF0ms97P7UZkmP|mq@ju'
    'k0Y-+v>n;Jo8>`_v&Sfi&e`*s592}ROFTem#qLO)?Qyqvq$wk)Fyg5PG!8{bPG(Ey#-SJ2p({;YrTn;Ci4hCG;7Z>)K}f$^iztue'
    '7)Nc}p=B=O(}*5H6AC@TD4<a=HPXDN<KwKzS3IDh@F6#Z0Z)l&v3=bAjuTOm{O72o&-l)0wc12{(J99am64I}joriXy%QVwVx|4X'
    '*&pH!U$N)-ap*8Lv}f9opfRDti2ZY&>_cuH$QjwS%g~st-XMEBF^{g(hiOrbg4nx+K_ap<p_<Bx(@T~mqV!6Q<Cwl@q(>Byg7RYp'
    '9VjHth}*0#+@=UTw{*U^4b~!O#c*V4hk}XO$}-k;^dL4fws`TyQHtlt=Q7BU4Mcq4E1&OeC+B)+x5Ptnj)>EbG!-a0-c!W>SKu#F'
    'J1DwOaQ9m1X3d2#YsH+Z<lq3wg-s3svS=e2AXZ@w;a+4z(lKplu?JinqC9xPF=8$3a8FKhLPw;NAzg>dxedTs7<BnQW+d$+vMK12'
    'JBJe|3GT5m7G#-*YH%j9l9qlgNLTV`VKCQB&K{Zt^^G$Zl*6}FOUp~F26x-5OB7<Cd$0I;_eJU0ELqf<L(f6GITO&K`=Z`0v-!85'
    'K7M)o`tx^hU*7)k;pgwZefs$G_Y#`4iV0&buT_~ZTWE%eqMpt1;ics0zDwp{)7)aNju0%H0k}JPYQb4c(}md>@rH9Xc2XG}vkaWz'
    's2WxRg`Dqd&k>4cj^KRq>J6MLe<S!PEpsNqxL8#j8JdzS;@&`1#q8M0vP_6*%zFgRIiP9sPTS5Q-4Ift&RDADKDaZW2RIO884i`{'
    'u_*3nR=}l(doz+9oc-YG@Kqw7uq;3_T%{hh+`#j4IWsSgllh34#hEvzW3hD+>sY_`Wxg!86W?Atv-im{7j=vSDj_X6ki6D&dffFm'
    '46R^-l3|cm%wMh#*)k;g_nDTBdj3j_=(-uRK<!(`qy(OdRV(C)aJ}e@{S;mcNKt4x3qX929BSTUiZA@VWI~x(u$9qp&sf`*9`~>3'
    'nE&lRetG*S#cav1n{S%PFkh8tLasTS>>{)7Dx8Z;QN_<%M)_4ioe8p1I}S=RM<x75$3EM!RHnTwiY~2gjDv|(VxQ>zRmU13Y+y<@'
    '55;$*D3W0%@QfH7;tLjun(*={n8g$R$R^7232#?K1|-0IK#mrudIOtpK}0-UICiO?f{x(lLR2H{iy4dDRx389O2vQ&3Te0SIU$%g'
    'nW;!`<!_5;BzLwNCg>&OfkK=L)<APh7_%2bUNB23Ls$dFKazyB67xu&gO!jxik;yy%u!fL{2GWC<38k`CLLHMxQZdoQlkH=H-sc;'
    'gj_7TmsMsMBnzy1lL(&S-Adj;eamhs6iCABrAJ&0kLv1YKg6z7!naB{#M<M<TN}ozvRP|((Y+j}5y##tGhfVp2vLc36LBjsZ>^$v'
    'v-Go7GCW)#30g^P1VUYAaU+#j1q3#1_Qq}JSHe5Wwv$ub?21c}%ka6adUc4hQSH<#OFNLv#9Ih&Z=X(F%ZqmlKVW<ia4fI*@U9#u'
    '8=r4>3!A5k>Fn~ZHoIU;p<wGQ-v#mI75881X>n=D%X^s}e4%!%tF$g6|8V-QnA0x%x(vILW73otw-*0sjc;H*%&&{l%bR8~-(km$'
    'Ob|)g8{&XmMf~D)9A%~2Cn?J&6JHOE9IAwcSQJ+Y3$rpmR4qG}y;Tk1WlK)eELy87B3G7_v&vyJckCSXt(XH`N3hz(4l>Imyoz$k'
    '5;&77EjRp^{7h(WTA_-O7O1e!n48e3zr~x3_Jp?s!nRYh#D7Yr(qUevZ^iCmxoM9ig4lBNWJi5GfmG(z&{LV2rOLF<0bxHn9{5#~'
    '<JC5sGDffks~kRTvL-Kc`$EZ^Ba4x&4pa+T)`{@1dV0lFSNYFmdHOW0$w?V<@i1K>%n3+O3<{KshgmElA!SgO$ZNt+G!a}7wn}-N'
    'NkAhSDI=bev@IjTl#xb2jAbn6xN#p6|KVpy*znvC`SEP8h@G7sBpl~)L`Ng6#t9IfM-Q1_9s~sE<S;6=tc_|IWQvRO2qwxVt&sEW'
    '=w41jZp)cn))oGBrBFrXMOq`@yzV{-1A=m&xEz?n5Jm(ejUtI10>*M7gCx)iDS(LYF(SQ?lR_%kd(jk#++8y2DV1detCW=p-d?gp'
    'w<m{Yf>0{Oimz(t=73g?U#k-v@q%3rRqg}~GQTJ&!{l$|Bn}Cq?R-XGot#L9?1Y%t%R~ay<d<0UP_Px$19I9ajddZo7X-7Sku;6@'
    'A{1HSgz4=TIdtK;(GXn{P)ui1xQ;|;7|?e_mvUG}93ZJyAzgqWM3;J_U??T_EHG7kF%V8trn7MoxlZ>NvvY#xE(f+#Jsgj7w<r{s'
    '->RJ2#e|N-aY_M<gRk(5yzvXzn}{iRrRjOX&9{Pl!PyI!Cs89({#LN}!qr!n;R~Z_;ig5L#10hJ(^bN<O6b=)BeEy1d}U$5O~1X`'
    'hQ07s$UMdENla0eai_vA>J^gFmB%*-&<HdM{@LxrS(IzI62ffaH%giM7(K#v;-g9ULOeimo5m-T7?^lcMp*(fO$Zekl#9+xPKP=&'
    '1>xviDd;$%2vZB^i$jP0#kW7befr@II#eEyvbK1YSIE4{P^Xw6-Vbh%x^tG|j)<;f$Qe#Xm?S6OU}sg;{deAyb>f*2b0I&7Ga}@4'
    '`OzgT&YtI@>HT{m;?R6Z=39uqLbwA*GZ$$D=6;Z%R`xAL##0$g0-b)@@H}II4w(VUAQF-)%X>s&dnPgO)DD3Y)rDy6WlB(3xFp9g'
    'HGNlz@|JLwa@4g#_%-(Tax6GG$$?I!YA(eAARpw372jB}<%cXLnCNgoJJG)6loG)UmSKdIP!PfxjnN5!+7J#A`V}AyPuK(z9hH)<'
    'F!m3UHyK4KrGPM~w9lYn!t@i6VHYR5sMg^Wu~{Bhe-N4hKOe{7vvZ4&#%&b)?aM#@^y@F1W&bLZi#A`o;(O|ryv{~8(csz(=z~lr'
    'I4hw4zlb|3?o_+5oI@c(<+j3wCO1fYYrd=D0B7kNagDpBO*N5Qj54w%>&7Nzg}Q4z48%DncajK>>P)71hazzl{i{RO<dlNXDVEn2'
    'T?ZJCTu#DsHIscy6cE}Gp^%CWYKpQEOOG_Wk_l?j2TF;81FB@*qI>8AKV33FL5C*iaVL@h8EP-be<tQVp{=o0ta_udhpSd-T+2R!'
    '<io8x&z(5*x<%a{hsypgod<!TdpL{kGCWA7q3HO?=uz$*EV(On*;Np1wsW2}tzHQ-yzzP`7MdbKb@@lLv%ML`)eWr?kYmJ3ZQ~HJ'
    '8!>f94^_@vNxF%W(FrjrFENsL7RmcM5i_M;-!1FrRYAm0)!YcAEdMuBkQY(vT}*h9=)H?BBW7rq#dlHYpe|il5f>tRd`Au;B|46h'
    'os%aHrf0iUcC7}HM^vWyibU?(GMMKMfuM)FCo|rA1U&qbi*z1-sTR7nqKJ3HcCzRw_rwFt&S)%JoDC^fiF0|@16|$WRokjDb22Ir'
    'lNC)IQD4#j5O@|FfFc2Hn~pV$bpcqZ`V<p?7TaJVe~__{2+qy1aMApOtg!nKlA1ZR@`YS4JTlW6!lt9#lAo5HCUbfbZ-g|IeZ%g`'
    'W0v&|0**%95Aw@^7Im&##09yWSBSXXR-Z4v&?C6eT<Rn)%Z#$)Ba_jI2;yis2_W7Ik;2h`YNY-ZRMp9Qq<YvB)p+iklwidfN=3ks'
    'i<W1EjGh_yY4ZGtf4E2|$^A9X!ZkWuRz+-)hKIb%#6gqGm(c5%<<!I@$(F}f8?F)G1Su>e&kVV3l!@AL*-*XJyv`LFtDXyPJaXGA'
    'P6GLsF(E#789Ok-ysxlAAXYtDSaDUxBB7G89?nVhDi&cxx+fKMP0nW%q4dGq6}t)<KN!P_POi>VM=Qu(bF=$&Ys9igH$_N)?d8Gp'
    '?_QdJ`u0)sv-&Dy0X@*`RncSh^zp{zMqqgalaehEU`kZ+`!eyolKoccdhajjqe4G9g#P3e(Xb?k0A<SFq7rLA<}Z|vW4H*g>@O6m'
    'h1mWCZ2?PXLI)(TT}6VWN4F9zB3%(TpvqjL7WQ#uXI#KfzuL224-Jtf$XVcD6Nw@;t1H`AzTzk@8p32R_;5Kb#`3wvd!932D|bT='
    'T)iTkcy&a6iGRe?7w4r&)v?d3>6Jwe3}UBO2Q?rtH)$`5%q*+;o#MnB1ehm3hH+5j#3)i<RYLajJXbJ0y%DsUzz8rYYuZ7}0#IX*'
    '43r%Gx>g6NRvZbDW74lwC^&>5P!v)#Roza4S=UT&lxa2eV(>f{GLL}a`8LyUxqP&V0L1|*lF|}j2uC3{pmc&1sTBu4S06GvBN}??'
    '@B}I1Vl#d4dO^&KRFF8~mgW+}N^~Xh)(Oi@We2r(IBR_-eC@E>Fp?%zi-o}OfklU%5Y(<mL~7!VBhqz8+Ud>@S964J622Bm#0v2N'
    'hz?E~17cHy%{wHhcAsAh5xa3U?kgH4juevHOvEiA&+O47ugx*R#5^=#B7YI<ouJ1qBT%;p1W`7{Qc%W{t0RUoBc!0E6IN2>b}tdm'
    'vREp3RqscZ<pOidyp@jsSEIKKyGKeOabfWVBkjcVLjW%+3fn_~S=Oo(piko?+N~%X3F?uea5l2Yph8qYzq3+H5q_Qs0*<dPwuZpj'
    'cHeLC8Qnhp#m@UMDmc``Y9PnCBCc5cI*KKLh4fH;#(SK7Fdds=nu*WOu0Uu%V(=p~6fa?jx4dUZ3x<`=vaKj`iqC93l1B&l7w1PX'
    'aF;p9L&%;G3S@`Smz;@%E*t(RlLVD;^ElUX94DcPXLF7PkA4wWc##W4Wh-8Ff({g!g=Y?z-fhb{bvbhf``62T1f*B#BHl3Z!U$Q&'
    'YfV6f3MHH!un?LU@8bB?Z~k=r;d&9{kc=vXWL$SE;z9`ijBNXYBNo!X2tYZObEOb?7&^c*YXU<Hm(&ZVUt*r&imZAGA75Ve6cJ9L'
    '_bt(ev=rUQivts1#V*56SI7J&DhYZDULlcl*;E2S$CLBs5I~{IP@F*Tq`$ES33S-th$Q)G@L!;tlXHjEz+|Wa1wmJUUx?2G0^o%$'
    'GC(2t2|l2_m!QLHq%fxAtq_V2z6FfQ%n7&Wx$)Q_v>>PvLC%-oS{;kY`ss&EoSN>>{5mH_pm*?40MG;6Txb>VEFK_8W(YGmPh=PG'
    'XF?z1hEr%u2-!VTj4@7y&gB?q7iGKQdZf)lRy@d{BfKN&ZIz2x?wm_*LB1>+!w1+DO((`(i9nn_(;Zr(aU+R*Q+!8=fkQ;kiloFX'
    'OA-%6@LSz$*55jVR0CsFq{c$5M8AtKhz05lOyzvBx<iscV`f>7R0Uup@+tY7?$v!#c%MH0EDCSEZl)G7U3uLvfmm)!rjqEeqHwrL'
    '9DY$y@I)v?9AEfVta=i4T;xNTkd}&uON5t{y`&<2O0uF~xeNOSC=#foL;Na{)ZV>?h<V@FMTGh+gIPUa`|ydzDbg+wms5}2M2or4'
    '!SJ;i55iqbcRU;)b3P)+L3<z+MATxjPJc{9&HGc~VPrS(X;Rxt!-FQ*fb4Y5M5=@rcX$}$n}uw7r%ZBe3@dp(uMhbVL}8de+&6`%'
    'QQRHxrueW0`LvIbB%4q>66iseu^OVv2sEOHC|%b&5M)x;%0r&XiIzawddL$pWZmOrN%YXKT2r0NBpSEqg-RC1DbJ*IBJr)CYV8Zp'
    'ZRLyN$K89{F+djHA_pcoKx}skIl9h>)>*~~iIfP{d!H?#PlRC+jhm2N%djI!RZ)~eEfH|0FlIuhbp1M#shlKQy+}Ti2uG^SxJlTh'
    '6pugx!M(o;oUDY_3H&Bw$pm(mvE+TOomUhbCw%L1Gg<@(qUP^kUDtgfSe^YhV`9y8;v{iQ;KSne0m`vHg*}K??yLCa<EM{bfBi*1'
    'bAR*s4{v|^EEluub&)-F|5p<7aD>ETH6=`!^wcDhJ#=kX!ce(uwNiL*YJG5i?s!#_BjRr1c0=VL6+BFJ1~3UE6D;1JJ+TKE@#%d@'
    'm#wAtB67E7&c@Zp@~$_ph|H;mkCCh-S+N=ptqB{Yt_9&tse<Yflk5!VUYyUL{v_w}G}`O#9;e})Uh^JYe!8!dg=?2%WxR30<PUP*'
    '#xF#`4exg+G#w;^&*>WP{y@t9Z0$~fOi)a`v7PBy?iWrm%;)zn{DE7;aA+b2?0GhDr$ojVDit@f=)5|Cu^P#w_IOwAxgI(m>Gf(j'
    '!p_%ui3qi;ol@-T#p9IibD*6TPSqJmVo{|c5S^okMYN6NXF^Uw>w59UW`iB}?|7gNiS(57SHjAL!azo^A=-Rk7rrkRa?9P6YHy6Y'
    'KbAUcw~5g2F;GfA2D-I})aJWw;qKjyu<5=>;sf?zEaH7wmiEB(VgE+plgC1m^1yCl<4RF}{UNGDj|H01hZI@Dth}2KX`A+_K)BPG'
    '@kJ{_By*qB<SY6zeEm7eqCU8$?GX!lFaF1$K7aY{^OuhwK7D=rac@Izud~%o5|vkF9g%>_uO~)x*eh<3us6a&xizQ9cswf@x1jx3'
    'giL7*h!CFtKUHs&Y)g(^=}m>k_B+oUaQrr9WioXrnM{(Y|5TYphAJ6BMKX(48idACY}FugVmjZq;OD;VMBw@L1Bb)$v$6NuYk9vk'
    'ljMxn9%}vexq2B{kV2eqSsg~e$hm<hG8?N|UO+;M#=;8}L<wOzn_LO48Ak}NRuv_@SM>=R8m2f_ghM;#gU#~co4kunx=ml|yv2Db'
    'DL`-`7QtKRO@x7N`$JSGAF&wHe{I|G{Or^5|McUp|NT!te*MGSV0o_&-!DiO-;e9xK{>U@Vv%9I#Mpq`wGgSM$U4ZME(oMbaov9I'
    'Jxe<=iDFLAg%hn<|5be{NBVPv_itbj5YlbJ7<h9xBN*rF#?_1Qx#?c)(T(YV&9ceO2b%Hx{5)Gr(}3Ki^X$roe|vSiQ&M#6)yBja'
    'U?;tjxK6P~7@8Z%Z12{<dc9q#FO%HeI$dqI-NBY|?k!4DyHFG}yLU`BQojc+W3HPj&Q)^Hx$6|OovR++ZRC3Z7mm~1;4<>5Y}y<@'
    '!E~?0pe@UNLEZ1AC8-+ts`+MJrE$L*^|GB_5&GWR47Mz05T6S7dD!ENGwpKQUw1Z{o!s|#v03z9R!nEZb=!v6JFa{r%U=Wm(U5n?'
    's)~(2S6weJEr{Xv=M9AAV4mkH!~asEo9Xi|^7y`ETrrBZgt0mco}_n?582&!31p!AO_8t14#vH#)zu=s)3mvP9G9k1M8u@D^I(eA'
    't4!SH;9{34%`P!i>APM8Pi2yM7YSFk^bSL#IW$P2E3w}2!<SnV1>?NLR`X*4A!Hg=ir&DF?fMHBU?CRCy{sd{56dWahFxHRV1u*B'
    '!r@rQ?1xd`tITZ2PJVWkXTIY>Jk3#AUMT`ttenc+>vV#&qSU+)!^Q>T(%kZtbCoQC4$tP*%z*v0m(Af^#S|^oFOOANY=36J?F(TV'
    'mD`o*vZFE!F_3TP1um09ZqLSzQf^IWr#K^f6CJTJ`7=x}o3jJ-ij-KaCiiIszq@yD$IIduj{nXttd;$Lmt$>hS7QMIL>a+@Y#csl'
    'Wp1z@Pzh{!SJ;a-hyoUEAMSk|Q*f5pu;oX`l3aQ1ca`hB>idLMeSfTjmQ||5#N;(Nnd}F6njA0uV)kbdoLb!2dw3vH^PJB!aLHu>'
    'oF!}J+W>N@WM{`K76pr;Nox+8XB~sQ&QjKzVfWx@!GX$BMEaGr5b|+a7E?mJmVmQsrRg!QZYD^dC9h{KCeUM>Ih^1Z&N`Z>5lk9c'
    '>L;Pyah7asDBhf4Yd>#3+|>A1BLncgW#A*+sOAnoKW5ij`A*ZXVjuIJ-(5s1%Z85=pWDU=#3R26<-yjqA&{V>_r5*_4$o{>@Arcb'
    'dt>vmyKRoQ%Oh;hQF*hDE)T5lnsK~u>>aq0HjY^SoWgWJd=HXsVP+&jDN=ajz1+{nIHR-pB5u`s2{c})o<WK^#VcsV&=*)E0RNRq'
    'DPSLbulJPl8Z(YJBc@Wws@T}-E_j=BegU4pm&@JeU5d-ByL`{uyYa*p0KD4)Kdi+SKgV^h`5w&B;cE}CQ8P;<UYw1cTR59<cT`-`'
    'o48HE&%LL+4YM1^wmYrrG;)h>+}?$5-h4sXUd~^;zvZuepZa)JRb>h&+3ns*CvKc<g<#dj-CjU&?IoYi9QjlQ!`V-CO56JwG@v{$'
    'd-v8FSy^)`sh<&}xg2g&3x>YO%dC;|F?Jr?=Uo>=cW?W^8GN-*$v^&lYgQS0-IZ>01YoezTYY@{+!Zqos2HvqhY)_Y%_|c#$b4~S'
    '%st1hWA)H@>-VwQIHWlG^QUtk-_vi;I!S~5Hl=-g{&X6^Z0fGM?>X4)yyn`L`8C?N=bn$T`{URD_s6d%`uctoxo^h%#8((U2R62o'
    '>86IUxrX~FVJm8*1exI~wWY-2%aat%Vo&gvbR<|yF{ki8_w{)^x-Z4-o~H#Dr%-n@C?KbU0^(D4!Il81S)Jnk%-b{oD(gjLML4s3'
    'YB^XtP9DtZTT;HQxBNNU+~qmh+~Jt#*(K6+<q^l6<q^cx3n@P+rI4hG>R%zz$G@mV{6XfM!xwS1M7H9TsE64z_<nP|)yae{p^0~&'
    'vet|v{1ur(4rzj5<R8P86=A$v**ml2WoES`Rz4ZX^8f&aT9~D-{EK?-fz`=MDTGh1hw-__G1m-tx-M@iWKLEwQFO`o-_JAXx;ug`'
    'Lii*bzKsH!#9hua^bE@9E?)KnP?W$?*B|58#Np6C-bU4iG?Z5&(XZR6+L9(>9%=06kzDE$T$3hItJuBbFf!Ubleik$?C4{Mp#DUj'
    'OfEbc^+wd?aqpj!0`;aI&i*KQ=eKg2mjGg4I&YmOYd$jdOSo+7XV4vba-pa98K#kla}pedEfwm;YucLh5!eG=YNseiG#i}(FomiT'
    'JyUn;G_K$#i?n#Wk`dJ?(cHGEmbbDV3DW3nQVBMqLIS^Mw50GwM<&FuP*g}ncMls`;uM~rqB(>K9EHbvo(MKq=h1F1MMU(D92QRU'
    'G}5ACD*-~PGh-XbYG6#6w3Oi`F47us-nNmH#r?=p>3nWc`VvE#u}JSoNGXK5sT|TzP#lH1IA32XLf#Sp7&4F29L{9*-nuGW_xWvA'
    '!L3=!Qgjx((8aC5jv<btbZ9rlFhSDKr;m!ox6d!W*7HC7-@p9l1KOPRdw-8Jqp{!NC%>Om%Q2M_2kPc9CEaB#!^Ib#G}a28WlL!)'
    'KzU2__n;K8EB9UP00d_y*{0MpjfT?DOu;c#q^FkYum>==9E|VJI)c&07R9wGT6+Y+oze<GYsi;OCyWGFf%~losx_8?A0UsSSnzFF'
    'pJ1A?1mf_%rX&#x8ry&&9225qaM=xE>w+Xr{^2w;wxMR)m(t|725_NxV1hP$&%@*-tjT!^*m)0&dFeo87dFic$Mn2x)%&$3rnvRZ'
    '%h0C&OyOXhP8g7esVAqp#y&EQZ!$dR<$IL<)W^QAZUu|OGz!B`F@=oULwkST*P1zr@KFq-j~!5G&0F5wJXFZewd#I4(nAL5#}4dn'
    'zMm1>fEEbP>$I=?SKDVupMEh50C$uzA5MGIrQK^{&sh>@33%`Q9zt?FTNp{9k+Y<h`jl>d>nPzQC#oDQ?yR<%UaPT5IQzM@)4M{C'
    'b2Ra3jncTY6i-3ny!5;SSD4>sq|Fw86+ez&xtSfo`vwBe*B9KBlD}=1gh^<Yt)PXbmI<fK$zmnkBYf`|-IKt(V36}y+Ctj*xhBkn'
    'K4^kHEtLZqeYdoqU-(Ay3JbbCf34>iw?y-_uNnj-xLKc-VOF2^e!=t~>-Q8n4{&B?ah`soJ2HFIs6X(QnSEgMWu7f#nhS)FavCXH'
    'ez)Y5Fvi`Io1>&X*MuF@=1bU$V>|#f{#p<0Y$i78_~YyA^xpx?;_w%)@3m&{UYWJVH8Y;Ui;NY}DyxORt*jQjCH*%7uXR<i$6b8{'
    'cmb6*YXm>{XJvIgo|ig0+xc*S0Y_cNS$h6@?iCD6Tv=EYr;=$Ll)&uguU{+uU;q2RvC5ZveP1IAEcf^K;X<I_2@Hrt(08P&qCyFX'
    'Ey6vyy*e%G+9Ji8PhIp6%hO&waMfFHl>TBsv=q&4^M-K}26-CDOTApCj}Y4`#+oXaP@LPOBgW4S=7QnQ5&KTYK3B;@<lra`XVt@4'
    'lPGW8LBIk)%wj;RhEMBx%@bxdZ?K6Rz&~URAYvIt28yTLwE1)Ihj078{_b&TeMiAuu-*6Ty?EcB<IorQ&MX}Pc|U>lzJPmQV7@Pa'
    '-%p^w4-oJJ4*UWPet`(TfQ3JS4L<<H4=C{qsQ43j@e9!S1#<ie`1l2e{0Sgwq<%#A@D5s})d#_IpVu<;K!*5WdAdj>OQQv`l-zT('
    'ewtVpjF~19#z?)3f!Q=78<x2@HK>QX1_Q`7mVNWWah#^_@p@_j-gSPi(TQHWg(t8sdkpfUcp5Na!9x;ZMG+i%n%Gj$sJE!bHi8v6'
    'h)5BKN+Stsx-}DpEZngu^QB#!lMz3jRyEW7Y?@sXVN*(cZc-9Hqv<pL{_@M8|NI_+6TVadzT^8W+syF<UR4al&!Ai&$9%JOan-8E'
    '5dQ&%@-0cv`ut6fUdQ!iQoFbZrQynz6m?zncM{W$xSJX;3dFQ^ks)myFDS{@>!~?j#+FjDu5rtl4v&+<YTvgftsC!A9p?%VCnCPY'
    '`WjgrLg6^0X{<591YI$!%)R4Gplt{0X+^cN>-RG`oKyvK<d!YNZ!1H<g?m@pf~xOV{sIxg-4#|2gzKBotf8$~!!^L<G_ic-Ey$tq'
    'y(r8)eT24Pe-nAm%8&wkOxj~cNK(~@91P}?j7V#WmE$=o%ekkRZ;d&=ds5Sho~z|JOhv>FP_+`1sVYZ_jZmWxIUNgFPNPPKLy#JY'
    'iRN%(Zs3t@80_XpVZ-OjyR*%8VsqUlfRFj@8Fya;^{@ZU80r50w*N3N2~ewtlFes_*NK=NDY<bG%C*U;VYls5-spMQkHk52uw@0G'
    'WuLP5-!cX9i(>9qjgztis4aNSWL?T%=^P`3vLqD@v(6e`IBs{6H&0u<M%KsF(Rf;#?PG+I6h~^aq;pliuR^JW+e>EY9t4A|Sj;0n'
    'Q3r@#rdes02-F*OIHi(6UKa&1<gV0#D+d^%#-{~mJ0m&RSiIbg2FHteKE|om;qCJf-St_rb#^>rg{{?mG($NO=MlS47Fq#ZH?Cj>'
    'cCyRRWg@%G02jau<L}c!Of8F1d>fFufMpo{B9LJeOJGIaiHMDa$t-DWaOhHS0o>^@L`S04FgOZv`!Enl%xyFgBpg;SX9|3~W_cM9'
    '9L<m;RMpK8AAVL&@ueB6AYjJsKESLN(h~*%h{D1=jG!PHG{DmrxN!!vy%5qIeq)yCAtH_MKBnPRW~(D|c(^6*WNsHIz{0E@Mv;I}'
    '4_)be&oLbp5EQa16^5I}V!@#@bpNAsHE;nJ0Iox)zsE29?v)eO(1o9ChUC2lEZx8jXOapmcrc`^;To9xDwgJu9O3c+4P+dEl%{U='
    'E8x+WI$;RsCXAthTW{N^#h2D=@HJWwGZLXK2{BTdSY`C-l+1t(I~JkJb~{*;Dw(qb>$-9#8Q2}CK5NExAT%Fu$`EKz0@Yv(R#bYh'
    'Rh$xFdCoY44^T-q8!WhrH;?|Qh4@(oI5*frlsfk7=Pk(^s*tY^DOU*O=C7n056FY7GeGi{YBnNJA40@tK`O&-uA-*&kR1wC!H_Wu'
    'I_@Fmio}w{;VbsoAhM%HJ0xi-W<lQG$8T$SenGz!M>}t1UMJaLmGxjSfK#7NVOEAgc3CwLNrq1ad^hOC>Y?A5x$YBXQ*bEDVlHUI'
    'OwfY;Fz!ei0H$)(G1hP%7sw@t0b@4d5zGT~^<3j=tZf{Ohge<)wK<oC!`=~F;c4XGDJU>Fc$Ve<iEXmXzMKL!fPoP`1&;B)#!AX4'
    '7MFqNX|Mzpwy%MZPBEX9q}&1{n|YmOX0HZY!KgDB&x9RSd<Zn4#b?lef_Y$!%W5XpO+=!3r$u^R*E(zF%aC<_e!;qtzt-oPfqNGI'
    'z$eXU<rpU+{=jWhrt9voeIxX!h2=qokgN=sG6-Gj(w;p{+SBy?<t|7j<IbMODnH%(+zn>x>mu>Y`rMIiASc<Hp^6ZTv*djhvjP*B'
    '!Ay7B9as&y(Cj24>_ls^Et`v4hTPM`U2G?3y0BHNbtP6st&k#ViBKE)4VUZb*k>Qcu<UpZ%8;kqy^!Au<dHwmz25y%JaPCNu`IN>'
    'hWCZPyawSSTV)B086j3_I$ZgI(a05D0BXx`S;7KGPfGg%1(VPgx!;PcVPa=)xz%{1TP9I=L|NJm70VXqk=T65DPnFOtD@Qh=1i5P'
    'U989$$-*siq#Bi%7oUKlyKdM1W*T&kj-|?u=Sewii6!7MxfnUjV_gBnm1k2t**d)61;1Gr=}@$|I764PbzVW>cL40Am^&}dn%v+O'
    'l=Dn>ycpn)Q&tRgXAp{f?wC_0l)d6NkNpyBW|#3;UD|1kD7gUq?uskn0T&)7hxz9h`8`&?Jhuv4%22D+NS?)Tlb7~$-}#m7#?25m'
    'V_QKJ*^RYDz2*)~f24GN@tC{+<DdWS$G`r90K{Fs!zdM+^Y8uk?4GmPtpG?hDNog(%ArGQF@)X7i~|o`&4fS2)xu!*Qc?9T()F68'
    't;A$<#*{cR?c!Dgx^h}SK{sYxB74caIz>*gponOWDMCE0`!sdt*#9Ihdv17|{$=3*{>!gF{#nVf{{7vl9O}Nm#{|*TgNBxaGkpJq'
    'DSLF>IFG6_<#}K2B3a^E6yI=qh40^^?@?^Kqe+K*LbWC)Ta|s5YwA)Q7CxWW^+i!4IW8J@l3007)(q=2O-|;2-~+ax%PA?Bri3Cz'
    'g#Z{Qs~nlw1!@mSFy(zDxLobr7tIalsLBZ}W%(;#rK*iEZF3lNO&6imrM#2(m(qSp(d*J5;eBS!$^WTNx9UpfS6Q>HeU_pa?=^Zn'
    'N?&m2ShGw*>c7bl)SpsGoS9_sloSQ=uhNBk%@lv%2m%1Piluj!W7JJX-3JMF?NgS#6|Du9l$6TzR}Un9H%l1@!8N_)9E~bO3jyV~'
    '&$#-G#ozw&habOw6c_L7`@Ju6y5@IiyznDE6gP*G<%^`{L)-EYxqPTyzQ|r4`j-zO%ojz>hcw1|<fYLK^b^(06`n@o*ZV2x`6os%'
    '0TJPSNcLpS;rjsxXgUfGC*N)QYIBktE<HvN6A0r_A#`P(rHDMaspQ;OPx|JWRjCb<AWB#?%wVQ-34p(n&E>e}pd<GlZW=s|5(TLz'
    '-Hngj@zk<)2=V4bts7UaIFL$HF4cv@rG=(5e%1G|@-TrFijgO`>e7e%?nBQld|uYtN|y18a_jxg{7{Z-WK=3{ixE;8?b|sEEK5eV'
    'Ddq@gv#zKSLOWsig3yaem|Z{~mUlNTB&yWlVrJyYt&Ht-o9_vlhh{RJ#Pu`l*oZ0!tFM9y06>Vs+XI5A=6l+xE%xU?I?QajO^A@1'
    '>k^l<GH8q`GqCvGos1R+uj&rTYZGmYYmsffu%J6^OOfl$k|UVc2c3E!AgiTw>XuXUl-$>c64*0H^;>s3luCxEPi~QL-7Hfjz|EUO'
    'xa3}(uC}>I3%^h~SK>Ear|{l{m;7m@)w%NRp6oSpPg5WWVs;g=DktmvQ_H6XAJNILtD*<>$eQ^~d~KS!s)WTA>NWl8Ev9o4?fCjP'
    '`8i$hZ}faVq)mUjvi$w~V||ASSI5qGI;VXc{-1aHyyf4%w>|Oco3HNr>cX#X{p#wUy8rV@KA-99)BW^0zdrHTXa5=spW*Q}WPTb('
    'UqkB|ZJ+V?H6ouu`ZaKW8q1&2{fzg|4EUN8pIP$Eo6k)8%&pIC`<j2xjQpCUKTX`P$^13JKa>8m4t%W)pS9w%dVJQC&kDl=<*%OQ'
    'hy|WCS04ta+ltha3eha^E3B7O0CHu47mIj{XyoR*1n*p{u-kdaFHO#?oCZpf9hw)EMD1jZ5O6bzaLiC+niWzNfT>y^w>d2MA|4E?'
    'beSF)YOv)~4LV9=MZHcJS~bDsmUNS27VeMyLXKr+;uNnDy)HkphvHk`5C>5f>FUv%DVNw`vjqeSw6=I;aTgLEuNbxP`R;EySV%0J'
    'Yw~CFw64F1<GbFK+xzAr*sNE$y#J<Ts(R5mg4I}L3-uZ%VU5+e<CM8&tn{x#*N!DcYve4^?IOpBB}2xzkBY%}lVbMa7^#M00ekCK'
    '^A}~$Cp~rt?&J|>hXtt|mZmFCXBivHG7neR^+OD((?fQK65vfYHDv}^P{fRh6P`YU<F-qZPDV)uG4RT~PB1zp9p#BCiVnsW@n|T0'
    'hlj~-=&t#V?sH8L!7zJIb_>T#*tHx@BP5h&A;^_gLKskts~&c`lfdfgvqr*xS@oI;32P;Uc0gAqjEwQ5>WaR)pG0GZ9zm>U?ftCQ'
    'zyI=g|M272|NP_czpd*#;c<43@5n;z#|E4%CIrV6-UMtkS(#buo<Akv&N)V4Ebt;$vnete@a*~1xi9$a_Y<oS=6B&<X8bK;tx$C>'
    '4o5g&7XG(o-Z&}V1AuzHO-@iD)U2!`fE~Mu?T%;*=P=lE7H13gv_+~CfuW^-jlaTTqf)7)$Rr12Lu%Y2R7c=!3F;|5Z%j0Jf=a^I'
    '?7K12Qt-8TGTW<sjet5JaKwjfJwRs`0Y66hGKr({rtNkCUzcf*-h)1r{m9<ZzCAtmb$0%jAOEO!HI)C-U$zU~Ntf(Ii)0>gT`5oD'
    '3X4dQo@7Q7i$4V{$3%A~3OSnE9)U{U!jio*f(*_*qH$-SnDqf5RdE%PDcpf6)dy)*?*n;GA(ml)$e||N(r|`T>|Ckm%93x=7KVbV'
    '9wO2^C^lAse&|oHgWDu=ssT8};<v8nS8-yr!?l_&GN@C-OKRv{w~NS;Vv0c67NxB-dUSsNWCe4cXeK8^uO#-8804)|lB86o8+!oZ'
    'aEn8y-6ky;{|W|ignQ{FN20h}fCZpJTZj~iidJ3C`S<ki=56JI573i~hk#?b4KiPA-{EhR;2n@cwK^GSdT7V06=Eqs?JDP!C<kKm'
    'cM0jEyj?Xsst}lIOoNHwEGYR22es?)T}Y9_z8Dq}QULDId5P~^Oz(pNyh6G0IvAao2;yE(LA*3$L2;z8-!LS!FAh`Yi+^{8zyk1i'
    'iBVQKyl10$VL>jXrA1N!87={iz1(VqfV$`0=Smn_!D5Mxm5WL+@Vc_Cy>iag<5NmF99~tV3D|m(Aw#t(X^%vCiw6;8U3%~;uUkNb'
    '*K}T}czGWRrk9$;Y5<CK8%NKY&8J46Nt&lu7sH~aT9}RX+l=5#UVR)mhiXMZ1_w%G#(?ZH(ZN|k%&2=L7eyf5YC(!6VIx$3=akwZ'
    '&`>iey&=qd`iv!F26X-w19nBBMyiaL93U#}h}2<|e}zu*DhDFI)cx6q8|<aSE}BJ>F*TKcSC+Wf6bC23_j-oROs<5AH6{4P{8m7^'
    '1X6(sn>C^Kuh3y(?fY<-sgNY{RWN(I)U0ypQDtZyEb41WhZn7*Ljq`qLW|Nka0&Av6sFK^c<5l1MD8IPe1%7C+gDd;V_v$cRHv0_'
    'wWym`>$y*ihU)SXBCF+10$;-?^9uPYzN-~_LH_DY$BVV3A;dK*EJRgoT_^hVmXxq<J|%(mlEd=dSP~_8eTf%#@yazb4_qL4Bk>Qc'
    'ATM0qO*l8(Q&WkL5N%`Qd10%)OSD;rz;21Znvl=(LMZY@sI=YcK6gv>U+&w{y=t62;Mq9WU^8vEnJzxJA63C7$Q&!W=|Pl&pF`(M'
    'l1KfiERkumYv>)Ohb;YPBN!goa8Q~;R$Xd};t5NsOipZa`n-#(THb2u$sXU-*VI9kr6wu}>#JxD=Mj$(U`~&!>Jg&l)4q$#6Qq(t'
    '^Kfas%;}Zt4f5%c(<j9v74b5yRR*rFde39xHZIx3Jd{#l{A+GwO%V#IJDMC%CM)1?txn3*r0El-d6Qs6u^2x0*CWmoqpM;6>0_S0'
    '4Xaw6$%WA=H@!hlH^xCRVm(b*NO4DDX?-d?^*&p-G8#;oqGE+Bjc0;N*oLbkd{*4+PqWYGXHP9_Kh2U$PQ5Qfj;C2vTis@aEl82}'
    '^|MbccSJzA|B8YL_a9f;%Ou0|g(a_B=85oxddN#o4Ue(-AAbA&umAqXkE{26sQCWgg}aOYcT!TIu{!Y9undPdLZq%Z6NPrddyxzU'
    '5iU!HOX01ODakZYmk{G(s4l5R7Ih30YOHMIA|i_}kWdy?ZAqDm%+ZqmjN%9laZzLBx!MH=_NTILD*BdU*at5;hK0g%PI!vms}QZu'
    'GK8#(Dj_*z<=t`vHh^G5&T1{$FI;Vh9K4!$(R3=wUr16^3H4wz6@AK1zQWP5!$e+W$`j7WRqky`I>#L+I#`m!W;}9+>B{1DkdY-U'
    'el2<JISP7J3i6g@_LT%1iq9Fa4j&vMkvziF=Y2hIVFV4bO75M{gXJ&GTS#<xir9cKniQSgVll%sa=wMl0gd@wP195KXVPWP`MgE<'
    'TZq)vX;lT|=F11-D?(X)A2E`tY}o7OPma_pKWDBMX*qMqu7@NmKbxoC6=q4=uwt*^@;T^Py87^dJy-X7+H1CY0)?~gr|HIrb;!l5'
    'P)&TBhms{T>u{@>j}v-%`EoKmnD@`Pw1;fybMN!hOQ|XUqH9*p1+$#;{Y-AsS>S>xTN=gtse4L|Z$CTciFAF{dmf`rD9MBl#am{H'
    'P)DADCgs*)Sm<GAD`CcL!q|f1&4BQ#OfgTr&trsC`}Qf048f6P1AaPVoO2a(aeapCoDcs_rox?BF~hZb2GJS+cJjTyecgM0DmiHj'
    '7T`%<d!~9O>U@aMKKDLP708`Q)eeC9ysP8{HMa2?{JHn}Y59DZ^rIhf7`DwoHK*?I_V{7$Z(l!4Ew@}{qCao=2r>Xo^_1Y88?G}J'
    'KleTjxMknVbT)?YdDNXAQ{5$eLsP=G{wzOt&-9>%_L&}cx}RAkfa&vyXCwL_{?qTj{6ByB+KI@n7(z^6b$_4lH((PBLG_mGixX7U'
    'CXXLdCbJqb3v+9$ayT+@<zUJQr+VBfN-ryOq3vC7Mb3JRxFcHe64-VZX}f@XNlo-pq%VofRYLurU<f})_*-u1e}e{)H!Wk;q}zih'
    '_<|Pv1UUE^hM+)oa0)*`7QTQCRZU-yQFY}i%#KT?Uyt5)l$t5kH<p#R3Qkz2Nvh+%ge-VKO;&b3fU-mMnMv1mt65wm1}3H2<n4-+'
    '<aJkLq871{efrB$;mZFm%2nIUlIEKlpw3m}(sJ&FLrdzw+D%0UwV+DnD{>kfZeK=%3{g;2sM0)a!WzhOJVj1|#<avVaxXCkT>&Z9'
    ')Vq?1bDmafiU17F<z~eTft$RO_1d4SrfIunS5Jbf_m!Qr@F{q~dqo4@wrtb@Z(iAZHMXs{z(h+?+>ZAWg;;sF`{1fyiYgTdHqvy_'
    'e$Ul5&dWbVYe>_JsI&XL+qtSX3CX9g5FD=ag+ze+HE9kOE4tMq?(<ichM7;urayJf7@5(8z41?pDtwtrqM=JsJh^h%@><x{lI^5r'
    'x}>lrUusdw81)MAsG7Pru3yn{$aTxK!C1#WEW_EtF{KR8Wm=zXvacG5+%)%|1#vj_IG5<te|ZcmoUvb$R0}T5Y!OLFQ81%g+jl>0'
    'a1NE3r%iFj{%jeeXixD7bJNu-b65R|+lem7&*@a96Opz+v%Sc&kVlw(o~EaW$tg1VWinXBWBig*t_GVFIF0@#NxdZnkcPTmfmKb5'
    '&stK;DYi50s{r9Dx=DmE(X5x2RTb@1hZ1x4WEY5RPa2DIW+|yw#o$yi6J~K&A-=8bv!<#p8mLrldd9lqc^va=cYWwtf5jLSUmk8<'
    '&c#)5$FQrL+`BH0Om~kc?_!q|ytu^IPzJw?@@m@;#_T+e)jJ?4?$xjAiQPD24qCN-4#Kt8qf|(-eEZs+-kViDdh+TuIU*9?mFyB<'
    'd6m(?1B}O<cTriy+j~_sbX=1mQddSW@3B;F7t}E$Ilok&YiD@XdFthr{2DApRS{{av8#hwEjiC1ulsOxJD~!{h|D|(#;DLMMwxrd'
    'QJ(bO5+ZQTH~lF@5$4szZPz3kciy%UH>y|V$z8btTjZ(X3oh`_KmJ65VC&!CvxHsf`}?HM)Cn(UJZEK|^t)G4<Y7zskb;th(h=y*'
    'j;zs9oQSxDNk;MXsH$;G6DfcW#lmuuk?<c)<3MPhK}yql3evAOzMkRcCR<+#K(B;ZxfAve5>zCzBvhAlX9^G+<O8f#g62fRubhxw'
    'mz@kg^d6Gc=3jTf2B*{&2b|9}SusI^$^*!f&@V%U)XZ9-ybITr8f1LR$W63Gs&!Tz0<4X+Y{|npuwwx>1q++<nRhyhZo!m~tgb4!'
    'ms*!eC)3IAi2~1{baS^(F_7lRm8p1&bQ_Y5pRfA2;{ck-zc)kgz0(l<`u?tLeJFH)`WBzh^Z8)oDKsO7dgpd>`R>EUYzSt@J=sng'
    'p73`G0X_u7x*dxVmHVMbZlO4)n><FC)~8|^A4{cJu_>*uLzvsdzDP-6T2@gjREA$NLuGC%j}~Xa()zsKNAdgr`_mu){CEHRk3aqP'
    '&%cqG-1ql)mnmSrzF)B3e3HXDG*2%%+eE3qgcNrfR+j995}~Zn7oK}he%4Tx4g;^4GZOxgcUi|;5qmz?Lk05F{m<`p`N>VJiq*@0'
    'qO1nmGK)tRo6Eeckc~X|6d7uK_7%1oG3-T#73sG=^`GZm`a`kb-{o+wh-{bAxR_xuKsucE_E5AG&4Tc`obpr8sn^4SOnj%vsdE$w'
    'Pwfy5eu+2-MEftYI=+6ETDojZh73(IdqKFxRs7u2^fKRcoPUa5JwJQi5!+X?R?`$-F?BB6Qq-x#F>Ox4xa48Q5Ipx9uViI<8N(b`'
    'C};0)U-h2Gx)(|q)~C?}2M}j$37EKn0VK}Bbv|Tj#=*3b6v7HAZVwqPlV8w$SaZ9SHQ1L$*t~o!6!Fdnr-FfjKfH+EHJ0(!veQt#'
    'uX44S<nSU{qnP>d0u@KmpoTo0=icY1<?AN4b6M&-UXmC4TM_MR45A1XemNn<?w|fh4LCKr?(Oi%X&tARo<}^rdhd{06>-6*!YNNt'
    '^W}`Ev~=2i*nU4hJGBhM8{2FPqJFL&rewTkNJhI+59)T>>S_A?>{E+DA#XB<a-~Sf3lsix!_#&{GY3?}{Ym%00?9YBNbeOT1Yh`='
    '_!-Sw_N?K#+;A#8b#p#`ap${eoW{Cw%)h3Sz^lVi0c1^@udkmyEp(9NBf6$J6f!zz##aAyzrmt;ef#>^Gff%T+abI|ALR64G|OW='
    'LNy(pKw7w6i!`{iFlca_3kuE=LaEf?Rj0o(FdMp4L$k=dY(CgWI7^=9Jgsv`_WQGL-wsK~tbMYiF|uym{?ieZC};+%Aygq@pzNeK'
    'f~}U(mbGWKXSQxhu(LIs2S0u7G|iO8`jWM<6Qg~Z0~7ZR1Gfl<H=z2AhC3sNVi8I)jifq=sCU?W&u=3GTJkn$RASI0qm0n0!Tt=b'
    'GCmKkUu(&IKTD@embt~nIXqlC?K}63;sUZwJ>}dhS>-9-LOont-6K-8hT;l+I||ke&mB*jPi2?UPY<FJw7FuKd6w@tB-*#jKaG60'
    '9-}xCMvr2g$K>H*SoSH`Y@6k$GOK%)9_*6GvX=TOm$S`U?$+6}p@qjR@1L@Jo`m)oMMYhJ@OJ&wwvVm$r{Dg|k7&%1td2l2hV|QM'
    'GYB5L{bx<_FP-r(?Qv}&&X-i*IA%+1g3#_aiTxXEZ&jDmYnKKo04JKO)^GtJr53zynC8x58h{I!qQ>hwseR34tff7sBP5!{J-EU)'
    'O_BtFqqBLcL~NSjU8BxkWd3ma4O`{c!q9YcQIbvOQ1y_OP7&%8r=l9YrU2Vl+a`Nrbvn#URFLdj+AUoiY89na)mk^9)WgSh-q(af'
    'rRug!pHlA21rCju!&=l62@y5Q&?d6lf-sbyV~wfJr$nFnS9O23DYtizs&rK^d7FHzmbBp8%V(4?9r>R}3Zl1+l=giWg(u(U{iTvc'
    '21>z~-;!x~TVgQQhYjN}LpXirDZzCq0YvGMZ+1uFP*Pk*W{xLpa~R|-7sxZ2eMo9ylb<`+v=BMdb(iF733QiK8MLa&aS?v*d48+N'
    'q}M!4O7BfBOiK<UoteLOp8+&@`P&j2+Xi9D`747=)PbDE;G>%a<CXyF2qJB&x7>1aaN;8cN-^tl%QYflH}g;~Vw@r^P}HQy@E&W2'
    '8^Z{C!&!!zMe<V)1zgi!rj}UO{K6fsuBYU4%}hPt?enpJ`Sl-v{L62D`5m7B`}-nc70qm7bEKXho`1o~@^{|}k4jkHpKAgX>+@IH'
    'l*3)%efjOz6aV=8??3+TAO858V8~Lw&tc4OeSfc7xLhBR5oxSH>z)82S3_+$4{nLk7b&|u-jLW6f<%C>_?KjQ2=EI$9C4Kwat&r!'
    'a4pJ3FLOz!8Y95z!sLjfUc0*BISmd0%wJ#yI94@g3*IyxvxS84v>E6yhU3ee*cHmp>r)~vC=fM}u|5}{ZSVp(>ToTcXYBlS3)JP>'
    '{gj|y7REvQwwfydmAAVKzS<CE^5)xG@QOvyyh@$xVQN>?8z{*we)_bW^O#)Z>zwRltM!DO2eSqRr2+~t&=nxLY#lG9KLUSY3k+xm'
    'G`ZkDx6XAoEkE_aQxFN!TtVsLn|3#0E<GQ-D$H>6`8>le2H-r}^J(qaQ_E-E{qev4$A9|cpWcV_zyH?z_xGLviTkUm=}iWDl)TY{'
    'lrHs@$mAv$?1dw<<3@d%DgE5{RsHY(7fWWnzQ2uYps%fcCn>!=j*!anvk?iAvp{vwEX^(13wUA7fnWxk+sPNqUxAOR9a@#+ZEnZC'
    '|IK|u|KS>v1EH!)TFv9Uqb#ahxFwttT~z90RTP3`eITYP>6_xDD)6s%CvT$#`kI<?E1RRCTc*}1$KIy}`eA)WO`b(N*wv`X!d+n>'
    '_*Y=Mt;Vbn5Ie)~&<H_Db|#v(mf+FoD-mM2VRhPoOsj<O8q+V^qOM~3CT(-B#Sst4cx#;4`VrocJmZw8xgJebr8|5^Ns^*Y{-h$_'
    'XQO9RD7v=P!`eEeSZ`&cVqrh`DFxDxJJZyOM%b*dCpSeSG$D+I;}U4bm2xdlQ%&{?J4F)(AWqM)vm%VfqNR~)Mll>s9CH-%>+H5$'
    'z9&sE44@t<5_U7+lD<!gzVB+5P@h@<sikYGu_v13v7h}}&D<{E5#zb%xigejMWJ|7FiSYGjC+81G*ZD5(F`-_vNJ2@%052B>CwUw'
    ';%;lQ3M0($ipCDmi&~rkaKN1THa?}VJ?L%U{Po}e5lNxf+s2u1i&BH*JDdjKBF+w^8ckc_Xa=S&Jk0!|CYvx;RB7%379&U{mEi5#'
    '3T#tT;H4v3rUZ5;Zn(qm>(zOPdZkSWsV{;>UIoSADw!LE8&+kBQWVT0ys-*SQP#?2BSXsWCT@cig~M>wk51|mh`jmS;HNV$yl9u1'
    'XkJ3VAZ}8-<0-CCI=qbBou`GsL5|@OCZp;NSLl*nCEzhpga_(#KQ$PjimJ;U(4T!)9)Ms8ij!EC7Sy{~gwY4oh}ES#QOT8n_;mr_'
    '_XA%nf&1sGovH+jNY!GH7cQT7ajRWmmn1bz=%R+A{_t3jd5b>2ilz^J-tBhNm9;{NC7D4)SH8?pDrztly(i5VlF_T#vMIsSQ92_5'
    '%?CpyWZv1!xaJv?`VU%ghed%Jd!4V@Yg!g}*7JpiCQNv}{B9z<Mg~DSJS6X;*%QP5^}OtWaVa`iS*b?<^kfLEF`Q*10g&pMRY+?b'
    'vX03(Wc@6>TU^jat$TKKljSm$a~Npfz9!@cp7Q&T_pbYQfBx~?U;gmRzY>vp@8=|z)VD4BHdn4kJze%lle!~~+~sTKVA_?COz9xt'
    ')z|v`;;5&~Q^qcA3hHnV)H`eb2UBcBe*W`UT@8IczZf&Zp(XOK+Um-`>JSjwt*yTHDM@vwL=(}wQ;JY++G5MP!db4*X1FW^X5~4w'
    'st+{{x+JZnK$Ca^^I<?tiF1|=da{2>iP>f1$-dCTBh*MrE8^PG+si>)`(M3r(-{p=2}_tFv{^Z)7;#3M&$z9ny06yrezGNr&v&mK'
    '<D}D+p_2JL2!bXhtA}|LG;wV<(=ru2b~Uu67!x9O)x8xhp(JmS`j3RPq7MYZciHcn(GU;hmmrW;BZ6GE(#H%Y)m1T@tGb`$(<vDy'
    '?{YQ&&hW75$h#anM^<pzu<oq|<uAa!RC&L(!=@zsD-rwKXmNcg`m2=x%1MLtGIOHJ+92D&y$>I7jez|6l=y{W_AC(66PUXpy{zC~'
    'weOPp*;O*7mOc&}Yk*Qph$r(zyiie*y&tLUF8=~RM9#-S!*+xEWX8z;rEF{u^!iRd`T_BZRxfwZe8#p<sr0~hg9hecU-y0ZRz<yw'
    'DXmh;+tM8{-QgLy<rGn+Rg|Sn+$N%6Jt(~`PRnm3ltme9xi6fr3R*M{0zOS5&qyJ=R`40+nyf}GXFoUqSt3RJHvr2OcVb%!)^jb;'
    '_A9O?v5C#^J9Vh-Bds2McQ`GIJ^46rOdB*)&wLYe(47Gh1SeK42!O@8!w#>kcqgGwXmKYiUuZ_h@R9+ei#Fh^p8JZVDeIfyxsF4z'
    'vM6^Y#QC88N{ZknYE25d&k*PZi%@Y~ab7AR7UmkL6IrAH12_`9OvoHZI;M{K(Y-rAcv|<=0T&U&(FErHNr?WedI34GS&2oFT)1`#'
    'ZuutiA3T5kn14Th`@8?+w||ZT?Em`vFTeiBU;ah5S-Rx<zHuDnI5-pe`4(q+w46{`IBCb<+=0V)&V5!NW~y(pcd3OlmAxWhFJAkb'
    'E8C@r6y&Lxx$Rqi_0_^(e*M?i|NP_KHqigy8)E0Ipx-y-3LiII^?0EmT9#yuzUv$>p1*$8`p>^b+x1=Jd*z5J^nD;z@a1-yu_bQ}'
    'yD6yXSXi?vfFVK8=PygErD{OMV49%F_)XF<p4vW%8*b{d9~DS_Pr3=WUQZ5;i5jN9BsZgIjKrj##l!!tp*)KQY6Mcgs{Y%4{6F7c'
    '_>b@ZzrXhT`2YH!e*fdIKmPE`uXqApqkNYm`P~nfWO7IMSM5YC&yd;Oiv8*qxm86G2m<1c;vJsOub(|)zmgN|{nYXluSm^(-S9UJ'
    '{^RQ({_-DjCa6kG`Z16C{yuN9<UcJ`2M>qaD<pe&WPmR{iOLLNcHCyEDHW-$`T5!3cF}+Mr{911^S}P~FaPvMdg*`p<*)z#U;g^v'
    '?4SC5cGIQHeD~B~Wq(?rj32x1yEtkn0-vJ2ac-Dj#glk5s&s!x7fCB4#rt6RMvYBj+TywQY4QK|Z$Ezh!w(6N@HjdS-kmc|$3IZy'
    'W00~fWcY!O701E`q325dom<b<M~TPMPf9!=b)GN9o-Y-jFJ+&f)PBAcfIg}~!k7I;D<`iM9)*`Vytc|DUiF!(Bs#bQrLPF~Y?TLk'
    '#^w^h8cSM$Zb|qO`nL|+x2l#D?A!IEj;)$4l=IWKPY-=M^)G)`o*n~w4IN<X`+EUFNAYAD$q?Oym$$CP*5nOb`<Q#58xGVxyjev@'
    'FvMAUA<UKS9wyXj?;bHe_nsIg0l4tm^-G#Y!`OV9uJ&#RebSBQbv{WCcmr{bP~UxOIM{jB4DU!OBPyVVsS2($+%udSW=Qp)dml|*'
    '7yc!G;muwk*q^;@X2Hc>{}hMP7%sl&XGctpEZrG_RY|xbnhFW_6lifl<2WPRNH}1wz>}u;?ysWvp2|+$Y)4bh=xeHqn<6)v>T+S6'
    'PY;ooOxdRhKZFtA3lQKm+3QnT^wcEHR>)0e?V->esuk>F8+gQ2-R-$AzZ&t(zJG<uJ_&w2CHN9hfEnsWBc5$(D$a(a+w<FxRxex%'
    'L&r2U4RZ1Dt&Huz4HlRxjznYb3}FRE;lx)rhaw<2`bp53>`N2#b+W5{?(<iRWoNQbt~i`=#Iu7bZL;E0hPLauFTdj8J@E*w=Joev'
    'UrmV;nAma1O_EhV_vKfd11A>EVjfHw>lUl2q9_Z%*2G)9Mg&lx>8T@9C~Mdz?>n{RFo=6zl8nCO&A{N|Y_p2=khgsB5}@Y34-vwW'
    '8`&*Yk)ougPL4H0ggX#&uFoVuKB7gG;SXMV114_&v3l;2N7)d_TuOu9r#!;1ObVpWy^|z)%k8Y$F%>h{b1`x3kLCG)$`zX&)yYtQ'
    '%0vjqjA7EnMn>EnT|7mYj>`OqF4rVCVI;<2*N*w<zst*fRN`S~)<L6U%}}kIGOQ%pIJ3#me7rF*AS?L|OAXgY>VfT791jOmn4R;!'
    'a(9oIS>PYXxYr|bKZXIViETIHQ<<_yo1qc0`!Ql+k$^kmf|<CGhmiW>;2k21D3`A`W8RK^+T%N>43qMVT_#`nd3Wr+F-!bV_Mri|'
    'DJQo0e#$UWC&TrQeCoNMcZ@}M+qRwG7TXYW5yG<59@*HRI{ijh-k0`^m@ew?96aMJ68Gi1vpbBDF$QWwj%>=eWuJBV+<add#N(M7'
    '+wbc;4i5EW#1)D8Fh<N%&K+Z<Jsg?FFyne&vSu6kQnhCE%mucn5pR|XPtSEcx}(Q##$r9#N_Bwb(xwatbnoY$8D{NcFMqDnZ$0A*'
    '$C$PmW$+w*%*PZT{TS>KxQ(eTb~<yUdNgUV@ZqBG8Dp~G6L2?~TFuUL{|-m@=l+Zt`SV-W4M4l2o$<hUlcN(NXhnd}cZa2Wrag@F'
    'mT@aLe>?AW%Wt3S6B3}%wS*tP|Mc7R4}qQ0dikn9^WE{&1j}R$x4hRaqj<)~?X&LwRnD_+St7Gm;DCPGhO_HaKVzn4o#DJYj)zue'
    '{X`HZ5&WkgaGH#s@xvK)^vw4zi+b8HWLeMpz|fa*oihD%zT;|N<3U#OGah6e?`M7Phw0t5dWYBDB=+_h&#vKTJiCUEB+;|PKl8;p'
    'N4(VT1yLxRK-Hf03RCl0$F_4VYuEPNn|>y_>R^SZGI((bPw;kQl>FBDnDqW_oAsnm=jNF^>}I3KKfW`z&;7h_IqCS6U9VDJIVnE-'
    '-IJ85v+>Tc>^`2w8?12y7w{B^a>s@>6=ToU73pKcg_p%0x*&+>44+z*-vU47Z6K5L^>c%<1hM9Z3UjYXU^mGU&Ghu+k=ouHCSfZR'
    'kD@+A5jrE#1lGgW#q<}`08~FWhO0K!4Fz56<*O(bg=M)Zc5jwwqI);()3)clOT8NJoJHHPkwLEHK)j9;O_9Db*dMtf0~eCZGUyf4'
    '+)PISB7;AzBtQe^-fv+lh`tunlT~>s=Iu3cvy!R>m2Y-@U}dLh8;I4n<-1wgI?cf==UM&^^32y13}&>)0Jw-Vv*X&_2`H_T4RFgJ'
    'B2R+(mThJ{?D5<{f@@Rw<>pXw8fl2rtTt0ln>M~D#J22e<J9teovX{RSiKG;?N(lBDfore+jiUv4B8zk2HZb8p+qt~#Jg8xumaAi'
    'G%?ZjUc%5!hECBuYS1VFtW|R}B+#!RRhEJP5KY_08snoh$sW^8ry@KF<g@tT@GiF$$!ST}wpWus4a0jW2}a^F3^~Fq@f0SLxtGKb'
    'ISL+{M;=j6e*4CQXnym%mwQd}HZ;m14rOod$|&d=?A7xbK(p>NtrDb$c$Bre%T-zlz4hc+Hx8b+z8cx_r8Sj0vzv5T;Y;pGx4yk<'
    'Ha(5%$;al1-;c-(q}2BvJ-SW3>e2C<=lqFdRRM^~=k6gwg^IN63#lku_M>Qrqt`VmHB=t~I0_Lwi~P*odUWo3^@RZNyjP0cKx)#S'
    '&m-`)6@^y1psuq0EX)>F%X=|>3T&g^-y-m#Uy<5@vRbtU0R<TZVih_VAFxI`rm2jjIAf<QQIPT{8l3^>i~#*S7)o^`8k<Ei1la^@'
    'm*GH^P!zrh9o|I}%|?u-7_e<1pWa2{PGC=q44s|RB4R>}R6eYc%zpWrI6Y2fx428jy3y&=2G;zx71z_rDqDW35B9R0j5CIY$fsuh'
    'M-q?#k(A^yng4N#-%x$x;?S0Pe6h0!N_tK4rLxz`BjsCMhUjqn;~C&vNF14d;B#NB*!dP0fzKqnUGdr{9g0z%pXr5HQRBj+YnTAo'
    '-F-Uhy3;$<d;5*`D^wl-R;IQCn<o^zJVN1k<m|d|=BwJW<qgHGIq7)IGomg6rp5^<GURkp6r%?=9mBZ?UZ^Jske^<G4xnNa4C$o$'
    'Z$c$qSetohgm*S!mR`b8OcY`Pj$>ScW!@YkU3xZHVIQ(t^lii@!=$r6Z&q|$zNfL4z0ZBjgiUnIOu|iajlH2>y&q(+L|8~`a{wap'
    'CXsz8BfBdp*97=Z8HO1Hyrq!kEs^kDGo&|rZ^x+FdgGWro1;W(#u!R{02taf_l^h)OUgDnLq_~=nVXbsX$L-NOYyW|H+NQ(U6UJ$'
    'R2>>!r_{2H@9IdwB~<UHH>yKs@msz(3&d8b%lndviG*J<q9n8MWl2QgIIV(Mv$G<u{j3*j3uu<Gbet&I600N?XXX>g!|j`cMidz('
    'bth;_BWMe^Mq}+-E5Pgx3;TYql^V$RQaBG|b}1~UJH@y+k=~VQ2)VoQdH`C>_&=0KYglti<+P7bd&5Y5N_ci$)Zii=iSQvzXU*^R'
    'W_4@E?}L-5g9=~yf>lpbO_Xl6dS>Q=TEX63x1$q$EjyJXc4Y=zvd8I3aHAc~v&D+0Cv{cFf#j)<9BNU2jh$qvX>tjSbtKCCigpP}'
    '7wd8Zv9!#kCDW7fLi)nOBIp>u>+<V|z}3<rT<t*nmOB!0`MG^_l!CW&A2+57d2rfIvY0yHpQ%EQgz4l5clPM1eojivU(Rw9Yqaac'
    '9kQb1(*yz1A!_if)^U_fmrih9v%B*n5Ws=A11Hi!cS^j`tuqc+8!#djirw@l7_kmYaNf2Qyy~re;WaG+HJ4dBIaqq??AX|3Xz&b5'
    '!?vhjj1wJAZK~>6(w<rSts5Kd153aq^G3#0@nWj~oOk2I-U+_VYU|PHtv=1)-uA}3rOsb6gr^I$eU*4D%F;R+@rq7PKIa-e#BiMI'
    ';&3va#aehbK62;qq0VuKkzkrht=PhqP{4?4afceyW1E^KinVpXYXfI`?!!6Qy9(%KBuYG7pX0*wt9oGN3+kF9QCZa#lUaLHAk97<'
    'I!4x0qGlaq-I{5OJE)E`*Hb?CXpS%rp}2uN2;4!)pf{z%d~;kn_(l94Mz#EIqdF2ViH%#n$8f$eb%*ZSW|nuoGqR06`i%xVXIBP}'
    'gQDYVs8fV3I-IMFuwk}4p6{U}7@IWpDN|Pz8sUx9F1i&x93v+0fNsCvwplxwmSQh<wuDokZPwyOsEldau?>a>F1GK89g19>dGBys'
    '4&Vmb^CMaDI5+_rbWXA(D+tatC^?<+rto&ikj>!joV^Axg3il~tVp<1uIyF_kh#9tmb#y7*T~M)w+IL2BbNd(3Wm80?Wg3mcFrz+'
    'X;m2?qXjl{*{W+q>QiO_Qy0zdH}lSwH9|5)p(o#n!7xNt@bN~t1Q4<+E$g3xTnTWwHrbZ$I!<>osJpM~S;cN@X_doU3*Dh?>&Lmz'
    '(3mK8T$ep&AT>ApitQaXhP0Gn)-yoH4ktnf_}ug1MRd3f$4;Zfh*gMFlMgE6?ieCly_euZq(gXqtDaUqV>w@68ipQf1_gWEaF=Jr'
    'I~!zAu9Lgp>H*hiwFtQB<Rh5dr9ITn5kp`O6)f}7ES*DphtXV3SgmH^J>6`p>Bs^6J+)~ruk~18?bv33=uEHD8DcWmebj540B#Pl'
    'HqwM1Q|j7Kz{aOhsoV|X*n|t2yf!0!dnMh?wlo+;nkj365Q*xWJ-|O-#{9cB!%%=rhV%V;kbX^R+mZW5M0eYgE5b(hb{e^p$%R4n'
    'C=|C{P9~S8kb1vn(=}6%BHJUxDOpiGjYCwo&2&3{@FoW=gSv@$!gsA^Z^K)ywzJ-+Hg!I=;|OY9ZC}yeeh+<O%hDA*d=n4V6ieo{'
    'Smm3@&6Zcib!IJ>_<C9;6~+VUG8-aoAwsBJaAS{GCb*e0W~=mzrBv(l3;Z*_QB|Ngxe_HFqE=1149z)TOBr$VAsGxsw-ivE;wV;v'
    'oJz578bE~fOL3HlXVmv3-*9{0O*)T}`b;Qm;VN$Id5tZmnSqBR!1THnN)WoK+U@2*bW;SzX>${AX?HS^FeaM(EGHeMBwM51qA_{6'
    '9hQ=)O%pKa*xaWe`+okDf6fu?=lPirZ}s$UmD#rl=UnHSxPDyn&7}aZOq0-q1*%=5PiYzl+jdJr6GmZ};>~P<;m=f3p^<Q)IyP?4'
    'WyB_K$7u^c<n>%L_aaP^=2;xc`W7__GYKPEZ+68F=(zRJg6m=}VYf8*FrxHbn5MM=QwB)rxR<uatj4EATastNHS;Q!^}0%ODDRbw'
    '3kkSa=3w4U(f5@c#eM!N73U10&f!!ySMb*Dpn8VOl8k~`Bj}A~J|eV*TitEop=PH@k|$<z3YNX<DakR-7O1p=DC9e<l#KAE@m4hf'
    'm|*bh$%cKHY_I3H)xWOi+Fb*Jb(-EPqAQ-Ho0D$3cNU%&VT1-G_>}gD41vC2H4RKrR=J?DsjF7ta(H{@T*7r-B4>;ALRNG)0##48'
    '%P@dk01AzRXX+C?dU+ycDOEAv9U@7-?LMPZJr&z0e=eR{eE>=}=oh&08@X>Oh2XY$uW>vHcRXm4uE*Kd^Yuk{%A&6~;8spgwFbtx'
    'tVq)4dxE@YPUR)=+FXmRQcH;J_2}I}kXkCwFMtGf(#lm#(Fu5W-NVd)t|%XU+O^jtp4CPYCTxAYhb5J1w7s#f-lr-`_2^ycles-x'
    'm3mZEG%If6;GN~c$U;}Zj-!lr$=!4$NeA3?E^|$n(^ldO?<#3Pw+>TDH95stVlmHOiL>?^2{N3pjG?NKU@5^AcNVQ1=ajxyV>S5&'
    'iWD6ryF{0|b?>5AcX}VcK)i$3>KxJ~uGB{@tK%tQ!rtYG_jT23$o14~eb+dxY1KTYn$e8zs+Yg^{`_Kit;fu}tLFgv1sJ`j&s!#}'
    'Tlz`QR*WW%!01&L4agos&rUYtowHwL0=bX+uAIvN!Fi^mhj1eF9@t#Tc~BZub!P9J7MTmc@VZK7o)Wqwo$CC7LEkFQTEeK5J6zAR'
    '8rb-URAa1!OnR3vlk*4bWL@TY?iLRBKBRc#K5zh{(W8mUnxk5+RN7s#J^gyPqUA5|^7c*DYF7=rcvpf697`Xf<?&QbX*lT(rIk{W'
    'r4yrEY&o_?=Rb*^OrODfh%(Q8L64(qn`caPV(;+i^(A*6BTO{97A>zTpPooM(dgam1>cetNuqmIVMXLiNOi8w`dky6FO*0ozOEdP'
    '^m9FjDJk+)WUkv|AA1zQ$1YSKQX`LFm7Un5U{#O%60&$H=0nY(f3VEuh*TX~;>dAHV(&qbyx66vWBB5!58R^YqWpq5IYX+-QdcCN'
    'B|crHmF18LT)<JRg12GbffuLZc^WJQH%s485tDp^!CBmR@xXn4U*Z<3f|n)8fczH!fp0!Ana*FQ?+eO+FSx}LI3{&FTrJSmVV)!1'
    'IlBE4&2+9!HPgAzm^ct~HiFS5Eu1yj6Tz+`K<-OCfv{3hMb|$4%&|pQL(w_P3k*5)x9*eP&wDuPjN~t8g;+1Yk%%$B2XchV9v2mc'
    'm<<0J9K`Y!mS2J1aQqRf9I6rdj0hCz{K4W;wcQ;t*IY~JAtElfM}2&`hFb@nBdM}*9(jX@%15l2Ly1D-w!{4r|J9&lrPv)i62#oM'
    'ODKhhyQ%IDe_L%IEvR`K^t_B0qM|oh+vl&~8@4MeZXo87Dvv|%d=~S8JHcrDki)n7H1<Z_F<apt`U=zP7IR{r3K8nTCo?)fIEdY%'
    'vxxM%=cUwbb5*B|R<S#aGTBKA0$Vq<1Szpg;tLm_r`FSuSKQHLO3t3gwRSraH6fUiI@ua7kl{gxEm~Av<B&e4M;S|)h+?&JSgE4v'
    'D2WkS$66oo3N=TpH22t(qC>Q91_6(II(AJ#qSq9^AsJI-ie&fWFzfSI_kA36=Nh%BM!Be&gb`6ZlF8rx;*rZ0!6sGnP|Bu?42-rU'
    'xHJ`%swK&r#j2^YHG&kp$y*c~cn1YHR@l3T9RK%Fau(Y{jLofDCf7(>Y7eI7w7wj@0v(r<7>nXDMQTWQ9e1V4QeuMa4PBj~B6oe+'
    'AtiUsF{x@aCBkTFn}^7JG{qW3E@{KHJGdx?0EU#?<aQ2I%#_Y;rd%;5wG>2b%F@jlV%+<jNrpCq_|K+Rpe?u$oBZHa@0q8~B&vzQ'
    'GL=Wvn2DO=%1%j9C9z<<8$_3RUk(>yfNqhbUEv`yV5ZbHOP)2_saYaT`@k8{w7ns`Y5nM(iHg@+A0C3`P14VBiN50qaQR&43_ae}'
    'BxCk;_1IjWmCRyJlU&!Xd{vM7E4V9<=At_l1sfxW;QJ}Dt{DcrofRP<Xy><JTSPMI&=Awrn&Y{Y=bD58>{ceF0<RN)=^@yazY(Pv'
    'Q<4KlnBn(aYlo;{gts$}c#gxkw{@ryuNX$MUAZC@<PM=HK{tKx0WFOcIvsYBwnAnz&cTzm2xH<4%E}aXep^ZKA3B6`O@~l_+H8F~'
    '<A!gEA7nb;_6k?jm>wfRQzki<aX(F+rBCNPDSBdx^_C4CIACLjxCHmNr1CL}6UY@ED{5iiSw-V;W@YW<8@e~f`Mos!t8eZBf!?@h'
    'a2vzWH2EH5=`LqKPq?f1Ag|+Ar=R0$$MM!NPR2Gg!<#ShCBeFtgt6R-J6YL7H>pTsTXoxJ4mEdngO$U;SdXVyq;?nI%Zd5aXIR30'
    'A2(^Ur5*;9U#zC>)$?w+@5Fd_@=1rJwKKh;{<-bLw~h_M2k3MJ+`^cv%0im#aqI}q)xlZ(T;o`^gFEM%JKe1rI?j7++hQWg7Q<d*'
    'kPZM?UCW3(Tg^-RbX@ieA20TX#_X`feY6?MK`A}Q)*$WLDqLfPxp=s?3vp!G)P6C3G=xJso;6Lo)<~V1>?4Cq<f22g^kDEuc5ybo'
    ';N}6Tiu6o}`2|_XL}Xga@|!=Odn`Ymc9@;b`n-Ll)&n1O#?o4WUgwSQu8y6_bj8kbf}LxYW(z=jUV&}Tr}Tc%)coZ=CLVPsqG3QG'
    '=JRs-BDM0$R?ctT@<|IR52StK7Z|vyX1NYDe`;26Dw;(@lW=fxY77IGycQu-rxk`~py#~Sj0r~gRyeT=o}^z-$b}7~P6Ac$Yc<hQ'
    'A;=SNz*N%_x8`I@PMM^5nyBn)o@lWG@`O}5#alaxrp;n3l^`&O<tg{nW%N}BE1%{yf6?d->8{8`VdzbF2veG>UyD4K>iD}??sLTY'
    '48OT;9{8%}owbD6aX%~Y)W%dS6<4fl@zfq-T??+81vF@Hvwli!Bwo<02cpR2$~jbgX|1a1DT@W(h;csSB?g&(QB2ZwSuM+9oVc}T'
    '2srengJ$Ko12~Ms!bM?d1_X25HQrZj!g=|$+g52`(H8S3NCK+Kw(#TCSYJWMlfXM~Ez3oT>XWErA2j)>p5D+RV>g%}%A8Ee^&tOc'
    'W_2`A*ss8QK(aA_cg~U3jQpfFo|3gplCrX{ovbcbuE};hfTzysH}XE)!nbsa`(;?_N3y*9T0D1}<-u^At^#~F_eM@~bGgzGx5pFh'
    '8awBc4>X6T7$&jncuMT&+q9+S1Ox5hT31YvTUjTkb_8x=)8kxeOu!mjjDw$S`PVxT0CQ6&amgv*K*BfS5=g$9RWOMJ&zk|lAe22$'
    'V1mh16Q|hh0v&iG%(BV$>DS8C?M@ATi*w%ego%NcDT&W=V$jd-dO)rj2^tfYkBRqka!{Pzm4?yI(R>zbX5Ao?IXRk7+<a4bLy?A#'
    'Va7C=L|~>Ab|%Q-+nlbQ9Gs>D4yVM-RO7iA6O&D8;#;5iO6PVh;bx*46W7}0KtAE7H|;Rb@WT@|nwU{peMl+F+JrCHT<b|%Ko%`_'
    'p{!;@edf#2<XbZS7<!f*KK?>EYNA?ZeN7I=xevdn*$Ym(=@dHGW<VyBUCo4)>B3{`oy!tnl<A=1P_4?c8R$!J5YJfb&a;R@VyfZa'
    '71cCxz%JaRxpv#W_n=uU`ljG4=KjRly5K6aCzp#k4#@>F(UhXaBj3xCC9Z)Zb2ngtTC9mP7^*e%WfuMecd$jusVn+|h|H0_@FYxp'
    '<_jCn9dU7PoO1s|*EGex-GXJvJiOgaK~o7a!neLSOHLeC3pdpSIli1J!F@5ti}g89^V^;Uc@Zjemt40?BpHV|m))oq(l2g&=^?(`'
    '!cv`d)2f+U;=ZnYqlJIh6rP2RG;ugDHg!{T^0j#VMOv?uSOl1mrA5b145f7mfl_MwS%#NO0a3hF+`P~gzx!DL7VGZBtX<tjhA-WP'
    'RDon8b!L(*zc?1S1T7--=bCA~AnkC9xyUT2rzY5~)y)@vuQaRs-1`n*Z!)BoS-E_kg>!kr>a}nrPcSx5ef4B^`V5nCZK=yOC+3A)'
    'WD;3kL~bWu#5G*V?ix(sBrdt!(99|wZYPYK;kHAoz#_W%F6VUXPwN4IEbzc?Ps^-hJdlgjp4sdJp=*o-w>!*u@s7zdb%x?rT5j)T'
    ')GTc-PS(rKpL6$`E{tL&hb5#e=AGqrZ9#T?Rh(y&oU`px&qBL7|4v*C+r=Ch_r|e0>DVp6jC1nlOk$#x$?4}BYYC`>eOsS&n0udV'
    'RnQ)_IJY26-2jzL2o~+?g10%@OnggQARn>R-TW&E;wEcop)HI=Tp&tv=337+lA5oRYP`GY&7yJL2o^f>j8ifa_OL$ws<Fe(o48u;'
    '<4BuUO|Q{r`_!7jftw(<H7hvL@7<3vv7r~vZ`03UaG@cdM2t5M(^ImaH}^5xZT*%?vq}zH0;k*@Jtoa1-@4}FyJ#A>i0|wf5KLVx'
    'B~yXD%RNeVw9EBD61eNK=_+nqZZJDn=_=9e#f?Bs6+&WjiaIwRTZO#K#SpXRUF_p2HZ2+Qa4wL1rVs;y-`R6cuDwhymz!>q;0d;$'
    'gn679Q>y#R?{$h+b_G#ipCVs>P9f~hh_u(kOx_~t@~XRdOo_6{FkxP3Jkj>ksl(p4Q}HWK`SvcNL{E_^o*KZI&mp2-D_uIhi*xYA'
    'K7JSX$;l_*U?#!6s9%zw6@1{P^pml$&x6&znKbWx^}b>iltZkVtA5y;^atleZWx&;MMe|HUy;3Us{bkFiQ}o1;B!@Mx$T|Y%O~VB'
    '6{%mG*+t$GGb7(c&1KU|GU21Wf*pa&r^qxjyNMYrzmsT3k(qGHXIMN-G4YB-xlaw6ufsOnUcG%9_cj~Jkdth%xKH=UznIQ%Ww5VL'
    'm0ux2W;vdwcDM@jHHj*?g!5TAFv#A36MeSiey~<F$If!wlGMN>1$)IIaY!+Hj(H+ru0#kkQdiEg?;`fD5^fo^5zN7B?&izrtJ$>}'
    'ednyl)B0h)xbWc2TN()52<H*n^mkt9KD#1sF|tL!-`Jz!RCdShkTyu~o2%FM5e`}<jBrU&!#unhMaqqq{$-jfdsj4pn}bXH)Q)0&'
    'a>G=L=bRW|Dei$qpW(@<Dbi`qJA+_(9&y09^KZYr#>eO<)yG}?3@(WY<v}`ZA-6{aTjmlZT@4AAIjwVp-T-UR_!${dUQG>{Ni&n7'
    '<_uCwjF>#4Kc6M@!kn`r==?Otd!3uN78FU9rP*g>@0AwYC<<4aM3<RTSK#9?V~ZrePOg<l|98=ELDrhY_VQ&EMG8-s;0~PTy{SPs'
    'qrBvYt>*t}vZd%j#Ez2EcUH5@JUW;nE9boKm3J{W6t7;Nd$CGyg?{}kj^Aqc5lVX(ZJ89M0n?YTpw`nn^5jh!dh7Yd-s^`oi}7}p'
    'm``WAfMe&dw#?A4*4S=El1Rx!qMJL9u&Vu=d_^f^+M~3r85r&3%-9-ZGkVeU>0coR=LQemPCX^oWl<nmb3gBvFXS707<V4a2J{qJ'
    'e;r13<y)w+?HSDEN5Gs}AB=))n<pPsZ%*X4rShT;`i!i>6ae!Ii6qJ|z{DzXnLNcZKQ~-frn~HVG<auiQRek{V3w$-g6NXp<LhT7'
    'C${t8BH_F)tm1&l)kukZuHE&>j?WTZa?f`WXXX)#OG%OB*|x}xG?mDz+w);DH@CMO-bnPDpCUJ0COmtx2^SaVxe2DFD0n$Zrz|SN'
    'dgLl1cxI@!;`)>F3-IMEreYHem_;I69Tq0L-8Dy3^NydB*Skpf73+1~DJl|Z21{spYokgOteo_eC|FtaS=2G;_wm2Ii?l2?csMC0'
    'wR<`--SgJ36iTycBhBmfW64%_FMoE+imSum&b{?%1HdvJ9_5ZNz|`jDK~t{WrN>6*V|`pH=4;C0yK^w#GAdv3p;D}85ZpUXJo-!y'
    'r>fk5_H32^_HtD&*1$ZLwD~5oJ|$+ox2M=sY;iSk?{$WKyL9r>2GOONB8nGgT{rGM&CL~bn(ek-*?GNpeTs4OV$P_DsPni;8+&TF'
    '`|<P^>9}<wmlnUI(iXB9Re6937MQ2=F6yD;TNr@T(N&D&<>gz8YlX@@tK<d^PLqO#aC(}57kzmqLmPTawpaD+?*3^*$<_W8yB-)x'
    'v14|SVJx0lf5CHuRM{3&FEGGa>>WLGj`haLQYQqj$v9(XiKfjUka0i{TM;dq@74(IoPOBuB79gukW;o?a~&s<`(hfUKC7a7qI@z4'
    'ZE%B17FsqATA}{alyox-GM!Am(1%~tXw1AY6(lpDm$)f;uqL*uxMr#9ntNYm?=~5><5-@gX&~{TuSh8^pB5>teZ_whm8>bRUtg{z'
    '(UnTjK%uHzSGwYK1`w2(()PThT)uQiMZRhuk5dt)H9?<7VruOD)G)SEVPcWCOvMRlMP2C<g`)V-eVo`7Tl~dEW<-O0C9$)pD-{ED'
    'eH^JKV-c`L&(F^cT*;|yP<(&%jYJAX8p!|$>Phoy&MjC-ttHH&5%n*#sEFsh*Htk@iYvALc?-s*Q7%%;Yk)^ZxB2<mH0+0kC-mjq'
    '3sTrXT@iFMBHOnEagx*+td>G&p$Lk)kW4^w*qvfDjtM!++Mr)BkC-3K6IX>RNehh)qlW1T^lX)njXo|pnTY`I&@+n@+Ug<O4}8r;'
    'D^i(+GiJ!o5mXcRPC;4p&YZ?5l~G>CT5O~NMZd)~RvB%i55vk&#4B*E+zD$!E!IPU13OF)dIDk@p>aNfcsQ_?_k(8-jn2DRR@ib7'
    'nYPH!Fu&v;f*z*@fblL8RBXQ66msGR80+9hiU-?s!w}1%D0UANPbH`>)`LfM<Fz*O+-pbG{@gq6gBmG%PX_C8@l=0m07WFRQPErQ'
    'yb%4{)T}bP;&D=kxWE&NchEP`pY2c=!JI~A%-BhOkYK~2zg_MjDAc4gve*nY-xhhW-6y)ivr}k(JQ_{I+*-wR3F7g~i|V`k{QRt2'
    'FC$f}c!K8`pgp+C&oVR^T%IHniSpbM(w|U)J4I9?m<*e7+cIYJ@JKMBd>C$)Q!Gxkqf#8HfqQKv+xXz9Px1XUEz;1<Nfa&t;;0TT'
    '66;4zS2K1rg(thnQ0;t9f0oZpWZ0wJoZ>|Nuy|)iwI6V14VG~R0a=p1CXt{auZCXGAw|5-)ZrF7U3au;CpmV*%||cvP{4I2h0{@j'
    'v8R-2(R47Kvm|`0ndGs_mA7~ReTpW0+SNiy<fNTo)Q6U?e)<r_8lo7NUlSCqxtYbFLeH{t2@2j(O%tc<f=7&8x`;t;!*CZFJFr|y'
    '*%_1Um@Jtsa9k+sxylO#M&Ldqgs(@FQ7V_M_^dgQvV#mKnMCZ;L!BZ{_7O!qkwlmK5KB&X!5i4Wr`{GQvU>nnX_rCQ8DYsq`Y_W%'
    'i}<#uNKSx@<7i*aA;uzJxd`g$*%oi#f}T7@H!6m?I<oJLY9EveSFgivI;#1j_OIys6{$d;rflmVtk{*jPkON8qJtAupGQiN%+i^w'
    'puZJzka|Y_$WaB^*d->0tD2XbWHm`0@qOBpvH<7}9VR;H0v#Mclq)Twq`Q8^!BJDRJM{vZ&vr}3v`*h)aws<AbMKXU&1<!h0FZKn'
    'i3vk`?JA1CrTeVRcN0+`bvZ!px|oMt1-TcOfzbStJmsNu1-Tu)Yk@GSy6SM}YAw&baB4=lvBNIK4S<@IxiD3u*@hy(DMg45+_RGH'
    '+@c$q^lpu(U6a)vlgea@l6+&`0FJA!SIseKrbbChwERPk;-@$t{-F-ksH>#V1=xGw@1pMYqI44&&Q)_fR1OV_O!3%2#x&~=fkJK&'
    'Ql{cAuzpYD-P{JPZ3TwS-bGgQPUdcOJf(SucIP_E<{eLTk0Cou`nBn#B8K3_ZyG)HllB~iQS)eeL_qGRW%ALdy{7Ho3IHctUvs=G'
    '$Q40BU}RlI5#mP~XT6(Vt)i@?DJs`=5@EY)DQ0Due1Y~UlR;fUsL47QDDHKV<=*>Dy^g29&vhTg<Z~4x-Moxqo<^7Y3l42&VHA+p'
    'w;o~M<(!wsnB#AcYO~)sy2#rs_44T4_M>IWYwrHo)%^%yxBe_lMM?k{XaO#A|2WA;w7O58zAKl@>rvV;O~TnNfk0L{x1_9GtoC{^'
    'G$^$7iw1>4+r{jN;n8Ri&Kx>3Mpz46+Cq82@Ejy=sft>({<4kBEU7-~Q*;>=aPkzr*Pl(dW*2jBP$fjzX=Fl}D5Iu1*z2iR21rlB'
    '<JM@fqug%OGC}&DO-JYsvSeZ@lsp2ozx-swv;ci8BfE4~a6Z6Oq)S|avh(@@JUd<ck)u^ywz##8%QaPC;br$-?wc!vU_GCO$f=|Y'
    'Q2-w21UUDp*99cx^tr(UsdD3|IhXVaz8hQddE?X@>0t+f64mTyBhyrnmB)2C5JuJqR*m7wU6pNzU~wkJ$m!&2$H?ize6xVSkP}{g'
    'bFGyB&Ez$k=iRZ6RF?|TQ||L}xcFA@V%^CzqR7sxP^jc=@^YQ4?Wv(}#g?g8UI(<C+^~m5rjsi;=ai>N0-YRQU49Qayt;%*?&vP`'
    '5-oF{zn&Xn9=IaLCU8xSP2jzt3H!1qWy*@ee6>0D1FLZNP~0T~S-x>Gi|UG};d*$`adey_WBarrm8rWb`I78nl4-<Rk>RRu)zKMO'
    'E<W#CefhXJngkg8!@Az}tqcLS_;-NibTzY8%gt(}sBM1Mc*A@E)lTwR+LE_j4XYq{nVZra_I<QZ$LA{W<sJ1WIo@?+xo4fCZ`^bX'
    't$EhTCiM$hB*i@a?r~jmhEuOwlCRl@BT7p9SfsTcKUd%JpZON`I&Q2}gF8+h-0UbuYP)TNS8r3}SSL+iKR>GuMBQ%HUgvY0_uuuG'
    'H2E(SOl~stNgc0KvMX_H%5mof@q3PB2gcdir*g6~^l8@V?CewF+2uYY=RVMZuJ+h1|JjH~)*UiRXLMh9J*Sv$$uS{)H-EC2dLc8('
    'yJ$uxZi{<J;G;vE6bI)Nj%U&{t)bTuZXSwk<EwqHN~GDP&Piovj*ckh)})vgC5Z&A7gOuSxSxA9v~FNN-P7$oB`;a}wta*Is93i0'
    'Ar+HFdQ+;`cWVlMxObmoFJU<12JcnVfUb_Z?Ocbhr<nWFl?b?}^p7+0!8J@g6c0U5G5<xjCq96djUOO9PUZN_WniH=pGTzV3*~3g'
    'DsB(&WIn8B(!EUmT~Pv|`)Yk|(5G<ApVHze*Q9ci$-!jgwP|+T8FJ&WQ{fUXhtIN=Fk;@ixo7Bu-bIrJ=+tIvu)(}0sf{uWO_m&E'
    'RLv~sxX<O6aql*BtjGDy5w^Q`H&?EjdO7(hy_N?X32Jigl<|3*#%Y?a8QNULO5RO^p9t|a8Pjr=Wb~4IYdHJ3VT4dGKMO-@yX06s'
    '9koKX@-Ehg#x)uL%g9RjU<@m%*ESO^f%tDK+amXeD2|&<3s<KS7KA87OY{<UUFK%%tDa#W!A@nlRFP#Co-7}x6KOtD-vV-o4CxKa'
    'Rl7e^uNfY`fow(D^k!32DwVj?I2vrXG?)f0%xXOR$Cy6NsK))#hdqqB9!qWo<<HgI+#s*Qj-vp_El0^+5+{>WMRr|<MjWY%&dbtz'
    'M;_LfFX1S;A9h`RHpc2?Gx_3nj*cJsODl9ipy&wED09Cq6}Z5^UCPp5+&2!J&bsCyC0^1B4eygqMr5L?OQnU^uCX?*E4t%c>n#aO'
    '=eO$$4)W8MNH>V-bnsr@A?M`uR}asgHE|+Xmn^?92dZL=(YL}b3g+*OM1ISV%Uj)7M(0x57RdZkdli4}QeK{1tQD?dk(}iUq0o}6'
    'g0S_Hp#s<Va$HY;D@%uODo4=vw?}b2&z-g<s%Y2gn>7v!G(3u#Q}PA&@s!30chU-pYPG-Ww@+V1a7(z%kj=lYK;?$`3#bA-|4efj'
    'Srlg1<`wUQ$tzx^dqncLSo?4(&X}D|aSeTBqRJ%?6Go}yM=W`d9V8H_3vZEmOLV+KI))ls1Z(((76aiMW@k4p+RWDX8SJA5NVRLp'
    'Xj0X|41Q%udNK}JfRmHq;Z<NQZi3)DSox+%1l>5IC8it#Eznuhu7r<t^E7~VXaP*AUVESm?CDnt_MtanNtOg+-2E^L-ty&%U+O2h'
    'ENxK)<$egtMBj8#DKd>QWAf%bxe>>Lm`$Ds0aJRO()SoEBIbE>!Lbtg_Tczy{7brByCB*wFT)a_-{QX#ci+{Y=`t#O7#^osFE)3f'
    '>Lk*31KskLkg_^yqy~7{`!AoSPqYZee8vy}QV(OfRsxHwIku@rGlKFbViOo@Z6OX6YNxkSRGy)!X*!GG#UV?-z|dumhZzDTx%Yw%'
    'N1&Q>N6keX;K?!>CudF^;b2~;1<&lsD%3^Lz`Qbo^10$)6M52uJCilT_&P4EQ4WyFL)*kz-$Cn{X4M7J;C`3O!}JRY?8JbaYq$I!'
    'HGuy2U$2CSdlDzGqy=6QhV8=$XpvK6Ck>*T;c`9RoGHDXaAs++h1vR7K{}G@ncI{5no#_4Mx!Ko;v(Gn1#(*pc-eDpnvBw7_zL`&'
    '#bkz(&*#qeFkNb=nvx;0I3Oy}a$RhpYy53uhm4|zl`jp~kvMO-MdNIgVYA%NFPNo5pF9`}fR<5D4Add#<G0Ms{MCfhB%;X_UV{ng'
    'LO<n-EY=(*Kd#O7lr*aPN=U!sC=h5N6V9w!YJO>26-bw^#dT6{dVNqvT_~kvLqw?*(!#xD6E;7`F=R_?oTTFJ<i9y+vM%aQo?@@v'
    'nwaxpP|aFY_3U0jV|_;UQK))9ae?dD!(9}>I9B{(^WChg?$Mjr?~=NhL@amkJ(Is6L9@FpCD8f!Q}&eA763J&JonChTOUwwC<ELb'
    'cgc<k?Q!?F@23^_Ng&tNy%+-NYO6*KJbiI4+;osO$*%7Ak};1GduEd&vZSUqmx5(ybHrLec{X5+MR;g)E?c->Hlury81m_tqy9B!'
    'Y=eyxb4eKOY{LkN<`-MvCRuFa0obmT(o+i{m9pA>O_J-?o_+wKwxFikGA3KINj*k?E^!wGR*vcyY}ovB;#=6H(M}F7!Ls8Q*lwuQ'
    'wQ*}qT&7!~E%<~@KK8`=2@al6{DuX<Xyt}FEZ5v5u!!uExn<X8a9WF$wcD%!KpWUC+uSq&0Gp%L#B<>Xn8eIC(b2S-U0P|9EiEqo'
    'D!I4kVdg#t*Y27&d@@26npvOEa3s;pO9jCP>v~C=-@BNMKd`$sDPiNVxf^Q-j_Z3=vGO)w=64oUPc5g2iGHyIpL$J^6QXgsbuZED'
    'a`y6J0hzZ)txmfj#rsRG?lv#YqEI$c#X$UiRPfvmb5#`Ha$L}1(@)0u!PIoD7zNJ|Ao9&vsn0`&5ifD_ZpB!>ki~O@;P}gju4EjH'
    'e10++D4sloD+A!&s_UfKQS49G*;c(kSIOWDOFbPL!g^LZojjLmb2~52Ds_rwWyE}n-b;IRKV8*9lcM9yoCVWVj4FNxK*mF38qF?d'
    'O3Zc7ftHmNn}u_*0QrE{+5yvhP&N~V;!e^`&eviY-o<&6TS}|rf#d7Mnq7ZDGI6}x&9D?A*FRwTCInGEnbw{xpdw4Bk#bySC5GC&'
    '=)G6`WHTmR@sM}!4T@h!nkmtO^v~7qRo}wbta(kF{xEyvY1>biunwESS)`^kQ;7%PvHagxF+PN<s41>WTv2-6<WkP~^IW|hjFGh5'
    ';?xCx3N|#Hcnh0v@zs?lXCOezw&F`VhmW|5^741`s>F*Di<%Mi8K#_in%BaQr!7B=TZxu+A3XgQdc-)#P7KO&pNt}()=yehQ7GH2'
    'p|~9K<z*OD%JFNQonbdjmi3EH?PFYxEp_X)5hvbtzd%%Qy4!Tld{KeUzMoXGlR<PKb-hOW^FG6)XADR1(=&q3o+MZ}-o^5XscENU'
    'Z#}^G`7{zvdSn3J$eZ|58ii@uFhxr5_Ao~Zf1Umq&jVj?oDP`U)rMtUN6hUERz_$S#XF{5urk`M{A?e{;B>U8Xdz%K5Zx4+l4xS}'
    'gCqVt?exBp`j!E_4`VY$U5;G6<CfzcF$oj=J;hPZEbBp(DhMD}J3i_lG&Q9JQs-yI{<tORCR8V@OD}nZA>nFe&k4Oq_7eS-5bbXL'
    'c1=fxJ~JtG+_YMgjhJ{>Q|w8-qf<lbRV_h`Cw6p;U!P^M#Vdb@YS~F2VW@c8N1PWW)Uz|o#%>`GO-HqnKBL}|wXHqAxQ5?W0r;Zi'
    'ZF7z^j=xPng08A2=AmJ&NuA8PvX9c0-hK&GSy!azwGBG<uB=S*v{siG+)6A3tyr1u)jKxMD%QfDdhVUBx@;xQpmM7US;Y(cP}T6W'
    'BVKWCP}Zqw#0%e1_P$?PKBd%js-$>sP?X6=s<mRL7UhBLz7Twi<CQ+XI?D|^37Tq%dPmvFmPLN4v17_FlNX9&QVXS}P;6skcS}zm'
    '()trZxwG>3bC9BVbQp@v#U85iE#h8pRR{eU50c{K5%=T!wuVzwIzVmXXTCj(sVn3lE6Sw@@P3p_%_O&cg{{DHP5!>Br!~hrRq{r_'
    'yRQJ>=B3K$*CO!xX>Cv?Z+|`aw81vrU3RJRaJ>PFy2*bN7<mWa2*0U3Nzfwb43=T=^V>K{#R<p+Uc-q7fwR13A%DKZnD)6Zu7fI0'
    '`4)XLljo3QS^@;3`^&+m$V<?1{LTm$gvnq=vP;H;90d~(s5|@xJ=yu<Q+i3FL)QZExhZk<ecF=6awzK41SKu;K4PK8u80qn)=I)6'
    '(7r2z5MMt#wOl#JR{<E;r`XIJ>7U}x4f9hxKRdOQ&}jtGT)&DV#7#wYhY_Gy5dbVIN}3xZib63Y69fn?jJX}0l0-xl%L8K$Qd&%&'
    'SnE@R2w|}hMW}C{)@vwYz8ZPtxJ{rQN^EP9x@bA4V=DHO5$e@*@7bqz1X#uc(jOxM-+J;@o7qDmaz6Jy`r-Ql-^X}4`6*`lwDlM|'
    'f=v_{KRG8R;g-bO;x|%Y{8aWxW>1eCChcO7EKf_*1CD9G@?NYYNHWxNJb8YWT9AvQ!S#~}t5})d43}*Z<ha2E#dGhM|3UxTX0tfD'
    '*lK7adK4d@s@eCGe(s6OYo;udX105N(IfnnYfsJRIrSt{@2;-)=CrTaHz#h1ODy@T=@U(<Cb|9jZRDz5oWQ~^^JTztF4MgXi|h!9'
    'BhGa0iwt)&4aAgLGu1BL*(YaB{(^aoZiyjAQfu`w%37eeXPVWdL+Gh5wM(*MDqwOQH8}gpWHc?=oV$6*9LBfLFY-=zvPvPzkk5yp'
    'k0*{6(hdGvB-r>|JHI%Q2Vlffd5I!so~c-T=(FmFJ}Z=<YEkDG`Tg-j%7Mgs5vAywWQ}2lxNN6NfM4noyTqsjeSVvsxD=31k^t4U'
    'eYnW<_A2R3pxjmA5Gl!<W$H*+wXVa>m8P8fPHgJx(PE00)fG`Z{anPW>PNi^whEbjpI_wn#9(!FblGpjMjBg5NpA$(0Q+FnpNe(j'
    '6=ffZ`qNX(X%l`s{if<%rKG@lIhu3Ju(2#GdW@f698Y&!@xKWj%$u~tvxE~GyCM#8?#r)a;ZS|9)knQb1`p%Mcj8<%9$X_?T~-hL'
    'hNkzdBe{}Vj%fSv{e_e~z2mXVK3VdnuoJ0Tb{Lee#Nv|LWA8k*q)y^SO97X}0r8^(!;zn*uXr`7T`#0@Bm4frIj#uS*kanl(y$b#'
    'fh<a&V#iIB-hQ5TVyr68D%k-&eb)d-d;%Qq=dVxHbB496hd%E*N-L(w^d!JsEn~!78I|1(X$T6E{I4yE5+!JCX4(+BOnoi-uV(6~'
    'v@*8R8z!*yBXKgKW?4=q%coVDrTCxq0SnwV7Ad~kSfmI-)y@Vh+%l;Lr{I!H8W79EXCyDF9@xS9y`-{}6~*6nW5HW>>BE!rd2V`Y'
    '8oQ;pSxuaSGKb5Wqo7d6<#N0Rr8eT7{{P8(mt<XXBuQ^8C3k#q1n<YN*#nMP&Bu(C9^A?d$qwnLxYdyvC2TH&+aPltu`&Mosloq~'
    '!bt?KA9w%V;ef$lrlvZab(I&rYvh8*=C8`{+&lYiD;Zza%TzddfLP}37&Q33t<1?RGjh7qif$t$pqLZW8CvjJb2v9maT*!l;h66T'
    'oEAL_w>V5eGn>sRK>tMK<JW{fz0c>TEo9xv&=My<f#ctq%$%a*1*~SDW+?)-iM4pHb^pTBlhKl8PT+#p684}vhpx>t`FEpR=zZ-W'
    'I0hwV+$Fvl{YIv-M1CMzpzrysM!a+8c{zthAb3n?=wYotaqT1uPnI(79KRzAgd!IC3NmGy3A>O@md!>#z<+H4O5Qlvq6A}v0$gT<'
    '!d$<Lk^C`|$@pzt`C;37oT1x=Y$3UXmYq}G#oY9Ms<52Cs@|&*x^gRkY_>iVYpU(z*n!3vgFais>TT8$WZ5RG)){r1eDexhXA66d'
    '^#^SK?-RtV*0{t|RyZ|U%JD!MDK~d(mkTEPbqo%8&QmE=TpPAl4iGzH_oZ7pS36%hnH$9f<|CR7@_FMJ85->^?_KP-ksCV*YRU_d'
    'dvyxTXRjG@AjTleQb;u@Hm;S`85PeEwH*D27D>yy$^)M}`*%DWlo!&c#VR4RNsnzbHW5D6;M@ZDAO4{60$~$l<GD-t`~ruqo(-##'
    'hi70GFbE$d2ud(0!=VC(li0MI1$aa?_A3Wl%5&cv0)wT}+jrmv+rh4GnlOi|OkdGW{6HKDUN<+(Raz9%YX-$bBU#i8oP_(!&3S@b'
    '0;HE29D0VUVK~aZgZjh;<Dl-Ihm1YVAdZWqIqYP^rTPRy*|GqDAadQvq7Nd>)Bp$2lHfQ6{DVlaEBsNY1no5)bO+ZJzZO3Z+n14r'
    'WbGSRG?MTgTosgt9ef#JiVjZH<QAyS-#a`zG=vvVgp7x^<tZ8=NY1XyTkMADg#+KqQgkDBbLiM<-`1XU>n+N$A`PM1=htE*eXce@'
    'F{>>BEz{ou0iVlduAQCyEtG}t9{0AG?$#AzZc-0HsUpbc=O^M9QS3yheG>ZIgb^C58ZI|jEeba<7KV1uX0Z**jsZm%mz-UY@dP>K'
    'jd!=kAjJ)oiSYRMhvPY{XUbpPBw8_RC|;^IPR3a~)@cMT$bSILWD5XFp|cW~5rl^Z6)^n!Cn5mQ5xk~!ryGhTz<#$x+<BTt=_4>t'
    ';dw`Fmnxo)A|7=$_hxbLi5?K3eHDNqpscE&Wknzz-61tTKpEfWwoCuML)>tjtARbNha9CrH6*TqouGqUcyXM|+XStz@isEqh+!2#'
    '*W{z3&CRR=NdbdJa@W%YN`If>_MhUrBpXp}1@K;8v+V>-6dV<hJ(U%-h<-np-4{c>ao1bulqtLz=r>X$;1j+yZ`5fJCKpJo5Pt*J'
    'SVlD;dW))r4zm0?_Xd9~a+s+gxS{f)@{Xseqq0*&0p32T)v8DlQXNsP3r3(QZg!(XpjZj|AE1-Oa1YvNW@rH0x_7WSplVi7l<JdD'
    'r8MVlB2$*#%+@`7|3v(vrf(kv{^bE9BUoU9@6PMVICX6Vi<SSFXCV<dIjXRICxWkD>78N)p|23abD{~!xPr7<cTx%teL{?2fcpT9'
    '3|sslh2qT;<pW6n4Hq_u=zqBK0r|Hmqd*Cc)EZK7WG6P-a(=gX&6}&!q;=36UB}x*Fu<v;YKH%KDX*pXvU?Z5Ie(G#o+IOE4R({1'
    'j2b}WZn*(g*>OU7K!p<m`I`SjxaUSzl^tTE!?bS2(+%_#w;4p#clmu^KSMlD1moa-<6{(v998xs?zipOlPJRAU4ss9_IpCPO3k8n'
    '+vd+y;BJ_~StNP7401RqyZQ75j75hf>;AAKF>tn!w<GxqE2EJzBf_8SkYm>)fL|dSdEk!(X8>Jb+{JOK4B`j`Ny+wlmvOrsg9rD3'
    'Dk<Liu;F4i#C8Gi__<3&j38nw#D9pmb(2P{V4a1ORiUUHq^#6}!ze+4I1UiZCZI?Ovx2#T&d1lfRo2JzYyNGrvPAVwDh#P4tTuvr'
    'tq~kM3w|u}02>ULe(_%GJDKeTtcgAz&@<Am)x9-?DH_6{mT#8>Lc7<xkIm_<Z385~S|@iU<E;)1o@crJkQ6G=Wp?Mvb|+HdyO3pP'
    'BXuJCy<va?%~NsB29(0fuG{mw-j!qofY`-CcS1EEBpyf+5E%heOfZjqGkhTwP>r}^90vlIi`HgDl;HmWQ_Kfe<>@7qA5LJiO9#tH'
    '{6GT{1*ZE{JY%(7n<T@mBz)A`$FgqKGT)L?V*@EGR>{+F1xZFh&?<t#ZXty&=_%^r<PBDGcgomAOX5=1kmA?$RP6<oc!km@_%#SG'
    '0=JN#hvq~j6eqJ@EhkrL`RScBQ_`>_C!{?czyPFqy5jJyp`|K0Fcl49ds=7hk;F^CV1J}}y1a3udC(t0a;GCvhce7S+>jMVQZN`d'
    'jiNS?#Fg|7S}hg{Ei2BO=%Yv!3vDDV8coq)h>~2`A=pSmHCIT9CU+oOEmpENScZ;{Nom#cKggbL1cPM2kwa7>TiN#}V{0kYhVqJ@'
    'PrD-)z9#f)N5l_>Tz;+bRFd35rIl9ok|wb$)-FjKRSWuN%UNOC8wqq|jOcqrj_#NF&;OY(_3eNC@%z7i|MlmezJ1jGd8Brrt3O|+'
    'Nk*gTod2UzE2EC9UN@Us?l`HFt?4eTq8;n<e_`13Uu8ylFth?u8m@;b4wT$h@seE6^$K9z>na7P$2!FM9y_SsWp{s-sm8nTuQ4=H'
    'L+$9~S!-6uHVfLi=h4!z6t1qSXsb8w$I`jbBScmg%|0Z0s<j8ZTuYb+k>o}B4}?`kQ(D{UsuA8*Mo<xH)=FE)zW0;8>Pnv1ZcXIZ'
    '{qRgR9x9j#uHX|3rCcioJbT{O>mybm8S-K)XbAZORi)eb3ssh#;Z`duyG(b6r4;?D^>)?bnRQ;lmC#dZ1@~VIPKTZkJz(K}ZcC;|'
    'RVNzywIC9t_f;&cYp+@h8r9O+(?Z6Ctl0{^ORenzTB$z~gKW(fEYe@w^rWvs`(hj_^(>mb)wMn^uf9_2qxHNhs?jxWtKwj2Xk#wX'
    ';H?4;put-Ubu^tfs)m!)a@D#0%#-)UWa;}BB(<d)SXTJj73_zSa}}ps!jaLEqNIItsSq+XRRvI)kOEKDh9tShP@@J)EyZhalxs(!'
    'S7Tgz^sb$Kl|4d(ybc_HjYNzbS13H_yGUc8i|e*peq1vOSthX{RQ%<|h((xtO0|;bYp7}tZz-x3j+qPe{p`2J+^2WZ>nzzOx>|#L'
    'j<{C*YLB(JCgy+n{<rVHeft>YQ=lL9!}zQXg4uz6L8-n{-lM2}We>0IXzlC`WdJ%`PDx>QWN_&cWM*G|U!kHXytI%vij!6acZNb}'
    'br^83A0ENBQ3pvd$$@rLY4{25Gyl&&e*4?+fBF8)w~w?xtN8LH5d0~c2mLaSoyx)*ZxAO^T-8FG6zWh@xRQzL>k@3m!gsHp(z(*H'
    'n}Aug@@NHq-^@@LOsN2439}@ZDM@+<08gTeRUeR0w3h>vFj|7TCzJ#QAWCe0<bNlPQ;<)|T7f!`VIdbRv1Q@cl>Y=plq_*R)zaXS'
    'Wk~D|faI_-Q{*t|#k%Hn(q<L?wlq<1ehsdqylqzsD6s*O1DrI%P;yI(+Jo~<?qKza#$;1SBWZWo4tJ6IoSd$N8lXVfi5^$7Dp|LB'
    'mAF>)x(WS2QF*6{X7j6a)3SCq_d8auOM<AN-Ix|y;6vx3*sW_-y8vD^Y3B!@XVSvblqg851!}r!Iw<vJeGioSx>_q#A(MP)Vs=SE'
    'EI-w@VxGP#y4t<ozYg6`&JlkVNRPteba?uuE7X5F*}uE_=6V+W)4zNCe8WoN>7WY!GPLQSOSyEdbTgIRONyH}fmPB$S+DAMwffWd'
    'Xk^O0Pwa`a`#_p~eh+E(37EA}@0;|M)cbf+8Bk@4=$9^@(r9TS;5eullRoWo@_lDT?w=Xp`B1g6G=0lgQ!0_-hkrFI9iP<{GK;VA'
    'oa;-Q0`W~RnNXsI=JeI9G!IQl3%<U;ni^mF2KTc4ZkBi!eSJjHUri}$@_nl4=bC!q)kIzM^&xcb?fU1kdl$)LDvOlrMvZ;y*+u?('
    '&yN%0eEpz@M@1(+nzAbzojRKDV=1VC4E+1DdzTVo<_XDRR2viC+-=GgjeGa}+$Ao%Uqf=AVz1TL*zYDoo^PaayLgS^DaMD9yB;Sn'
    'wPcX$U`Fl32}~6|$Nh=7N&e->Uw(Z#s^xsXJP($J@_9}c%r+cVdC*`fYva$2rFNPhsmhBQO7c|salj&KC%7ob16nd=N^1i&fon$M'
    '{PkU;HMn+Jki9C0H4stO78pcsrA+(aXdxYMHw=jF^Vh<jY#29MD{)&nk;al%xg|*3YjMUQJGd{%*voy^lIOo>R1x)VddRNPYzN9M'
    '03|qIc?L}QAZ@ScoC=E!@l{m5gIiNjnO`dc-MAQE=AbK+Uk)Po3I!>uep`TE)!Hr2V}`?X$=U^n$sPf2DVy4UMN=iZ;D9lTxA4Fz'
    '4ly!>dPv(BxB>|wo9$6OF1&ZIhRBo!))vMu!pF)bK_QhLeTQ|x1V#~Z)+#n~cjiD}m18@?<=QWf3?r9_BotyP8n+`my8J|lfixU5'
    ';>r&($*ukhAivy@x>B;na8rCv;tZfKm|t21=_aILgfGX%jq81DaVXJcfrUcnN@UV<_mqU7S4r|sC{7}~N;y}8weUw2(-~XjUb(#s'
    '-39PVuy|Zb4{=D1$h2jvS;nzrEP5Oplm!-WkQ4}#5K-)a#0}94NFpB+5?MMYSy1nE3lOlPz}vJWIuGcmdVEs=s;|01<SVAAA!?ZZ'
    '6SJsL@+gkMeB-<d>0QSPP+%?RRXJPeHfn>-Br~Teu*`PRN5z5|s&L{8(N9)IGlr1jRTTG7xK)a?wRS~oVts6f9^3U_e*Wd>fBUze'
    'fBosN-^9cJNt*8E5X;xbbSaFgNhH)`RW`f8Qd9L~Wy56;pVo#kUa_=7omd48E|=%kRE&B<9uJiV=u2O=U~7kvdchc9VHJfY9^QDl'
    'I}!-c#4El7Ulrj2b6L@ZzgdH-LOM=;R5gDc{Qr=o6D+T0mWDANDssMXG>7j?`kBkj0KyL|>i`K2yQxBMyH*CRUZt8n+^@;a+%!NQ'
    'y#j@_;diMmy9Niv1G1ec9za{ZuJ}QLuA`be1APT#0?5rGet;5$WDX!Rx6DC(F59j;qUlm2<P&Iel=7iDp39IU;Grxrzyt!c?<P?G'
    '>faok%ez3W-(c&XG=r_5Uu!V%KW~xy^KSLZJ(PyT02HujpHv83n^huA3s0|icx@PGz=S<^pg-w+3n)$Mc<xZn9Xr-nfa;z6euewc'
    'au3j!`&$%hrz~YLo)19Djs+uNz;mB5FPg`WOSyF4eA?55Y=wRyvk7*glJE3q$+O;}0d^evmYU7OKDfiNB`$;{3f#oRB$NoEDYK{H'
    'N^rK2p{V2}NzqjJSs_uXEJ`HYYo?sU&wx4KsC&RnhS<m<ABm8PQ{#HUXsRa*1Z)s^vYHn(fog^yF!MNun-`kUZSoe-r25@}rZa#{'
    'L)&<Ch>a2NfKQg3du3fC6aQGF$O6ey$Rxj76+t{CPsRttsw$cIfOV?~07VwNh*XEt&EjHDHz`adz2j0%4dmhiPWO$aadZQmisH+L'
    'm1Jn53*@7S;Lt?ihYRpoc<JI2&_+iq)t4=<;)+<*S15(9v{Y|>WmtHiqOCQcf}#)$6})1!{-Bylq{JT(uiw6LPkHQ@m|p0`Fb(*|'
    '^el!7Og+m}fppjtltQ%%aSyD5wOPA|n_1k4nYOGdbSE9XXbYLWexOUleG4_n+e$HH<9G&Mi_g!jgp3fc6CEpEMz4$zm?^~qGpIJL'
    '_3?hqcx$1}kCz)&$h##uHrwV`A3#*cT66rkD(Ut+yzI?kfXY2SenlB600i8oQ20h7!=JfYn`ybJZ;QQZR)V652AHiQrOV9eM%6vW'
    '94vkb3x8~rtG>AHCGQh&*FzgiR(bXqij^|ff<^4dl0>r7*}6$I3o2u3>hoffDn*+3uxyeLY0?QaxdY`9m@9JU`F&q#NGs+kvu<K%'
    '7~++c)*szRd0`Ufc&U~2E>({eOXbcXuk1(f%rJ~&XVx$r@5@JDfN8<$ti#=ast@0p|4=oIOa_t4Vo@1DR91%xaWJ1tGMPJ~s8qb>'
    '^_b(8x?{*TE1CU*2S9I@`tELj8pbTs4WNO?H>IgozUiLA?iQ?Iy?ud`RPkTE<nq?>n*L4x9WNh!!T7W05CRl1w^a+@U}h~))9o(7'
    't3|UT>f@uj;q>Z9Z`SKXAtX}WPj2$#*!M6A<F07c=9_ODE1H^kIN7Q{EEa&nikGo+SL;lv4?oGmLP=l+k>iE?zGl2N;)*;LQi<F{'
    '?msrxEgo-D!D30Lm#t)pht3N#D0R?H*9nMZ4_7!O1+mqF{eag|tq<psz#jLnxr8j#v`bVevx+AGH$yT@e8h2#;fgLa-$_`!$y0>_'
    'DdP>vZ(3B$F?6NBkooY4qBC%j8sI#!Vby32d`69X3gRo<Mc$XK(rht^s2Il-Q1;5;_JTgBE?8ky!01#{`dU*<=y$wI5#(a&x~0<M'
    'ycK_?LJHSV;>Pu=bZuKaK`UsX>)FCRUSUsxO~$I_zfzyZB2f`;WC>9D4(~x#zgH6y<0F=siECMqgSVGYeP66EC{VI0ExU?>-|~Fa'
    'Bl<1?*=Yd+woi{x@NG38d-)<(C{tw-!&Z>hZU0%Y4g2CqvGP~Jj)$G{=UO1k%sq-VW(b=33A7d7{t5#A^HsG4_pfye;)eCz{({1X'
    'oNLp4Ow^pm-(2JVcr@h?ctfgcM`6URgO!piI$SV}Q&FfeiaiJK>ie2YP`*}?6P;e~s>V2Pth6V-`|<u|!BXPQMY}6R$p3lw(XWYx'
    'uoAbovKPx9VaSe_76a>ecj5<$$HIvKa#scKDA1i$LL9aRI2`T*YUAApNDIN8^4#IxP(2N!FbT-wRhz{tjuH~!`R=*Tv&;`W%j`An'
    '0Q0Lg_v_|V_sBi>Ub?rFmfi{59I{9E*5f(hp0Dy6k8;i1oa4cw)#2ymuP0_Y_<xSUyygANhEthcj}$HJlym3nlM7MgB`wFyDVk?+'
    'VHOuZj1Ql;?dAE3xE{|pvo0ST2Cvo&(MvpKNQ1mdz?EUOzL(uK&`-viQqI$??XFd<DORkr*TYB7`8LZ5@AU46Snd2X_9*U8++Sp9'
    'E}w*3%`5EdYxvS)`9#j;R@q(fJ{l!Jg-C5o24wmO4Mr0W5sFU2e-)t<*3sv($4M0D5wm1SOBj>UpO%QIEQ_eJ(`h3uUE>;Y0XxGM'
    'FvSA&X64_w<8}VzOM6xD58hs(q<h9^81X?AAfz)62w%E|I=<LjLeNsFE%YXnAYcD@?svN=BbRfnxxExgrqDl{($i4_A;R5JDi8C9'
    '$JajI!}Lz8BF${CAg|bxRb}}Pi*1~E-A)y4u$*X_^*F(&Bvwpf_d3-wn)ZWHs9@J;_cD8h^{t+MTQ?TuraeOki{8!Mqjw51oaeH~'
    'ytwv&*Tr%28mOa=&Dkkj&j~eOTWF<9)N5LFEJU(TT&#`$<ueq>MXIM!Dc4F?-^Cum^BmP*01wuKHtz6s=vyr^PW9f5p*WwT4m!?t'
    '9n^c-U4QWW_J|dr)bJw)M6q|b=3p10sc6P!_b&F#$H#4QEBGgf^Hm3}=CgiPZ{Omw=PvcWh9g_vC)o}>%UM}><uAoyGo)s%hq`R>'
    'SlQM!W8R?Sw6|CmY1)6}LFeuI`wZ7IKqchK57&@)7}X^xOnQQ3?MD3w9{QObeCsAII#2W1Ei@f7KTAAvg>w?4`2O(y*e(t7$TW-c'
    'jSzs}jiya^ocjzswD`&mBlPg+%p%WGk{R=kLH3>1tR?eV${h=1sCO^$npe<M`DP<OA&1wmJ^fd&C1!_~Jtom?w!ADV%F)3-_t~u~'
    '!n4EwCh``0aYt86yC#<HWf&7*bYGqNDc&3j-x<M}<<JIo?d{_^@BXleK^(SSD;!}M0&IdHUhrm9(BAg(TyfudzW3t9%VJxuLFjon'
    'N-TXip-4HuFS~aUaiJFzg}OXcWIy>u1wT){u>%5l_lj%QG)7POGBmyWR3g!fawRYL<%Gw(U_Bm=a`FOq%_%6<6r}%(!=P@ikN0KI'
    'Y-XpwZ|vOr|7M-=aSZ`K{hWKhn^yz-A$$mju=Wk5-xKZ^gT#Sva-e>oRbz$buXe(H-qXY5#x=-gMa%@LZp;E%FUQyJxlc@0yIta^'
    'pq+`Wslb;o4AE7`+kD>BBRM{mUfmy3_QKx-SKK=oKJNZ_u6S}$fBgNoAHRPSZ07T14gOSKt=Dc*&aIJ66m=z43f^YRU~h(+0?Wwr'
    'PnwNF)%53tm+eW)(j;_zlf0P(15J`GN(h7|1)>tFt4VvMghFeQQ<4yNO)5-Hs9q;+nbT@KNy%V>)C3gdrjbH5uuBQpe41d2G$PUj'
    'BGiBzWgo4^7A2^Y^|z-vfCLFovV1ckx_~jnWy^Qae-B3mx7a+o*8H@2P31n4q8ka`=Onmkf^;KkyA#dBYl^}L2u_0U_crxVXSgW@'
    '509gPoa8+v)Mu02CpE=el(PX<g6tf+U6>u^@e`!H2@KDY6Q9JGnV{QEhc~2Iz;mq(<&pyHLT??8gSV<rd&-~GIUvEHP2&71()Ywv'
    'd$qy6X_0=nkJ|dKQjvbz{$vO0Caqiv!sSG6Fu~uGL={E5sNKMIRXwNj%WCUBN}LOsrr*+1lVIaXtTh@3Y7Y-;;jL8S#<igK3fF2_'
    'lSE0VrY7<*dLH0>jq@|-be<t|ZQMuCDKe)7XNiD*2mBdSRh8#40I>2~gX4{J^rZ{KZ-R%D+$Bgk&R<-sHzF62sAWh+)<aWK=WkC-'
    '-phmV(qHMgBBZ*fMx;X42>?a9^jgegIg@&3CZ}dqnPBP^`_VfZU&lK#+2t~+e=>+UJ?CJmb^3gOH57(DsSZzVeyf8#9LL{I=S-eo'
    'ZbyP^Cjnx3we>6Fex%Jf*^_f7?fhJ;FlB<{C*fni+8R*>3xY~%!fQ@2@FWF7)#m&H<`8qha+N5BNhDy@o{V{>zRW<$OgZp`Bq-7{'
    'PLH1j2)3DhjCdejP{JmEtAnlK77U9;PaeLh1`f=$X@ZVZHZBR5k|-rl+7PMv`mX{(2?~ZJ&d@|neiA`hqC6=nr>Hw%L1}6*sLCx-'
    'd7iXEOq8Rik&PpjWu6?>Z_1HU6EokqnecJo8O55A6iiCcGo%K3k#X2~GUXpr=e5Cu%3n<oS3E~gtPT)4Bt<S0Y%i&I{{&55s66Wn'
    'Q=ns4RYsE(yMHzH`8(DP5Fc-WHfrjOer9a<XVH@=E|<_zO`RddRPT&-ELg|dVzqe&h$ds@0}t1Zzr6Zk87QiwnjMr^J<X;j=jyvP'
    '7ApXt&Eru&AH!5XQTbuT)f4buc$bpLC9_D+Ochn)bp}|?K}+M}aGu_a8P~zov6wMepO){U@!YvqS1?^QN;HRcPcXCCQO|LWXg-4+'
    '&cuM%z9S9r#p;~Rwe>!o_8Li~Hd;to0I(a(DISZ7Tl%=mw*1)yJ$^D)SqYO=y=-gh%6-%7DO_PYcgd_?GWBMj=GvUSj)Hd9yYcS@'
    ')VcBzwf{>MM+I{jM=Q;8eU@tbs$t-tP%I#U++K<$BxP6<#Pl%?SUVfT3;1&bg^h>A%veC7>6gdHdK|OwzkUD9@8AFXH%;|V8Ao2T'
    'd``(ZoR4nb3}M3r96)Wo9V_$Lz+e@}N)Tnd+Imv7P1QcMIcIYY;7frC6OtE+`i_LCX`&wDUJFJSnZ?BRFF|gTh;Jk$EEAFcGz{}p'
    '?Elps?n2ohiJvA?)GvE<+}1NpzL_X-66%%r+6h3F4?k4ok?`Tu5L7hbxxd<cLp}`B2(PxrKjH|9XhlM!G*vs5gZpEvhLV&TE4sh3'
    'HLo`P!`uN`lc@C9$kF-TLq}f>d|CqQ^J?>6e9wW!tgKzoSS1pw32m)JB}YPFPdHibIf1$M_S{oZ{Z|`&tf=E;usJ@t|DnydV4~KE'
    'spk$AlsclFiFyycJIDE^+Hk(w(O2M--4A?g<-QVRGiq17ZjbKHw*ZVRvo|I{<HZ=8eCx$HnM@af>0jTvl;<1?V!YaXYlD<oX@{}S'
    't9YsMyI@J^!?``S5AV)7%uKw1vAO^BxB@loep`%(vEE}3k7I!8C!dm#=6$t!UytFzx7+k5UwLfa9iMR+2d$l5LbNaue|@#L$A;Ez'
    'mh8&9HkDRQ;I$IMB&m82qJ45AM3m5NOk~~?GCB!}!>i3X_Lr;2t>(qV=2eJIqn&*p&ujc~5c37^a%OyF;f)j8h_ANo1HKaee!MSR'
    'hPOR=8=`;v@%#6`{`^z==O4fS`%l08@>BXBfB*T{KmIQ8b#I?X6zl4RPx-L>*24%5DvJgk0#gpMXt+lcxC!{4*#4=hbCJ>HrI2wO'
    '!VM&#8-ubH0B2Anrv}Lc-B7tN6$ZrPM`oHpzTl>I@1uC|L8hRdwA!usby7vulNuTF;X@&(1f~tv)YOYP8`ShsHoVOzbbBH9Tp+^t'
    '<%8dWuY(xeDBk~CH?rE<(OYNlptU8D`_bqjO;-SFgGGz*!SG%giVEzYmS;UP-kX17dk3wb&|pP&)Akd&NlJQ#^(coeAbnApAh(C$'
    'N*;DRxf!boP_*vB1tJqKmK0Z4*df7T3j1g>^SufOG$CPHz=ph=<C2)m(t(M?K!KiSJVM2AY_@1(0fvdm-%aKxLZb|LP&06jq+zlO'
    '++4o}{Q!?D%QnE3fxpVlpf&N6L3<c|mY*MnGQ|%$gtE9l%sSE0`=r;)l@v{cHz_|$-D8dMZlO^v%bpO>5638xnSw9~LXbps_Zdc4'
    'ltAAZc!itOxLMgt(k^in`f<B3kY*Q~(n~^}nhC4r^%~!*DVI#cq#EU9`U*Vl1|*#YsvTn=^~@%oP<Jyas6tL=PS=O`Xr)p+4PwbG'
    '*8^4$R&js*$w&d9N&S@Bx(UgU{p?usnN9GavBvYNW-*!nR-7vd?etE4mYdVz2*S_a{R6DWrG*RxPlfbp-!7y>996ysuz()LSz!5P'
    'e@v}TQ4fl<9{NN?DWgba^}DO5Ak{cTYj}B|0N1vULbXA0(J?uT=BGJ*7|wv?D2Ygi53Ar~I((LBu6tR@9`M9Ha=DVS$<Q*YCe>zi'
    '6Xzi%3sq=x(u{hR_i^^Noo69lkv#`GaURAkJ3!|_2Q_`gXjvsYlfj;S6Mx9&Tk(C$ymQ~r*;Ue*3?raeHaSPv@ir_AiPHiSqPy14'
    'sseI-%wW&=1#;MN9Q<|gll)0=AwVJ__?S$GKeKnW3{7^44CC&-BPd}uYtZa(f&m<L41y-%xnac(q7*rJ_!Eew?wBW!?(ha`zMVZ6'
    '%|<YU6B;2xv#JqqWDv3iSk?{LUz*-})AVjRByrP*39eCw7xeu)7c#a=ft^)9IG3qZ8Ru^5c{iyc%!Kd2-YA;too%UIO~xHM2Rhf<'
    '1)8|q@79QRl2MvONNj%e-OM{Tqta2OD#e9{)+bF?3BJbeatvV~_T~5Sy48uuTApNKY^;rpSBLGpmtD3UR%{o(oDuE{Tu5HWyWkgz'
    's+BIp27v`D7Tab&ceq4Onrh9CjNY?ZAX<i<k^_h@6KT)q+<Ph8$ymz?1mLUSliOFpgu?-K&%G2+OqMaOIVcasQ?j6hFw@C`5)RNd'
    'pG({gLlt+p{z`TbnMDT^b*y8pY2?MMwzqw8{5PMw3KD*f#ql_kx3FETh?&{Uf_Hx=ewOG@$Y9GBU>HYsF{*V-7!NL_DluSef8wad'
    '8JM5vX7`mCmpD+RM15^MOdO?3*1jU8vzG4W+*s9phs!rcl)3O+#G?!3=|ef$<XqNJ0{1zNWu)itInA_WnKs04_nz@#l@TU&V0n<3'
    'ZR}JxmhYYr=ToOlHG!8ui^O<0$gFp>nm4P39|YZ28ZK2(gv1P4_h$FrMC2{=LX-qM!_Lu3UN(Hz3RdZLsy5ziWXhF!A<j}z=OCh@'
    'W^;VYyjcl2diUNEt2PM{9ou|A!yPZ)EY9oT(X=y2-3oPriK-Jy0q@bc?A}EWKy`vw%Hze3wY}Q~j`T~RoX_hI)uH50!bdh8ANM9p'
    'IN1Ij`;omp_wD@|&GEyJ?Py^m2Z1cWf<qR;F2oSRF&)#rkuvhu0cXK>X4SB#wll1Q6tE@Gbk9#*#ya#aiZ?G94lD=K*$Re3T1T+m'
    '^<8ALVDl<SP^(=z?jg@ZVi-K+9lHay{Qj+f7}_q`x^?2J-qL)GnywNvud7zc^YOXt4oZC*(zxAK$4H#G)Hr-Y;!6zeIQQ9;&AWND'
    'ku_ak&UYWljtc4J=iEDDWY~;=^WeCS5Xlh`2@q8zW9Q!I-+ssEXFcG3qRCN~hI_)pwkW4{!k~D_pqU_%L;+NYiyYMq4o3UB$iUhI'
    '9s-&s%``O?N9dP>5kiN81Z^_MA0emVx};U^mawzw=@2^RXM=>#kH_wtjrd`$$P8axsA)<_lSXi^_B4}udEf0?RY$k@@$u}U*l>6!'
    '=jY%qo#>7UG2^m#)Z*9g|N32o9r?3qxtt&S^JR*}g5t>O%Yjq%4*4FD?2?9tj2)M4p+UEB^u%}8Z;QMD62F7LLMzK(1>+!OjZiR2'
    'RSm@WtK<<#zb=!o5&B>8*A4A(jx53^Befy$CR+d}T90f3yAotGHbHMF;)#UociFxyu~|3=sGW@8O(^LdiHO&E_^6PU<G5}R_<n){'
    'j#|l$S8F<FqB<UuE>glE115v$?3T7Cn`pn{NpKD<gtKMCQRgHff>O5LD#4w;Bg-QqaS}qpqmpvUXtL!<fSErdBM{djheQk!?eG=p'
    'D9Kph@zT3ah)wq?Su*yB+oHfuYoWPDOL%oQeTX1(O!#fMRkZGG?p&eYK5(m4u-|HR5E`!*xD4GH8gmwFS!VAcp})@jRmk;<w=Al_'
    '27k+r@8K3HI+b!l{IFHPOm=FEXA#7}R>TV46Sv~p3}i)5&_Ra3TEd6oEmXkRJhQY^?a&wFL;N1cTrz59`}q1b`i0*m-?&elry6I*'
    '0i5W2J8P&KK)$7<P`xi1<HP(TsI)roDM;*OG^sAoGvG8oABvZTJ!o7A-Qw8;B4BA=?bv`Zf6YOGv?TpcCU1LwiFBVir+Pgz!6<j{'
    'jCfoT3_C$n&6|lvkukE75~rmRd17S5xIwS6Mqd#5hQl0>=-w~xu8j_HGWW0dP+;=S)idw6@p<u1dSlOa^;6`GHJ_dS^k3f=@0agC'
    'R8Zdf=XiCtDrG(uJd)y&L$XaV&<1}r93}z2JRdHMj=u))G}_A4>x!2Xq}!`iT>{e3jade?a^Cwf{_D?w{oD6nzkSU8DRx0IPMpv7'
    'T0Hkf6fI@98-nEU_Q=LBxO`ONhEP%K$Y`%@y1ymeo`!6VHtnq=+FhUp)a(_l3N58JS|1@@+4=~%@`gCNVr$J3>*4|8B?P2w9ky|Z'
    '`Xmq1ARMgS)aR{3H7@>~ZfzY==biJkG)hcmc+9@u8l2{0=0T&?$B;>c{BS^$sw?>2BSJr<>^ZNTcv5O_ls0PeLLg9u9twpXn533o'
    '3*tYdFPpHrXlJ&H5)Xt@qV+T_U=4!JR*{7lZ;VM;UuL2qs;VeP^LSQ|7Zf^QU!7sHpe=+LU+Jn=u+Fd7YUsEsX5Hi<yVtPO%5yeT'
    'nTiNY<eKhh1Znj8)l~3e3`D9?vSL*hXvt93y+ZuDS^@6BJ+>`cUPEyT-m+y)v&u>fP1v%GI4pheqc&ZkYBAA&ZS>H@*ruSs60iXF'
    'vk7&JGFB797Hy<Z_o;Hb0OwSNS0Mx?BZaz;E7HW@S}ljJXO=^S3J|DvGR9RN+)^OSz|U+_vSJZ*0PE4}Xqs)&lTB+}){Y$drA3QY'
    'N8DWWHDZ~UDo<tXDhOA#qg*n=po0D8NJ8-}$~eG7OWoL1M3JI{dAJKLF$QXXJ7wgns6wWUJo!unVdAg4Lepx>8KID9%E5sWVJbCE'
    '9TkFaV6EqZy-?Kx0zN>yG2yiHvB|YXEuAO&dBO0g%bIczD4CdY-pI>NB@_M3VhF3U5r{8UMd{q4vQBPr9jNZfIH@9U$}M#rAAEv}'
    '8%^;q*I|_?eIpUHs_>4AzA5}}Rc2)>nONoWP9Z%hgoSP%GNh>)PmXaa7~uLrJ$RLOJ{!D#ss~a!kP5*m@Iw_}^=#18QSZn7)c2KV'
    'E@;Q_MpfD|I#2aLlr!;Hlid|mWOPNP#i)}m3zT_Mf)NO_>a~0qb;)GDDeF76Tvc_%RGPh_c_mnT6}%D!k1D*|6d<0=^pyFks-&FS'
    '%T?85P8l}{uciQyWTdAuZB^K}DOaZ=<Rz09$21c4k!OX2Gw1lL7zn4_nrNAQwCFuFr^kr*RcL++##9v;nuTRZZv?56fW=c($5o!j'
    '6b8RS=&PMrRZw|C6e9yL2gmfje0@dPr#4Mhp{=R1(q>UFS?WxsOJz;v*pN-s2orslc;k@)YX$Ky(OxOOHkI&hiUCc*@0z?*0X@{x'
    'T)<Lmyok5Cp-3q6)zDE<%C;$wJmu`P!Yt!AIHuJUMHRs)n>O`R4p}SMF!7$L@NLS`YC_4NU)bcuqGZ;T)triRw1z;Go)l#{*|65I'
    'Ai9e}yjZ=~kV(|9F{ckyy;-tQhnC2XwI0b@VC%ew%EI}t!m>3jkV^~&t+%jsc0~lnYkh|<VN+y&>6^E~x**1~ftuxns#oCpA%DX{'
    'z|C~P3%>^i4z8t&Sgv1FWz(v#${~rzUlF$}wD1mfo>%)Gf9-GrqUfRbwV1S4d#dmt=$(*Mi|}`{FBI3GTUsSihlA=^8{=rPLlg-x'
    'wB-{*Vl<?!DB8?o-h}&GlQ%~iQtP=AZ;z*S-LwDnuX4@C`V<$l>|p&=b+@=A&Ay2lM`%v_5SM_tg~%1fjWZ<E%>68nx+2ncuMM%M'
    'd`vrna%EX)i!Mk<#BHTI3~ecc+gdFJ2&hy7Z9}0QQ3WV&hb%G0G=m360vQU~hGxka!Q?g|Qq5wZhCbDDO4Ezt7%Xur#8&^Xc#Z+5'
    'sHnV^aI6|oL(%@K%sPTl6}B!&PPLc}r68%37|_bX^aVkxc%H|Ez_o-Vl^Nv3uuA#xh^0{d)nJEGOG~vCNFa$-zaF@##Neg9LDdQ{'
    '%pfO@F732o<Uyzo4yYmf289lSbQM9S4au&`OtpWA?<B*lC>4ktGmI-vP>~B`G0v2w%fS}_8JQ{SJqEcfZL9_)QTm}lv`LQIAPFm;'
    'w}Je!ZjZDGKxJh9ib5$cmYdTJ*#*Ij;3+Q)IamwBz#O(dC}>5BswjFrsE1P6%gWn|Z#`qpBTrai%Kbo#K^rlks#UQcswY*`X-qq2'
    ')6!ZvYXieRk>!$es-<tpB2*aTRO<+70GOz%f%E)Y{I;AQiu!ds7VlxFL8IP(eYf6tzkmMm+rR(u|NiBtZy)FL<ry6o=}+ks$=;gK'
    '5Op}o?NQ4K+?;8NzHOD4JvW8OS{cQZ22J}EF-;un`ySJNPW#A7nYxycS4*E!J`4H1g~z3Ev_8)yC9bX6T&U@yv^g!UqjhPR_HC|V'
    '^;@l9qASvhvS2Hc=cE0e(j1o0Ea8PY=`7!ioLjdS1|ob8q^z~BQ@|+{!nBx&`T*>UvZbg17BeD{H<gbOLRVeu`Z94BD?EH)lZrQm'
    'sHe;3<T03EOnuOzSkhh?C4jTJAhKGekb=u;&MjIa>XoM}4)Mp9sWCpLYk}kqkpe>3xkNprTG^$S)7&-VSP`JGeRnOZe6L)hiV;;x'
    'N)U&uq)<Ae4ti26ejs474j}9&R!S;}DsgRFp_7DOCC~(gvC#`8h&`hG``+1Vg5$XAe;mqeJ6dZLN~6C98;zcTB60LHC5^jzfJr&8'
    'kML8d?#i$!5VW!{J6Eeo1W*WRDx){VB)Or%5bqL8kM-PSy|jxB%{@gDXiOA2vn%dExj_o4lcz25fDy#!Tq_7=p^UTw?nAM^$@9of'
    'YaEdmDP;#SABy#pXW@BtGcIRAD1IFzi{yM^Otj|th!z)-4l0o6MxoC?02jhHDcs(4Ur*mzpL1{@y*C4)KxwK!sf1fs>YK>O(8uZ)'
    'R4gRz;I8EeI|wf%!-Ge9i7}q!X-wp6)oV(lO$MU8<pEC#(wZ8O>O=nIo(J_Px0fUGQzq(+@&t9Ve4U97qtreX>K#Z}Qhz?;ncWdO'
    'QsYlND#YGAn7oQ&8B6wR+;B2MROPOQEUzZACePq4)Gaa<3JJ(Wqad!In2dYrRU(t=Z`R>&nFTYWH&Jh_{So5f%hOj2)>P7)?Twt&'
    'WrhYMyg$HtDqT7WIICL<YJ57Uqk;E<YPe$7v!KPoPso$iPv}2$`aaV)JW$rm!Y+!F8x#|q%q~@4vw~`~SoKv`iK#UEN7;D_(oJW|'
    'FbPdO3w*R{AkVHp@zHeoGQ-WJ%ztr?{zYTlll4((z<jC3<Wo+iiunYxiS)ce&upHR4wgU1{!(eIsdTMe-dPbj-QMI)v)*VK&=HNp'
    '$zF7Mh;l@qN(dH)Uqzu^KcNtf1}!Q#Tt(8R`5a|nrmz@HexmIf2@bUBM-`jlnU~GQtO&#C$cZ~Q%hGCHI0R}fW2^NVEmQSIBYzOP'
    'VT_2Dlg;iKavB8z#7#pqpd*RxV1T?s)}zg~TD6>dWu*=ecOHuvJ14vGS_1tiHMD$l4*I%Uvyq;L<q7|R%69AFcBsjg;}HctY_34r'
    '9tK5rsW2>}z6-&kHey~o`Kznyc@Q|u^fjS~MH7}ndsJlG3aW>xkf?T+Q?9STD3d}}`c|0+eM<#Zu}$1cmct7-t_@Z5n$MH!cec45'
    '78FeA8m14N{X{ACgcgiN8b~6GFFK3zs@e3zr#a1lp|R-m3bxQi&ibpl>Rv61gmGEmEz&?Udy6EDhxrfmR|>&pNZ@hNSC~+%ik!(U'
    'DdEfsVJGVl45hNjux@CuO%d|LwHYnU11y(_nZqY~=56W2!-9pY1WuGN+H|?U=5%qkT&)u>8k|f62aW_f((BcTJcr;YrktSOw9%dm'
    '2!KZN0#AW3J2~TtYe?4iI*E!zPhfHcv0$ho0Cv^#)LP*h8binP89uBlz<h1OhQ`{;IptY&49L=~4m3(RP_1L0V#1I`DG0(RaBWWb'
    'oU|XX1B^KVO*aY5Xvj~(pBV;e?sD!iGvqFZKPiNq;OKl_vwEala2Pn{fvAV%H&6Jcup`O>!pUn_9a{&MZW8j5gHDhQ+=Ku;)Ywus'
    'iXwMW8fwB$$(`kFP%t`i^tNfZ<nU{90>Z<uK{QgZ%n6s9T&@ZGrd2;%HyQB+!m=@}J=mHE5SN3rnZBU)Jkf}ngp}bFzsjFl-Kn>Q'
    'gl4jfC1iGpC$!_hk+(I1(Lm2b6>lZTW~TFgA~>H3VhwbtnsvM2XQfs)HcVE`nhr$4On~58W&O>F)yNnT{8qYm9^`HIx#w*J+_bkQ'
    'z^?scL99*tIweQlLeos%26Q(L@mIVeJu{lX%7B|2LLhU9${dJqNUACl()bN+?HpS3U-^w;{(2`Z6nTnpKIbX4>kvO7Z>Iu=gYzpB'
    '>yTn!CWt>tC}ao8S3#i2f7LD-LOlY>q$PH!rmT$M4EGy|&=ANYIzkYzIuhbk%Ha7VB;!Oz<<2X5!zmJEYj`6C#I?3AR9&XIF(S;2'
    'upMD$D$}bx^#$kt<aZSpQZT|-C<_qjpD}KWbcjZn%njkistN0~i6-aD2;I=ndS8Jj+*iOg6!Jm{%Dua5$n8}b3MX0sbI2{O0txtQ'
    'ZJtHp@=c!(B&KwXP_`3Q(=t}kT0KIdnS_zTu9>G-7&4PiQW8$vv!(D`(?ZIKXvh<0l5mW$s&p=2RS_rBY>cM8ZlY>?WGm1u*OKvf'
    'umbuXo~w&v$LTK8b|i;*DujJW?=fduCVb3aud-fKcXF4Ou_e44VPOR+Gy1_p*D^yVt(e{MYl^-Ql1DW`5CW`O-OSZo-IG&$Nrb+p'
    'wzAa{!-{@cpO3GlHR@Qv>qOd4i%7I&Pq7y4Y~+rRwzE|+a|=-$0_+C`A=&3ml7`?5B2BWu!~*3a%B^#{Ywu8rm>Yh0f+`cSt`g}p'
    'f(P4@T6$mU5DA_VEuIBMh!R2kI7Uc_fPG#cPnd|n5(_kH5j)>T^#(%}sZ)rU&SnNZ2cPUvmR^L-6Ag9}YzGh-R%7`B_El`2+WELw'
    'S;jDoeD=U*KvabS=;ns!cgV-Yd6zCm?$ivX;R$7?Li2G<jtlDD#AK9*d#<p|WM`{wtkF~{=f;f{cjGShI!O=7ExITBN?dqpmv8+U'
    ';1_AQ0(u8OnJ@x=dYtSxwPq*TOD3*9du4(XvSqm#EHKwtSEegN0VBe-aTnhc$#U&B5wfze(w@tvTQO}LHDw%^Qt;0e@l4|6#8fwo'
    'XGc|T-Ngr^<7G(CWa*M(<xw=}?y;LUZY-UI;W4c<!q@r@|04JZGTP1W-mGr6G7<}zdYRIO30lJ$<5;rLMAd}2cL^lQ#;IBn@!K$E'
    'tsN0bZFwfthN|VAlyEe=W-!pVaMy_Q;jT3%nJMtwZD|kaJIAjD!H&k&f=LU96Sg;U+}vGb0y4#BS8;@MPSo^X`@s8UG<rCvvcYzC'
    'x9bHl2Gs;wb5Ga0s6|Oi4VO=uFwBP|qHt(-XZa{}&|lAurdr<`_sL~}?Y81s{A34g1R^j-CMcGi!}TbKFO|2&HCR+g2}CQd4bf7c'
    'D;JFMCZn2-HK=yl;_kxT;4*2@uS{Yhu8n76*~(CPb~|ZMz(YDz4oj%Mz`;x&&TBU9@P<yy%xoe6h>Gmcv1j8ADBF7V@ZHyT&toZD'
    '>$<z9z>fv2Y&*1aeD}Ujg$tMAE1h-@i+E#jc*<eHh6^j+=<-XIyE*LbA6rn!X_H1oSkFDqvnjoz-fTx_=sMh%_TbYzVmek$i5y!>'
    'C5@&@h2z?=wUv$XirqZ4RXEi?mTPR3S#sQ}&o(-)^_|Gp)1QmJQy(w)I~$Jdd~$3(IvIu@n~;TYE?I}MVvE97#>coqy&Wy&WQvzN'
    'SWYLrW`rMs*01+(G`2D%nRK2uql@oNG78b%!S>5fAG3AM$czw)(lQ+zL(NBmnvm8ii$h-Bc@7_AW-9JKpLrWB)SCT=EuCf|zX1+g'
    '8*8ev2KQ^O_K9PJ<lItlVPC<X2}zh+<T*4KeCHBa0B_2%E5jQ@%hty;`*+zU<F2tK-p6i^b2yOd4z_s!vEA$S_6xR#=VQwry%wNH'
    '2SpK$%XaL+@Uv1SAzxYBC{hEtCHSyf^&V4Y^~RDQeHiz<7NUlDICrP)6nB<#X?cL))?fu*mNAYiY^#(BfYSw@@rC;(Q2YYh#1;dG'
    'P4N@|RmsOC;^^mE%NuL9ZCSXcLJA<ujd-9e9H_r7>-wDM-<L|x5=sM6pG5*#*(TuEID+uSn17putwMrJDGOMwmo-F%kq8Y@>2~iO'
    '*e~r`Dj3Q|>RT2`0T$Tu*1{9eW@oX&!}i_qSU^B;>&7kxd#dcxQVI}R7!=p5AV6!8me!_i;o}IKrac#G4oedm2G|8wi){+Os||tw'
    '3L0C+qN+-Ip^u8-w?a2C9k0zABIa?Ys@LrexfV;iH24pI&BN&I{!`a4g;1V;zOveU7wN<bwalhezsMJY*<@=Ug0{uhT7zWGW=+#O'
    '?+h>jj6Pd=g=yK2%X8WpID+`?wG*mB$@mJKP}VsiU0_5)={H(RW3hFFOWdugYb7`ZgE;32uLZFZoEUymVX3yaFH|hBebGoIcmi*|'
    '?4UR61Wy#aL#v%<S%){$NI8r*VP0gD;h|WF&1q-qyJ+9KWJ9ZfK8qlbc2<k5pqAgIEu7bSx3sPw^nEo}9R%vsRsb5%U0ACXG9i;G'
    'W40jY+`OeVue#rotTZH0tm*VFJ_0kaXwjixit4%IuZ6y!wh9}445|~LrPdN3$LMh^1vvnX9(@Xb-#c<_eXt5-U_tj#5-9}RqOK7H'
    'DuFNXspL+TcUWF&cxP50dQLWsfP#Gy`o?m3>N$m0H~mWJO*kKIg9`%l)I6#-W3gf$zN~6(<fN>VBWSkRDym!fE4s-*%B$i+ah|*@'
    'r7f!>_)De~p7Nz41O$Mfy|V9ze|oS&ZCi>BYXvX_Mk~CqrD7C7K0x`Aaa#mbmD*T_l2q)6l*WX%C#by85@7Yl&LH-uUjsd$SoA_d'
    'V1k6&_yxj)BiN_$tGFq+gasuv$S>p|2KJJYaM=p(4C0JmQvzcVnikq*Efk%wk|X5z7U*|voYx}oD&#O#;b6OrjKJnmrmIzCnFB+='
    'u?X!STRIqgeO^OFLF+J9oE{RW2^+79>JJKfRH&^MRd)I-MNlhN3WC`Soj8JZh-gf=5Jp_Kl7%pv!_cfKBj5ArKkO=@c%Gm~GGGyg'
    '9z20-1Y=z+D!^KMx%SWNCo+f2JC=FR6Etc)`x#a|6~)JIfY0Ztl#W>J_q@(QZ|ZkL_GiWuP^ZD2#dpJMU8F3q_744>MVY@WJtLbm'
    '4P_tjHT$muz51_0q(Kf>2>mjTR0FdtVZ~p;QZO&;wy@Kp2vcQ{zCtKX<`%L)3rbe`)5~ZHw~#2pM#CS!!v?Z4VmAubns=)stY*P7'
    '?;x~Z6jKv&K3Cqw=QS)1x^5biT%nAaf1}wRSIfdJ%P(mK#nOm<{I+&jDJmF_)$XR(5Rf#`7eX8D^9V?rzhazts*WXG!R5SUJn#}1'
    'MXJ-079HC=dT96x{7RjNM_qXH9dk>mf+AKt3dM>rUaZyl+nasiwa<5tGs0#Sj~-F;GNw^B*o;AoFvMIcC?`Ock0&S_UHGf>qq<@t'
    'gu!_0w{^1BGuDRPm+{M2L~8+Qf_N4`nT#uId)YjD9mZm@@)uVCg4BxhBC3`d(Lx|I`>B|))dmfX?@EP3_iS*u(|=XqeWTXCN3`z!'
    'E*9-S{`l>WzyI`4fByZS{`tpmf0jn&{1mB$FtO+Jb)7&-EtD#?TWMBNk&D(W!LkuL2DY@2GqxO%5U&_I>LwZ%5ukZnq1%(K11zT4'
    'YR%vRSg$z>Gf>%fb8)j1eE!Wr7lmY-<1|zNkxK-p-z-_9U>OR<7Sw9XG~g&}n+6=Y?O?~&qnjREaOMzvL9M%4Pik4UIqtiy(Sm5Q'
    'XBJ&TKA7$c*%vgntuNJ3n>H0WYSSkuqnjC~RqxuUEk|ywnSq*$oe-Enru&%s93^fH9HQKPhUnGPlq~nK23M6rw@Jw*%f7D+Ll5bw'
    'F7a)?rhq5i3R(!aTR1}-qF(qyO22f=*lAttma*HM0E*2j^vFtSRCh!|DZJ@clT+}lTd+Vo(vH+=)%L%ea$noGZkBG1^^Q<!RVuNY'
    'XJ6){Q|r&P=oSx<A?y~sQe966xYrcW@=@{_9qqlQ>DQ_KZs;b%k<{cFbvVSDfI}TclBWK=P3}<q!_(5`g`7z@7r4w`H)p#~Xl^bw'
    'ff)#SV;L{S$vb+MK8k8^mC^0!Syrtr9ibK^zI%pwUemf;fJZJs$7`-ihW8wSBuTg*uM}$f6Dsq9e@eFO<#d2e(DcT-Lk!rCWgMhv'
    '&T3cJRh9Qov>h;qVNLqCQMl16Mf;D*&^H3|<=vd2Zq+Qnch7x*u|NOir{8}5{!P`4S#&Drvuor(#e{lY8S~ekfUM_hj=Ey;^?#W{'
    '|2<;Kj}WV+70i5PwO0Ej7M$r;P8=al%gE+DM39=pVE0zp3LRG<2E{bufJj5|B@})3n?{+dX2gfpkhIyL)5cY!dIY7kL#%;PS(XH1'
    'FK)HE^>{(n*i<IB+CRbBmE$u(geu|;`94S)RwincJ|Ov?D;`~5G=iL54%TmjuPJe)5R3gdhGBry)P1R|Q8lX~WZDM{0H)oUK$g<U'
    'lH9jy(_b&H-KUmQVRK|o^||G$DkY6&&^jua^I>Qe{~rFk!Uz2O)hFG_-46-13SO))raeM;?HxlwMKzsV!0SGMR~F~XJCHZ3T{IDO'
    '37rdY@aj3^i--SR^>-j(AM#WM?@&FRFq-<i{Cm}q$^!<0*|#x_&Coh<nwHiEb0=jl&K-u#ab9;H2?IA=s_<dOI6>Wku%pHU&=fem'
    'p?w3y;sMo3eVa_Fp0QzQ1&C6)3_3m3-LWoOg@?R{=D6OmQ;_umo_$d}_hEdSN-0(0viUb*8vF8`C!awcq)UkgATv?$3ww(0n3j6|'
    'Un`la&|Spx0W}a>zS@t%^GKqaKT#yJ4JAg&MY(e*=SYd;BeUv<O9e7pqZ@@CVZKmE(SCrLUZ}*eB;|({l3#c2URt#D3@h?jFy(b3'
    'JazgVv>j<6g$#H%TYyEeK_)pY91TzkOGZqfM(mR1;zZpvI~v4&`iHYt^6Z%OuWlj6t^CpnFG`&>jOnsIE}}rFvZ1JhIApUJ?zxJ~'
    'D8b|tQax^=@Q4%;CJIGmo!so<S7Xi|tN0HlSA_`Tt&!DQFZU08XE=RtGy3Hw&yC+d_T>TI5A{6g`{Ca&^nS@NFMd1Y>7S>^o<4kf'
    '_37t#&p+Sdex2ugy}#h|P48EJzWXy0P_7Z>`FpfH<L8d3XN<i^-FqB9BQtw5M;8e`+pKY>8s|_ts#azsQ|euw-JML;tMlT}D^Zo{'
    'hJv<ue#DRQ96#VvLV(GjLxoCCXeF^6Az*8d=!4IpDSbk(yR7^5u7WpNPbm;)kD6IM4xpvdgxro&K<!6~t|5Z3=^=_r`cc4?>?lSv'
    'sY@{j!Js_PWH=vfL}mi8>A-UeP#m&ET8d~W`Gan@_MsRomSk<>D5XM`6^>m|&V8U&$>UT%27oRToGJdr&#GJtIq3n+YzMB<qT<7v'
    'fcmS}PTVROEX}}4{$`O(E}@RpOcvK-VxUp<|DSe$GU>NyYsP05)=gjbbE=5_H4;r+vk{-_@PHti)gg;&?91Uh!2^LTdNZ_8!r08T'
    '4&jJT$mZ+|Q=q0jYMS16Jx4n~F(Y)MW=v|Ga6Gi`QP`X;qBeZ4B|B+`(ymUh2<>eas75gMW&5aR#jf`b1zr@Z{S24~`N$}$_zu}W'
    'T#PQ(W!=KY_vqpoP6T4ow!R*WCEeDCCO@1_L{+RRE*j8nIZScHF>CaYfO+9~-6+5o?mI1gHDNzw>yh70Jvy!%%)xtkGf-(_Mj%aO'
    'E6O(pqua9iN5M2yegcyTah9WrD9a#uZ?Or)#gLJJeyRCpxjW*iN7oqSaj3C^QgRROzy|<XW3dL4wla2x=QjNQ+>FoN%>YNA?Rm#!'
    'r(z2k41|?CpRfmKC+uEkb)(szI}|g~sc8;5e>a-ze%Ow<1S1uy?q^=SOX`BJ`hWl9@4tNi_0J-#UiA~Y$aX&U4T7tBncBHTEn<7O'
    'VXH<%90Cxv6{=?LL>vCiUPw<j2zm>(ip`owmG*JF(}vrt5Z=!3^Y1+m2w=|7{1(*E#VK*frhOlnLpy3$R$00Cj5hZ{jR|hS1<zoI'
    'b8tcH+u<l&QhSq3lkYF)OEkq6f*f6x0lE*z7r%=>1ak24?vVE_#!s()-+_z=VSUDHN0;4#P907A1zBB3b($h;wHL?X4sH(h)9yZp'
    'Yn{N~aAZ&J&%0n*;yY5Nu$t+JH|Lu;S(})JOhd;sFAUT@2fYyt(9&M))n4m;oXX|ed_Kqd7Fa2ZE<xb>+Wf=ahBDt$93;W<DzrAo'
    '`CbS!b;=W$zK(QGxV9iNBC7+)>uJL`&I-Dqq3p~(C&oK$Xn?4sRkCTLTe3(|ENIrZD``2OF{b2L>J#+)p?@$LpdAwgjRl-#f-<+v'
    '4Uw~|zg}$wSxBx-c*f#<c&|EJOyCK+CrKH`hV~c*B)@)Fh<Mn~I`(P8xzts9TOdYI8@&5+JJT+l&vBj^1ANaR2#h{L$I!*`dk*Js'
    'kodE_kE1>l?ZNR)g|YcwWoPGPydfJp=XO8yTZe1M`#GV-o6~(xk|mzgZP}g<6(>cp_c&yoeNZm|qghvxf}Ww>?|a*YHsb+(MCvsX'
    '<lDNIdn&8}35wdJci@zVfu9-;jU~Q|HFl`ChA_9ob^F>NbG{R5n{i8;2sMOG)y=dZ4ptdpiQ0OG<N3H5S1~V0lv=q*J1nbjb+&jQ'
    'heaFSY4<zdfrzTjIOG|iB)<xC#ns}>$HJKR+$^L6JLh|j@U7W#fS6X>DMJ)jNZsNNbf~eSEpNVrSe-0Y0VS{Ye7CVDz*M@J+lrwr'
    '9fuo&3E9B>%yBkB3$dGyV(2QNe8hbiKgGYrT0_b3s@O4n@CBWJ$1h!Iru4jQ8##`>t@I}mLTlsP<J>>{p4f^!ZEt-a+KlIH+kV%?'
    'fOj>%2k}U2-aYFd$Jcq*Hgq`O=SRvM_SZh<d%ogpK*#gUEhzF8+t3`f_EP=$a_zkr!cBN36CUtRN%)+sFT*zEo)g!==Ladc+Smxv'
    'wLAOxdAAStj`=wm)4Tfke%eTPQqlBk*ZJ5oYNjT9f_+gFgmznB*dvHZA>q}$!nQIu;H`$GyZ|1RZRL0GGhc`uMAEamzDNAh*lcS+'
    'HMljc<kYZ{OA)45JNx{?akt|>_uE>yi!tA<SjAeS+4pL*K0?BJ&*?s&ak5b92%o3ZZr9&vC^6VqX?&xFGEN)zTX+s!YpBOwZTlW*'
    'e0cp*Bu|1MT#M}3v_G?WXU4hWHm~;daoRjHRPFGt6suS@s^FHMM90t5N1`dd{jv;1)L3j#7g#&1@l9@@4j-R91AVG>t!bnR#JFhd'
    '8DJA{c=1as@9UoUF#1BTwsrywLh%OR05ZE*+dxXK^-UwFSPicMnt;|tB2&5+E=(uFfGns@)z{+w%td8lbbsAv`+S%+w%?cc+U^IQ'
    ')8}lTpV58b9F(LQDb2+i(EtN3m{OZTm~k8*l{KZ24hQ6+^A`}9%yF*e{ty^1gc_W0SSUj|Kd?{{YVa<v$r=I3XW2%OEL~pHv-`H4'
    '$G#2hK=Jp*4wHNeowGte4EwsB`2)E-Ihw^DBaZhkK0KOnz3|M8y<YI;LJfoGfkw9%<|iAUI6kl|8Q&Gfws`(HPG7;s+emaTwo;9R'
    '{_2cLV97up)hFIlAjmPwwa1MlgEdbZHGMHR{f#W^aXe3%U+_*tVx;eTo_R#d)9=-`_am16JFYfljCpQ)yt#tW<FoNwi?Qc^7>qIR'
    'Qzm0IpZB%(%l%7v-yi#>oT%nef1cCx*SaKeym=1V{+zdcsN<QZeqYq+7fKCn(r?j@3QXHp&i8!xcV}Kq@5Ovi+p%n$TkpEp#(DOG'
    'ZMxsedhE4r<MbHI{w>zk&b6%34Uv`MZ#S@kK8JT4_!R^I8m!D=!>LyG+AZE=+E!7j?{>C7fXWL#389GA88h;T63m%j)`eI@@#h9U'
    'jTKi%jeUaKguu{9i*xSzcio0v9-jr`w1ghXt8MWj{K0{)TE<U=;u?DO0(kw^27rzF@5uJG(GwH?qD<?6W&7;+w+u%vmIt+-hODVQ'
    '(8)jAj%}U(Vg04}iyg*g5!QY^SpJg3&3*p`?Sn8r&p+6?p%NQ+Gx`65;l6k4tfETN8Ai{KLK+U2mwv~0+vCd<U|^mH_51iQ_f7xt'
    ')meFqrBOQ)jrs>aT>Zn|e%81Nr`7fG^#<=?<?lazL!<F?Pso;1IiI=1R1y^1B~)tR|1I!pCW5v6gDc?kx2sPzaf<ywfm!;?4esy$'
    '@qhpF(_jDTKmPdfU%#Pq@%dWALRa8ZRdY(~#IKJ74Lts~aQFU4;~xu!Lv&l;r@c9B|NCFl|M2~<-#*5t*gpElsGou%DDP&PG|^e9'
    'eo0|o#<TNrV!7c6+%8q@7h;SdWjCDDlOyj$p_p7CDN#xh#t!BCOqH>yu^UQ^C`~?;<kEsNv|~^l8MQ8L%6u{zNTpXh_gUmMhe8el'
    'XpKT1t@?%v*p6a)+z&*j6s|#XN|R$0&>sA{Fb*qzLaaLelYVQGrT;)RP5Z&2Y>rIvP)eeUgGfOdEd)cm5P4!lnHtf24cQp7qk^83'
    '`jn-Cm`3Jp>QUuy67~j^re2e0x*!h8l&*cCv7)b!umOBM)V^?}Pcs3z1Y6(GRn$teQsk`$go__&qeY{e8%TJ+NpC=2p`NF1QoXUa'
    '*?Hhstl6D8=>*krDHF`+-q4>kvjquMbBB#9GokBIDx8}NKBQCRs3<KuOmhea4xGU{XWx&PNs2IH;sfi$l=$Rv=iuQ}?~;)|(a_!1'
    'yZ29ioK43nLmoY>g&TQ0|49&^ILCz#bOCS`2E>Q~0kBIivhENGI|S8IqYkM@Lp`l^?*7U4YHm;_$Q)CYhaR8nVu-*^pm`QVyywQH'
    'X!$`NKOJ&JnVUS1U<B_I?qvTDzy1C%%DB7?*^#=ne(sa^ihRp7YNlJIHM@W+Q6gQc1=X`rUo?AUK%UDURNe2tegEVC`Tj${wLg!@'
    '_LL0yRH95@C4S8*&1{`<-NEv(<{Lv<9AGNl0P4G1{~R<<NRpSQbAZwnZWi}_4EP^^|F`eIeEZn#Q!Fj2GToo`K=AK0@0PYEz@bgq'
    '$%53KmOvcx1`!e63#@=>{x>{&KwC1UH9DrGrB>LzsO?u7h4#l)(1X3G<5xk2h)?0$+Y2IEU90H|2f&8Dmuo>PXt~?+8X%n|fdNX5'
    '+zea+dr;i->VU1C9F(+s@EZbC!Z4|A)E<!CLctGJ^y>?5B{ixxDO-4LWKBs;Rn^OE*-Z$dw=&KZB6>wv08YXGUl--aoCmXz;7W@;'
    '^&Ziun(<IsqrqKW;-H+wrXWR}7J?JH5J7mLEkZ5$16-OSEFrA=@CQ)NP@~vke1MEYo4X3n>mmjhHq7FJ>zygWE-)VzSKuQ8-ZYB0'
    'O0uFQ5LgDgAOxaDEAIhHZp&mgRZyUdKvDNrrBKxj*(ib@BoQ<=D6-JhuM}fIge)xeHt{YJ6?zY$Teo%sRp23#PF1MyJ>YI=d^h!m'
    'ifxA386@JNVsfN0#?#|)bqeLV;d|6}cEbj#;k_Y#)==K^`{F$Hy*9#IQSB-e{?_iYBBrrS6wWAyNkC&)HT&9*KXc}pX5WAR?e}kT'
    'M?YWNF#N>)`C4XRr*ifQ7M?VSb0N&`$a495tP{2NdGA~Q>(Br4Gg+-q;mIe;t)DLkM3yf&BQ4*v;3Kdr2VXYIUmC8ULlb3}P=F+D'
    ';LUfM3T)iu`U>;8qyNKnx5tPLP+IuabWIsem`m{@nfydisqm?xq%CWPoZ;6d<Tr?xRf-Z2hNviVp^aE6OAz=|f@sS7qXiB4@E}`D'
    'ef7rC18BRENu!C51BglOrtVKD`I@Q=BpP>c3X02!0--Yw8h9bfW<DU4ga8KPg!|DgUrjZG6{}r%X<_8hazg)jyBIrBTv$TAc?47-'
    'Rf}4p?@U&rZ%Y|B<PuRSZiZK+3Mf%nHpv}PP=WOIu?!({NBf=&Q)RcVwiP{*?KU34YTG9=!nKf7w#(cD&hr{v<eyR6$}m=C3kP2;'
    'K!IYC#8}x*hwm5!rFh6e#N{oRS9Zo(xeI6^4%dq1-<P2q30hbR8&Zjy9kwXSogi9MsNdOWRT5p>-U#V)qg<jemidL$1jjM%_$}Uo'
    '3DbcDt_STiaqe{cpMLyb-~aVre*SOT_q6r73BRS%K2KN5*z>IwzOkbFkZDWUOihLyq8p?WftH}wU|sfx79f1es6Hg7L~9mr7Ie4L'
    'lPVT#mA#;Bej|rUg;gV@Uo->asMh{gQK!;uiTetk-`L_)h~2;=@Gat8b#*i~%p3Y+$|mW)Y~7}gA{~k<N5v9M$jXnKVm0SusR+H1'
    '<S5!fw&HU$3^B#qGA$k(@xfv)nFKKcXzT<ery=78<fU0sOg?XGJcB}FO(M!kx|){qxXr;)5RU^{M125a&d`gs?w^34oMAKp6*36m'
    ')n*nl$J_8GR-HN+Fv=dZ;T^kPXrT3{SQWvR1V2|BHo7Wi7}a->e_z<$Leq^+oq!?P%4@x92xJv4Y;b!ggQf_FKoQYU&6T9y2ch)W'
    'I7F!gv#sr1BfM6Gkc^iXfU6NKD;jOh#vG#aZoN0Xn{c#=%BBW_2sm*=Nxqm+9gQ=#k7$v(IkQ#b#D>nUx<GE;I6-4)B1S20S}4>L'
    '7@kIs#!cj#iiS-i6YR^7WAZaBKaTKVDj|6_E7cj?K?L-k{_JqKA5zouSes5CK!vNXX4X+AgMv&{Mo8gSTjh!H?VLu~C(x!nv%ZN8'
    'icDGpT|^>HH`58-GIu)KS9D27^RE<c#u8dxhoSpw=2M4}015Uy{8{>Q)K5n9a9O7*^r|8~x5<~U=iWic$$Qi<#B3?`-P8$ms^(0$'
    '1hDHyh~a2<P-);f)IkqKlQG$e1#B)`_1)ARFhQN(Vfwq=gb_9T49vGmmoOo@S-F;g+7-<JY899*e8Od0qN0Os3!_39+mc&~;Ps;I'
    'w-MV6ITVJ-qY26(EOZHfhnR7Kn!HV7Q(Dp6<in~Ss}Tx>_M;(9T{gu3M9wtC+QehhVncv9QL;=NIN^uHG*x7)A&OTH6Vkyi^oDHA'
    'x8|FTISNLOr&<1JzE6<rqaj9m?$xv{sf4dIH}NVAh^{Q-+La^54b1@2<XD{qMEC07h8+QH6ibqu=yF!%1$<Nsm4XKe?jpYsoc3`C'
    '1u5g)2J9OLPI0F=+VeG6Rk3E|o?cB9AXp(aG`u<?!P104PgJotakjpEBbGWGB1Lc;z@DmNKrd7k3l7YvDp=YSvrBCLZj&_@$3=P6'
    'X02V8B+Ed`=V~aeyTq)})dQI$KW`7c4ZWMhI?yteqtXg}))3XWj=;+=Kz%CV>@-p!U{hly5*#wju8_R%bGeW)p(l@lpioFtc<tV9'
    '+KT6o_qmV9^8L3TfBN>3UfY9BSihCe`I?iDx02Yg4ZP&iETtj8K=)$YvW_rc^jB<~<+8k}Nu*;(%Q15GQo}$^Btftk<GL?;4ZC@I'
    'QS_q=vLlUGZuc8;1!Gy$NxhOlSM%8#pGu2Cv)l>kq7^O&V-PWEguMV=HOs7!a?!Qf$yqbzp<|CQivVb1q*GX4p9foR|3k~Q$_d#w'
    'C$u35Q%@vh5VO=?N(C_q9f*TCWPwFbVu|dBk!zf^hPAir2D!-o6+on82LepOQb$YV34nZLa*(4!Jf6bJWDzQq&UMAKyt_7F>>x|F'
    '2+s^HQYaG>T)%aNAn`GyiG^B;jwH&MsnOKpON7QtGV%vG#$iP@onRG(gUN7oyGQbVt6hkSA_sR+!U{o*1U9t-9w<n|*}d=VD6zYQ'
    '1iTe6Y`fW=gPe2H6rtL(wDqv`MPd_G0p1m*(&Okd<0qb-koE<GuhpXxAH%J!U}_*+tq7eLG@vj&1)T@LHSoM5N@5Q;9+DuGUkF7J'
    'woj~UaG<IAJ~|L7;b@_EaDh?rvQ{?)E?%g0-@*FR*|8t5&Inbo)GMCi@7Bs4Kq`oI1SBDCP|4l9DgZbD^%rbn6QyFXRVM0cswl?f'
    '6KSmC*23F`f+2-HmgI(#5|wR89p$YY!a`^dqx^;-G1QQozp@>uaTnPQgqIr_Cl$ncH$e=t0`k_Tz@or%*pZPJ$idD(E;t)jTy)hv'
    'B@~y)6-&&;?&?4tXd_^&)#{L~B$$qZEDQC#Ee$B@fd_~JmT~Z}6Mr5<!7c#o$BDlxvhKe&BKUd_ZLUxtQdr<F&tY<}k^bMl|CjH-'
    'egCH)fBMtUfBxy$zbPb}g!zB<EJ|sguddRYtU&+2?ZVquOms|gEGA!@%{I0LFHM3G3=+Tu8#NUW38sZw@$VX3H_VxqgPJFy#uCj1'
    'U5JyD3JM{9E$EwxV*YWey(i$8Y58@sewo5wCia)<{>2EqO~V(1@zspHjmsC4^VJZ2F;8EN*4vbQF>qha-WOx|)kHpq^NTtCYGl8d'
    '-p3%nt@Ibm{aEw2h5usp<BPxJ5WoKIkH7!=$KU_*+mCM_<$S8(L=}qlbLU1ur{ju>1@|_ZsjrNrRQ$C!Na~c7YY@Y5@#!vU=|pS='
    '4(=7&o=j1+zPY^l3fr@;6(e1dYu+oeGK*u)^f|7bmR*RJN;WIMOM>cB`*rCE_(|n`6+wbJ8uBWR?Fe^^N5qBQoBI*pq1YQroX+ox'
    'wR-St`m44T70N#Cg2?_T7-xj1_63A&iYO3_G#3=6Bkv*>v`(>;_(7uoq9?z6g1`Luh0ts6pEBF(r15zUh@nWIh?Kl5Fu4^~c1yY<'
    'U9>%0@aV2;$<6y|U$7OjiDTOI!DggvP8W~urt(`9dRtqcE_$UcDy2?PoxT>a#}*{0EBmyyhwA#zK`|qv;;iB}(On&t7YyYTjwb7n'
    'yK{=UTy;#jIc?((S`BiGj<QZ5Hcll~EoByc-Z)P=Ghr`|x*jyg(Xet$v8<sD;34O$?da*KQVi5@M?94*!Vrna(MED80dWRzE~R71'
    'taQNzjtu({0mu<2M3-ZBv~e6NFg=Q)(g#O#v1>6p3N3aG&ZAgw*ML2O-}!z@PIaLYj>3Pw^X5sVLcDp>*XtkU8HP6WN1@=MweBcu'
    '=-yC}VyK4WpmU`vqN6o<D97XTl(Rhonvym?N@Wfio3qLBzP{kuXrM^Xc{>*^-5e6=(#w&#7}}s5eeR(dbOhiZg#uZ9$03LPWeN-n'
    'pxAaq9$)3$#ZRH;bn016{d#fp_|acYJzb^(SET0JJaANfj}Vaj3A*FeRMJ0NRH87*XfkFh<-H6Q0njo^XvB~+>0{ARtJ+sL<8i2M'
    '_RT1n&OExXlp|AMk9ZB#DS0B=!evzdA4De<B;tdr301M0q8REhLuO^tI@q}E2_<M&F~4!4o=kuWfQcoejiN_qHR!NRLJ!novv)-2'
    'xX7M)KM>8x8fezfP?5UHeY8PhfR)nSI~4UYePEv%j6i;Ez_M%}BOV924J8#A)CHQxxEZzI2XX>R*bkB$a%?*+K!qZ42%3rljD4Hj'
    'KsKB_d(oc1lh=SIvM>LzOedr(xUe0(D506Y(4La_6(=kZ1aL7a&(guJ3J9v3p-H78{cV~hF>5TIcXJ1gU(iBvnM`Jfa;wNW?8r+%'
    '_1cO36D;YgxDq;l7~LJs#bqk}SL`v)1tRSGqLx9Frc<v+7{|Elvca%WW-h+1xP#E!eez+IWa!fb46qk49v~C|)kkc%Pit?<+MxBp'
    'ZowreMe(8zsu{6}?xI3PeEWczLH=bQ+)RXHUDLp$qz{Ii!n*eaz(Cbp4<a!*ltvDC4hFOpqC{Z8sFt%MwnrK9ZR#6SAVNWXr-+q^'
    'FAz1c!)jebLQJKaE})9$qN#7}D(3=$!hpz1%NMn&h=%ow<O@qb3whp<Dqy4FXFxkG>s?%LmKEyi1%3dP7Mh19E2ZV)v}nRq(I8TR'
    'nF%OM%_Xc{*MD9Qb5;<_WFoV`6mv1fi%OsMV=`pt;6tFAqmQO7)X}G<8sum*NR7xDkVw21g)``(rk19WurFu8Lc$uR+0Z#J)6%NP'
    'SJ8T)JYMgtjz@e1Q{rcOK<xvMikQ0qVMmWa?boTsf4$ZXMaQZ6IkMzb2%UmMqV-e?(hLn&erlBt)=UY-qarhErH&#XT`{$z06BGD'
    'M+;dOn(Acck={rvK!*hcVjQ71PG&~FZhsWO?Tafi=3&3Q5OFSOVcbgNGWR#cKAQSlL=s&~?v^#`S`Ihf-3BtE?*Rk=8x9OYG)-MQ'
    ')UBI#`OKTCso3w98S2W2J^t6*Yy0KT-~U}+Tiu_!OrGBQ?2QG(PIHw=OBV$tPI?z8zOtxn2r@LV;T2ujPXHMpq*^slz}LcLqCydJ'
    '%N^-W0X+H!c%V-QkOub@Cf9Att79;F0B+E<M<rYM+D?WNR}`LCQStPJJOh92p;z2tZ50;0FTf##Pp4}&6qLaNnt^?*S|j>IY(aOK'
    'Ks{takiR@wNAbbidJ+Pgw1@|2@F20E&*>}5v5l<CyLOSxRAq&+kQHZEnZSy3J%O=c(GFVgqVBmIW_=*ppg}gcAGWgt$bc6{;1F!&'
    '9EN6e1X^*i4e*eO<(b`@J+3vmiN5awuAw&)SiR~$9k2vkGW7Ul2m)+c_xs*a@(it?{!b@{wd?y74gto2&!ar<jxu9XTFpej$`L7R'
    ';8ZPn{Z`3L$WAJ5S)HF!z=Umgw27l;XIlkg6>*Q+q>8c@isn>MR*;yF8toc6FNK=xah$O88Hv|s#X+mHWm#3Z^$;4!1OyIw&aA*X'
    'C>^zhcskR^!4$EL&z9BpvsH#7`MylJ$#~wZI6ysq7i0hciffzP{K;u&>#(R@iWOE8%BYpvmUB>kQMl}gxT=ECIMCPv%Q$eHPRFGb'
    'CJ^CZ!h=V|^|b|8oQ{H<0~4|ksIMj}ABUr;w=3v-R0)Lk$E&Fyrbgo+Q%{M&kqxTAv5zbuIzRe}X~v1fmH^FOouH$B21V}#qt!Dg'
    '8Zx+pVgf22SF>4gC>|2_19mR8eEBs6Fb-o^i)B2d(xrNI5D&!NAh_i<@lFn1GOWftXco)*zMAZyd9rb0yR2q4CGD#@n_2N+&9R@R'
    '{3fb}&w%=!AEsDO_?uTVLvO1S1o0}HF6mb>_|@$5X%>hOGl4>#psebVEOOZd5&^hq!~beR+&?(FiiN6PP2vU0m>jMC@i_Pkn?Pj&'
    'HAJr$US9K){cqeC`Y)O4SF>)15^mEsYwLDYK`V>yW4EO(y_($-jdWnR5r*^Cggk@*sy5IBPF_t&l@CYZGU3S(<uJnqAtZ*zMvuEd'
    '&VkUgi7_^sR<L(0xUMw=Hse<tO*M}huuVU4n_4YMo$Sd^^mntiAeyL@sBdEE=F?E(zGmMS7H-a6=uQ`&K`&1QuX!(+_nS=w=40dR'
    'A3p=qV1pBgDs`U_jLx@NeKTFw94r$C42ltok%V$lGXx(LSDOwMsdtDF?dT}LF>1Dej%^rr`EDBw7~eRWJTD}>3=KS$vguz~9w~$)'
    'dQ%9!<K6%$dNnD@<OB;GFknphz;tp`!R`Yi64l%%yXtf?U|cNDfN55R(<lj<IW?h1WzcGHxd|^2LFz-`ZTP6^GUn4{@>Xa#u&IOz'
    '2i6nL6p9oN8-vNss28Gu#M6}Wk%J7R@~JvuS^MDCr0$n_+$rJlWf~8P)#xa1!5tQ=dgp0k`#ROS%y|l!;u#=cuGlE>cv?sfJ17c?'
    '4mucvcLQTbsmg&hdNAM6F$c<mA&twd=f{2d#&Y<ziUdXSYi*e>xMz9dd5+r@m{u=yF5(<s{;K)FlcOq68;qKhERb8ol1I@zqfpCq'
    'G0!(kd=C)h%f~P1qByySHFw@-Sm&E16)(!bw`?vWFR$88dfMt*7{GnL#o?FFX%{J%1<#6De;)AW7stC1q%vA&^lL%%yTfWnx9x;l'
    'CQDWcrf0lJBf7G{7_~(>9nAt_?^)ab0G?)iTqmD1%UchWj&Z8EBKXTrT4&&W-EIb}zCm1hTcFM^Rt~W`+FI?&%4L)u07<Bg*cay<'
    'be2fhW#qPo6WTE8j$v34l3;*n@<CkBZj#*Od(N?JywAb+i1YiCjqv#fQDx-KobSu>z<|or-WDw5S`vXiUrxaUk&_WVj8p~EZp(13'
    'TmkNb?P9H221}vr<Z+AqcZCC@M4rRoD`7#aA&{V}fViqng0VT>Ph#6Sl!Ax94cr0>?({R{W`~57NzAbG4pXq7QG504zJiQua6LG@'
    ')(k3w#tk_+jvtHQah#OD&%w3MagN+(qsI&j@b5N&9+CG^R<RiSrtN$yP6Elq8S7CHns3FfJ2a-wEZe0AB~r-Xw&Lf6^|biCNEQ_L'
    'e@E(X89s&#Ufh%1YW4X8VL)#g+6oYYo?#o@AJ1Hm5p?pHC=Q^Vy%Phe7CSK7$mSI%w~9ppqYt5#rO%glt$Yi{L-Y5Fq6fa|iG)=#'
    'PhE_$N|8p4jlErnL3>*0QEkTR@HrHE3?C<12JQth`-@X36m%?am-GzNS!`i$Y5)`qo_pNW+Gx$`ZJkpL-V+$38;&T<TMclI#Wb7G'
    '8Hi{)HQpiO^W0JYclWJTJpAr#VS1#}>zDgri@<$mq(<AZO+W$8+oFRu+Vs3eXU)(<g_vs#n**(0@j2?B8uL;8Ts<GrD=~Pupcm5O'
    '8BqP0erBaxT@1^Xv$_%$V)W|bP|shZN8SInv5*$lZbYX%irW26h1m&P8VOcIGL`tNffi+sPxq$~Wx(ATZ}s`^4O{1}F=Zn&#%f*O'
    'g_Dz|Y{z3&v4DHNHd!&=I;BS^V4=u_dQMT8W5B6isV-wRJ4GYz6R;R-$Gn5|bbH=q7w|moI8Ir}jNjGyb3b$QVpq%CeySRSaubf9'
    '2b4j<(T-;$KGGjML-7yK*&jOzd*Sgpto6t9Ww5Qb4HSgf&?N`{ul)o~XCSs=$JzZUXQ#PpqX_M2@RU*sZ+ke6;P7#UG7RuX<+znK'
    '-)%w-#b~_j^Hq3MeNS!L0|54fM?yeqIwuXWoxV)AE1!}20XL;M4v5;=kg$iC>v}WK9ik38t7scZAlh<Pyl_(8;}^?>iTq+&fw5Yq'
    'V=i%R+zVB6i1}&Xjb(ut5OErgAIsQ$Isdtyk{fpjB9VQ6F+L}ZQoc^hbN;>F^;EI2)%lNx-VUOXL8nLi+*l7<iS>ALHWm_pfL0@{'
    '_vQZg_ZVn98@o*i&l6pNZNhJAJNRkLc8#hw+Ras4TstC`^S<Z(jmqzPBGkGa3{`s+3l`uQpy}%4yoU#p5(m|w;O9Q~z7EBDGAK4n'
    '(U982zibo;@E#j+<$oe3^;~-g<svX+nn#TYFd7p`OWhyu6MX(DsN6XGpd?&0weew)99W(<ft-8p9j>esn)tM-x_bL_URR!T?yI6%'
    'P`ozU?^yObgvc!c-j8$cH6jiK0?Y8#`WWjq+M^z#Ds4uAisL)NW^sPN|L=XC`=aaFydJAtdiZu)t6W!RKj5w_asD$L^1OIB`;5j#'
    '7t!rS4I7oPXk=Yq(%@>JPsP<;muh{6#$B)C{%^qS@4x;0r=PxkjP?22vuu~@e7;O4RqHPO$1kpOm&XD_tqs+{bFRu556Tx1z8nf?'
    'xK?T`fYSme4CC|U&khK3su<P>>qyOTeczw>hX?!H_g|H~Nas_$k+Ll3b8IU^U&4j9?j~>&b(wDN=V<E@{(H`i&aMxYY=O%R02`#n'
    'jscjCda${JKX$Tbg&ua|PPG}Tfe9xRlw^WK*^vR3Ht*nBIg3~OZuYzx6+Xp2q516QjM9MuTSgf$)w!Msk}t*(uS#s8Of(i<Bmg@I'
    '1Aw5X?{K-WXeu6q(S<qCc8hp{L*K*;j_Y&itnMyk)BWya_6l%v-9m*>pmQly^#+CICi^1{El_?OB<|T&Ok#QYJP1{nr2(dkDx){r'
    '?v26wg3SAZ*ZYd%`-<&*5Px5AfM3ypUoe9YlJE^LG~`fS+V?FTogi5cC`&?8f=WaBC~G(fxeUUg`?fR&@x~7n${57nVsHumRH!GQ'
    '!roOA`O0J_mP8b(*js<FoZuGUW`RF?+Mx(eX_;{XUzoDft+Q*~yh&%D@&BH3zx+)Aum1URh6SH$<8v$8d<}ss&y=pB)RK~|f*1>2'
    'n{FLyGHF4Z>nqHLqfJ6%5DEv?CK6wQn`>qv>}lq<d)}8P`}t)C>gUS|a?t!!I-TV~^%o0_L{$<(Han`4(0aNjcs+!aIx4QF#gknN'
    'R_>S(d$2dYP!aDP!^5?rZhD6V@i=v*><5eO^;gsK?(i?4;p@lw^2y#G^2_J_`q94(g=c`gN6nX!^d4RB5%(U2&&Yg_)-NOY8P)HR'
    '{)~P$nb7W{P6=mGamZfP2)L<iZrK$QC|7t5QAPQ^=;PTjw)JphP!Vc_1EHnz1wKN}2UV@-O@98SL%#&+4`uqBME#*ve~8xK)ayaU'
    '{?M~uzIMeFO_c6Xaiubu5)FMY2+q_{tvW|UMR6m{e97$Kp}sUVk^21DPM0WCvaZoCFG~3^D%288nbfOS1jQ<~AV{xhWjw<K{HY<?'
    '4Urhc{Q$AGo(Kn@VB>F4gffQzKd#;+S(j@&(wj;e+9!_fy6CE-(qx*#B>^L0$pj<-FH_7J(bV7|-ozCC`PvabvvbeB@kbrIScNZl'
    'vpiMHbzi)+V$xHzr=T`17Gk-GZR3_e2QH-7MYuIXPgRT(I;{cCRkaWt;Bf}t)e&ic91E!m&Ev$T3G{R%;nde*uO~>$d;m2s3!EYd'
    '7hJg%8LHkalgQJ`k_pplK}CM-f|F3Nyla4Ub2ZAgu*Nlj!kHli<R06E#zA{G6$`vvu(JsTcTH3N54Cn2OQwZt#A)!&efPk<2vGs?'
    'N)VreydgP*3Jy+Mg^wNL?|CTJM?Txe!{OeN1^g~5{!h-zf_vbhDKCxICUt^_51Eq^WG%>KQeq9<j{<cdeuaDk!d4_$(7?*h_D1!('
    'W%eTHjRJ$~A!h7JgfzR{nWT(^e%5V@SP}c$OUXfcpMhFNv!n2nzTWBjV}k$r*Z=y<pFe%g`Ypa7p*p^WIctZ2YRZEXh8a3It>aSx'
    '5fm`uC(boxpDFABWv9m01WY6qy7=kisy}}H6Hs%mDsLAy{JV>FVIJX+B$ZXHo}^v_EH0^s3}jo>&me+MGHIaFBfXFPm4qRJE-&fJ'
    'yL+qJgrzYl&4qcFa>(1Lb(_dsL-dN!fhKwB<fxNI9VzN0w<cPmkXjx>?Th6%X@VK<)Dp(_uo49npj3_VfewIh8A5XjzABC`IxtWj'
    '4;%qjCasCAkV1XP%Ehltwd8QHno}jR#IN$vkh*DPei-)5t=<MbKZ;ApLeOF=40tRF=!ERl;2N;)b64YOb*4D(V4arE#rG(v?J}Gd'
    '&rB^4zs28Ul7HasQl0>In0PU*?zJMB%fMe%_0)X*rSjy$H_y<HMT;p2FfvzD-{?SwHIMkqL~ja}M&zCV-f9YJU<S43o>rVat14GE'
    'QD7*IyiDrUR_7ERuOBRBwh)#IirZ;5qiG?qLEMYz7?bz2&G;*@xUzn|E#6YJz+jTi>a=8s>s~-nf@(Fo0e~i1KK8?&Fn%-p+fjeb'
    'd_WaHwEW~GARD!d#q;tN_?7!b^awsH?us&u4lWtns|7+i8@3D>*ho&kB}h-c!k0aFmT4swzvw8!$D@UkKu&v}hHZ4AiP8N)30=1k'
    'zLblDR1pvm&H`_#mBwJ*R?FP}XlSf>jD$RIJ-5ON0KckR*a(PlrWy~)?G7IS2PduRdqHQj7};7VJffAYk)!wWh0q(4Xdp!No)sMD'
    'wUCNDEni~*LHlwtO5oB+Cn5jJ@aGOH;)Mu|Ca}J=gi^E4(L~uS!?*(fzp=Y+b*lHjob*9dh_H<ZMKb_5Ivo%hEa}FZ6emnGHXQgS'
    'T~}oj9HF{z51wAE!ImMzh$F-runqc#sg4$$ehd^*_<6__Pm389GIWg4C*unWzzh>UYOWoiwXj0Kfru9L0IWAS@-mLR;nd>G$0~tf'
    '151T~@c@vlh`)i`3LFm$DM{lDl(PU62&0I))C&<g_t0aR3H`Q=sx(tRY)HeLG`f2x2H7>fKP8jBO|T6j16i0tmWHf|Jqr;?4%7iM'
    '#Ub<!N;oV;+!fL>%6)s{gNddyeC2XPEA-(qY(|D~Zx{qozZn!RaAs9La1GHX1e9KQM6RLli~WSwUiYyndI^P5^k)<~OON;Bi8ZEO'
    'Pr@uvEEb7`L$qovWF#}z{Cf6_qY_+}Ozv8;3)2cjmn;IsdUrAPv#iAt9KYoP>2p!WqP+Irp}3}sDs=ev=95zN(owfNN1fF};)Gs3'
    'B({qZBC_Dc-9-iuk1Gja)(7Eap&UNP#=}tKXG0eFBI{z!9`xtexGF_}pz)z|i-|;8A-Cei;SChS<bA}s>vh(siJ+J}>Ucg<j;SH`'
    '`?S^m+pj-;)%rc|1=zuzZ!um-&d^LmpVR>#Bmp1P0Uv|{FLi*J9Y5&`K8OiE=n6ha3?8}y4*lfk$>ybLhkHn7LM_WtLBOvwVukls'
    'LRE@O<A{;12&N-3nFLRfNX2RduvGhNav~jEYjSz32ndLL$n>ujUsIkztwAzy3X1(|YGlHo7IM^~K1~_Y^*|Z4br|V!P6q!*HGho4'
    '>yf(c@O5^hc+@1vBzl8tEQi(&2rslc`ju^NJf$eBCaH`2&x6Z@sfrc|`=n!Oq2LIp6aEE`)vw?T(O(2bsH7d@LL3X#*M`r*WHx;N'
    '!2Z<JSCUysPC<BW^Ay6?2-#G4OvYj7`Q#1yU=#Y_82Vry`rsq_U@3aJi#`~QK6#El*^fRrkzS^x5B{VNR;3TFr4I(C4_>AZwx$mb'
    'rw?YQPrj#z1?qzv>XR|*gGcIvUFw5#%EUM6f8eL8r>6gt*6JBd@3{Z@$3Ok}^B;fw@zXuuf6s&r@}&r%2oO){X)|u7xL}OfJe85W'
    '3E36hfD)xpx~YZGElp-25F4pEjXD*3#gqoa6?+%*cik`)6)?5Uhf!REqRI=~x@`yVDcE$_n`$`|D6obFm(O-prrxDaHq9QQ!yF>t'
    'Q50(epSH~iM}aI~*@YEk)i+fXCR?NWj98U2h7D<iqHIKSf<`M?6Zr2Uebms`DO4LFPg3+s&_@z4jVcrLxe9f0=G?(AC$3ps=Mx+@'
    ')#W`pLvi)bWfAV`(AmIi;T}{$K_m+(hWWcorF^KvuA_HWp(dpChbgGYIHg+oeoP9mp!yTQh^ng#v^X+Yim9!0VTyPlZX__-+*hdp'
    'f&4yi7^LYMRN)bKBgY-o<kYn^bM`GBh(Xf11fN9}QZXF#HCwAe-PAk*sBE)&s3WKAteohS8^R?}22k9gt*Qbc6R!d9Z-JfJ42q)~'
    '(acPpIu>_jwsJBb#@OmEU^z^4&E!wId^18s2`_g7_upw!4>XJO8NQ7&eMU>0%qGHOG$<tzr)!htfXc3EiUW0J15D@yGpIhO-eEwh'
    'KNt<Fd5-GIAmWN0WS=h+HSFQ?p9^uxHa)rrWkhZ!*QjV=r9{cIs)k79?leTa+^V+J14sN6&8%sR76Zj(&$R8xK7+S9hF`wk879c1'
    'dEM%$r`eM0o-Q&94K*-u3e7wtn%myPd=}4DNX+jD7gM}FVj74^$MeM5_njO6@$dik^Y6d@_RD|z`EUQ}|NQ-rpMSdjzVF*PL2p{E'
    'rTjz`ddZ1|U!WB9+fDQWO94DDs_#4LGAa>^C4<=yr;L!k-ZAO$0yVl4<}jgW!1SmMZ`dx>T6m(0muz7KYKd*C2|#tJm|D*2t}XUc'
    '>=YfrB6xQrM57XYv`fIeG?M64=4H;O>jKP@;|%Wt<ddk>7op_*l(3JJmN;+Yyv+aOZ~yNv|K%Tk|Jxt`^wZb)79dyO{CnPJ#(md*'
    'g`zp@gm5w?`xW<!s$);;qDNQ7-$q}0c|sqQPm3^J6O&AmVTOZ_Uq#1~Bi0mah&pF7={~apM=WIL>dzhj`q$r_ATRCvw$W4=(EGP='
    'daBd+N7sw)vI0n+cIcoL3hCm7A>-QR(2~;W3fVtLaFHU5wKuwef{%h9M@uPP#Rt0Jgy5Syj3){hb(s}K-@~m8342+_k#s;a*&5*O'
    'Y*RgxK3)gArC3KFd{F6)YxXM|ksZ?Fg1-%xT5G!ib)c#{RN0y*`T&)vS2jvutGm)$AR&Af#R7a5WpDZbt2|I52i1WWtCk25x)Gb&'
    'Cd`=vyuN#R@*U`p^6lN^+Vb@}SQ7cYngt8ygv!;5zHyfW5-q(+E2Qh5kyFrW3K?6+0qwkTFx>L8LDeB5fK!TkdIJfnw}$hokJ52y'
    '5hPILvNV*{;WAcNfl{Wyk*@(O-RrJk4!Fw>Q8juATvz$hE-(&=%dQ~H(>LA@`O824^IyOG_UB)}{Q4Ve+I>T`IKQ)@eI7=tc4_id'
    'g~xJ3>%Y8HN-c$1Jo~)o{)c~6pniQP1=jQ}zV8vHYMB}*Ev8LXEqpbP0_@xb@03~=JertX%|z!53~I~5+tsDl@+&g2Y`A2JC3cu?'
    '3GM*ab0pJ@u>KV}zKm;x;5RIL3;pSCPme@hl^O&qQzfvb&sg!1WfxNoY4%qXugP2zU!_I*P?_3F4Y(HfN@R2Z_qHG#EPTjcDcMN{'
    'Ti)>w7Bp7!+@Z{ct3tV>O1=jws&pzUDnwWbj&x{*ftsTMEkaqO`SuQIWZr<ll_AEpl9yQo7KRU)F%26Y<yE|m0;@U%JPO?6`xPMU'
    '25A6_6k-*KJDBTgs;oc`S7W2t4`dY#y(zFz&>CXQ0zoeZpfeu;KTiQS<h>NRbTxtdg}_VmkfG_|rgkv5yOpd*{@Kc#JTp0Cn57@V'
    '?N@~R73PLyiCDg&*xf83=0gi8HMx&m4DvFQ62g~UQg%1!Xd#R=fQn|i4Ys_CU1htFTWb;2{w^QYKGZL$s$ZW<VKW*618x}nMC?M;'
    'NA>hEoIb|iFMs^{12ER@zlnZoeBYBQ90&)5aX;G#%(DVb2AcM-IUOVrDZdMEMZ0gpC=ub)$`UoOdoE!D>XgK9-^?3D#hDx|7br@0'
    'z?D<?>8WJ8``er@w)&c`lT_GiIYA2aSw$+-g>>x;Q&l-u6N=r_4e(`E%z^J$$(yJ>q$rMThoq#xs6W%a43xkXepPfqMf(7+1g@gc'
    'OU?p1Y%1K9seXx%9h{b!yD6cOi3~rfDtmCbCp;-Ss#AEaMkBd%5d>8bidd;Z2KU=D4pCJgQ8g2RMUw@i7I(hO^C^G%^Y_30`FBVG'
    'illDkRo!28CE09YECTbbDrYQ~$x8;YvehN&!XSXDpXd?_-c_det%OS10RT8OaPod#v0TWT0IMn+=>Y_Z7KJ)+p)Tq#0tBIZFw=ZL'
    'RUCynP;k3tr$aSKRZZD(Ucjy9q)UMnruhGTwvW#J<6f|;_^jX{sO|fnF9h$i83nAVQluiYjv!8j^;;1m_;IUZ&~*@SYLMy@5P?&d'
    'aFmYFA|OVQDm$eLxL;v%g|8DvR@2C8f`+ft+lb^4=qk-LwFLCwn61?;wk$r&p_^z8kTVMMsG<hQRfp;n@D@+h``7bqZ~yq~uitG~'
    '_HW53OI&AuCu#YXV^@AyLa%l7SWK@K^;%Z1we?|vy;j*{slC?Qheh{Td9UU7T7$2J_+d551U9gpJD{dlxeCDaG5`Ru7Gd~1S`bPT'
    'cd>65X)JZgQQnhxi~Y3zbYxm8S7SmaGs60#Tm_ZKGL=EI66pZqYHLdc6fp7iMV*LVDzy7Di@WRalq+y`p40o&gngYd`oa0A`{3?-'
    'RDJY`bc^h`>B<gmV^qnppq3tzF63t1qH;+e6NII*Lw#7LrAJG7lRnHip|1d4Cf@_VyS2R-uH?m5oJBZ=qUcEPUVoGVU)A+2w(OC*'
    'lKJxVp{L{h`29~2%$09xG%25P&q^{Zax5{*;L8QAm~>kEz~`j`p#T|K3T$lEQ9~C^KEFXbdODMZ!54aV=&mp~6rzxvO$tmFFn3At'
    'GT0uK5<=mFg<BU+!_6qi5JCvg71AEiu0RB2Q-FJxy(z~RFh-+_12jRSIDYzMl!_ZQAq#654+X=ZDa3{~LUd=48gMd_(2K-&?q&}('
    'u^7x#Q9ip!N6C)BI#ekJ<OA%BI4e+vZv9iNVoMu>u(DQ*!B)f2WTCQ}B|wzl)(&LRttt%d8fq0NlqZl?-7IC)TXVnG2;qqlg@xrQ'
    'Q77LG0n@{x(VNt{b0aC|=7b2XVMrS;21bN404o77HUXX*MFtWR#K<s+HJ`^Mft|SP<_9vuldTz;E4*yrT&sNslJ@MIK<oieQ+JG}'
    '-VUgiM7p>^%mh*n)C(2J8DWzIX#{AF1m{&V3!LwakJY=iQnbT9Z^t-JfHb^BeoA19(4S6n_p@<A&}xzIO=KoJnLkUbb7!a=gPTtc'
    'fM;|6eZD+Bb*y!En9=(zREj}czpO=)fFv$B=VUaVrtWX?S>T{_p9E=x=ox%0d68DFIiTvQI)qz*!E5q?9%G&)wy{3ssSSN%?hrDH'
    '!wj67=M32*z@$C%w1%3oSV#s5poKCmhY3zWg?uzg{Q}5Xt(`L<f0GcsUG!8tO4as!HsyoPJ6)(?)R-NJglh)eBq<e<x48`|U>A5G'
    'VO80&g&ijG<*9Z|0W;EE!f>LV+0zxeTIkrs$@tu8VgWKqlw9S2so|#0cud$e{1nB&7y^HWX@NK40aK=Etniqw5aMKbdj0^D<=0K^'
    'YrvY8m+ZW=ZZe=*qquZlO|8rHD90oHg&>u&gRjNhtZHXvIE%^u{_7wA^v}Qk_h0|~OQzjy_nJ~<OSJrb8#1!04J&jcvM3#1!FAm}'
    'g6^his-A+|G{q67dP|cLJhP8bqCHm&RZN9hQLmT<1<LyaPH5-kPR?SH3b`S~Lgj1_Jyre@IeLU5rp%3|6f3gyD9_J=oW`1QW&vtv'
    'fju_qSkXbL@6AlU{Qev>!x>j~_X;oSYfXKYrl`e)1Wq4pTH3Pf2zeWFkSDxwK2{(dr9Y<?8|6eL&aAgVyj4MmDL2+<PT(zdHT6kf'
    'L;V}3x(RAdkSw1?k>;r`15Mq`FwXJDl?_n@Won5fUOf}cxZDfed1J=?DXcIRd1zYUXMuBf>aERS#+`+$b@_t)s^``0Lmm<2XcOXk'
    '%E*Q{cWOaxib2c@AFkf1B56%p!V?h@+2P1wH>r|L;6zF&OwG7WQ`3YSPa%#8j!z+u2{WtG+mpPW!a>u#zWUPU|GhiIgx-v2BR?gW'
    '+uVj(NeYj5b!6}04xZ4Ksn&EVo7iLvO$a0vke}qU6snnvJ|tfmnl{z%P3Xfk@+WC6%@0#<d=p*nBrhqb>Kyb$osSqdy&cwBq1z@2'
    'PJyGRib0x+SP_VCqA8u^5Y-tb5G{@c-lf<Z0@yybp*f?T*q&n><4Ay&%Bau?z;#lH1M%sqXyF85p@_(=U^QD{fTJb9Na@q5&QL`_'
    '!HB9Vi(vd#<$WfqAu5`fdUbW9z&erD$)_cKRRVvqz?dznX7-u6+R%tX&_(wmv|-(h3i)30Q)lv7w9Trr8b_Q@JyDZU-He_*{cHo6'
    'gIwAnORvglMvl5_l^D9oN*03n&EDFtJ=?5hs}{;Tf&^=qCJS_uVNtcuM2kx_OcdC)gybv&J?V@_VV#~C^PWE$w^cExiFl$ISUjpZ'
    'qdJw3R#)=unXGzK)td1&A%IhL;6#>D5!~m@T!k;Od(3Fu8FEXpO+u_Ye?9x@7_Dkmx%WwUcQXg^<tJajzdozQ@Bi(WzyJGRKPgsy'
    '5B&{u+qZIG<%C2Ud@36rJlH1`s+Lg!#anEHInV5Y%gMeX+x<B<a)%1|u?9^u{((?j4s(z>EmXTuL8rh)(|8Q7!we&a(yEF&z+9*R'
    'OtkiM7xGJ)7T|t?6H$hdp8!91lctmh2u1laOQDX9`+MKcpTGZC1&q@7x=mb|(!cwr>?`6$34T*~dMpEl^qi&%oUC-H$kBdlcCpx2'
    'l1&!!{3U_-%qNXUO<z7!r4xx<!)ZJ!kUi0N7rD1$axYDlwSIl1JU<eDZ!+*r5I#x6H<9>}T-3Dqwum^BqKx{c=EaEur)IRdvbY6s'
    'aAGR2|AAKirJ65H{s{H?{e74(zy8CIzkWsY`|gvgQ7Z5)@zDqIcAHh22NU4p8A$G_1P#c&^Q)0}Gk!(zUY2jhaF@sE7QcS(o4j4r'
    '-f@tMcBalPx7(-Tu-H8g2IV6&bb^48tv7*N3{cCI32nAf%=7geagGNAKpw+XMly6~3$}}|cZVqC?Vz%naSD(eHR7Y4!sr$NhuG&F'
    '12%FQUXhVaHqru~Dw-XJKsRNj8&hE}ewtLsx=Fg^+2J(h>VV=qS34Kinp3xNAWBVzZi+He%h(RBOvZUJrN<6J^(_OE6o&MDZ0o;$'
    'z2}X~VD(+b9Ae3$F7#{%mo$}%xePm;*~*>k>*Ls`L;mvHzy3f6awnGk^nJfmE!yV##BR8-M-dq+=&qq8m9Zr#`)<YwsD#`$<)xAM'
    '92<yAmlmOPqTZn>rlv6Iov5%y5lnI68e+%d(ST+v&WeI`vmgrc;V?rh-$d|WPJ{SlHPvQ@0VDBbWmWmG@GUjCBZ@JDgjgwRHLg!F'
    ')EseCLaVw*w9S;w;>MJk0yYV~axwglc&!|E!nGD7S7^79TeT2NXtN+aZ-AfhRs@1IUMiLuUtED>G|0gcYGp&NY*{l-oui@VC(H{%'
    'oLaYKi*rvP8%y%ng20`RCEJaXz6?mZTmQsp457_*Dob?go2J;7h{-2B6_@uL(;*C3NbVJ6_7H2u7A(k^;Zq?kTWnn*cGZ-DqSVo{'
    'g+h*2Krwzifz=SPLZ64QJVAeg@sPE`_W-p8sX=99EPy3|@kI(&Vbe?Cu{x2_($y5hyx__53<7L5dgC(?e*E_z|M{QoR*v<3`;r~V'
    '$G4&>`V=Njzz0&Gufq`CkV(+8-4KXK#jMs{t~^}U2tTxjutFSbIeQtw@SArfd|CsGNHz#L3)%5LKKAob#k}fSl?>X0ng%VA;0B7B'
    'vCG%9pNr={d@e128pJaeJ=vppOIU-M(p%Y-_>%+66#s93A=uuOoF39r!rvGeG@8_hy^;HTV;Li~)C%3i0adhHnj_>5kL4NUKA+HE'
    'c)3HER=V{VP@q$shxaOKYY;Z_hla8cO)J#^xvBiYP&lm#nLH31kP|pG1;Fb*G_0FYxufFisjv@(libFkS*A7MI0-KE{ncH9Ap8WH'
    'j;d8~D+9>^ucT2Av`%4$u83bF3!yI3v`TbDD`iM@p~4%|Bh`k9>Lk{*uh$V?kg3}h8o`w5$~U79(~;EF5bH!3WN`QhusV!K85~`9'
    'Pt!>|ocL~Gl35`m*$|7C5Fs|nX16>>1399+dD9HjlkuSS^wLT%74N&AQtn_=XcaAs1`>rhfCg^|v8nu7&j3u-Lt%DjKd!4uu=|Z8'
    '{Gl@oY}$zfXd3N+e--EKvKCth<ATtD?h3r0K71$MqO!PNL0;<J6L}e6VKS)69PfUBg~_3a(BasZ=s_>QFEv`VaDb@ng|ld+7ihng'
    '8_9q-l_}!w0?p!iCm6C0LHV*Q4K2XMO{tfr^&V<*aKGGD!>b<$rik`JlP3uNvn|&l3eO1SQvg{a3<CJZuduB{jIM!WT5^7czMNmx'
    'V`aTGv<fK7U<^*GuNv(3L51HSqMxwa8&tea;obx_M_L1jPg~q1xXl*26pdjIxN(8)q@)WFGAQfH%Pt1Zwa~QzhuUK@ov4ZE^wV8$'
    '8iWd(n;Y){f0e`sR@oL1antBiRs=Y-Sy7}dYx`v15Sl#*rl0}DZS<T%U8H9+@fB?ulw!7JXnJM{DNm?nvn#yTwchs3IFjBEE%lPF'
    'xwE4>Ba^(#Z|M~J77N&@|2@Z6hKBmM<_LyC*ShXY3aQmKv-XC*(FCzK-Oa$6b%lVt@VBVhbd}w81d^;mwAhc5xE}j~76eWv099wG'
    'sSLj>LF6+Jw3c-D$_e)5{4SC@%IY%MDR$<06IEm@?o{@2m(@Vlv?p?^Rua<8s;hwtT)rz6J2PB#aUhh5iZ(4}+>2YfI4DYQW62aa'
    'Q(c8`IVi_c7+@sc)ex4#+D%E-dfOA!@W&I1R%M%f>5;js*1QIE);w-a)7MbQm#;q*tnV^8M`GCD&o&@TQs8*-5;eDv#N?r}Y#D4U'
    '$%9n3Vuiz>TWcB1gkp+=<hP1r2YpDB)=*@(SMF4aTIE$(t>`{OkI*Y9RL<9UHf>WZNs+5A<7&7|b?!>n7@%Es4OO9O!KK6be5!96'
    '3LJOPeR@E^grm?2b<Ya8_Qst85lxi2vD^tGt0|Y&RQ75Y5`VB@$V(jbP9?(z%2+bI2Qo=Yyp2wmzdywgkiE-LA2P+dR*j(ybF?-s'
    'Spe?TD&E(46ZU+__@jFdQsc7D#^TJv^emvMdNyT}x-L01pLW$P4h2iPGL=I%I75TjH~URGVJDL9LQ%#0b~Z$GQt97_73S{en0#|3'
    'd?%g15f66(q2iWe(qbEWR?1nD2>>=a75Tt~AjiNrn!?Zl<eY`UuDBVdfqdmDn*p%sEWDwtcsRX8$Epe}TI`oZq`*5R{yO+oF><b^'
    'h_?)?u3aWz3E}c*Wg%6xSGjb0*sNMUby8yK-i8{Q?5o*0pQ7}QyC2<A;Vz8K1mNmc2?*=#zSH~uM2F*NV{YP!27Z^`)u1OT+ABiV'
    'E;DZqdYIyp!==96(QSJ6oZP9fY>Vky`Gk$wnC=uhGY#04s5QEmla&zhx=+6nYF%6LdghnA>n+7vS^O70LXoJ&l&d)W@(iN5J`0Om'
    'ocaD?c}Cho&M~746UTVt4ln{W125TnT><8W>8F`>RU8HB<5{Lmk(^qe7A}X*BCIi5cuSp-GN=W3MKP}B72@vq3j-IN9`zkESQbS<'
    'iY=}O=DsY%B^VzDhOiB%#jiP({04e}(!9YWaSJUQX#_&^BeQ2&>r8q%7$*XLEkR70FpP0TI@b`I3}(zG2CFrw4dJ>-+T2s$lE01M'
    '%~IJ{6TW4kWF*GYB`gFYGpkdia`&92WIE=@!s(m50}9NO-jEC+*?)_&7NK*NI+9xz%joX%tD+R7@2d&kzv2}u|Gvc5(sc19TL3R@'
    '*>h{O5Wdo{vg<ULEUbDBh$Rk?rrBo^Dc7wnkr6b_K8qZta0d%>hGWUlxkb_bJyTCK#1tG^_E_(gbK!By3SWeEg}GR+sS+qtkj^H<'
    'AzE2SxH?O}y7Zc>O?zG|QQVdJwL2ud91`YUWY)?&_cipImP{7{!O>dtWWaoX$${3D5WkXEB=@Xhmp5nj=FvWLYfrxI&B48Sxi?q$'
    '=I`E|-kaz9$o)O}z&A(ukvDvDiEn=KBj@<$A>Z8Oo3DIwm~US5BiH%lKl$e5L^JPT*2uvW@=8xW_4xZIxbeUJ{>RU*DXHW8Uh$Py'
    'sp<QE<GkqrLqgl792TXXLa@W&hR#pb(pn2(NHE|kn;jHCOKe>9ridEX%@GQiXc<Kty>9p=$YCLfOx)*)B-dq~dYzA!j#a{=CP#4y'
    'fLGB{Os7aH#JK9s)w<=}RVrpZtqq|mSnCw<SK6eiXrlshcZ;wbs_RtCD)fhJyW%OJ->dw7l!*+?%$!BFtokI+yLw*mkN^Ak-+oeL'
    '^1BgUx!<$c@aoY_%6>cWHXtWoK^+<TIiSl?->JZoD$2{q+ECYBLG4h%T_GNNPUg^hRYs};m8tT_>V)-9*{Le*pLQsEsgtdWh$+VJ'
    'CIFf$AQ^j!clu+i!qK3g*kOWxI#-K}LlzHbO>M+jdB#R%wxa!&qUW=NwhD<-wZ>3SoQcyCe|=9D=Y|s$L=5oc(^TgJGUGl1pA~A|'
    'XDdY!)mwdrb3)1jxRlTrXMdUv;<%0Qw+_u_lCDp4cn6>1oKTot=mOn0p{#+Un%<qExvz);x+|*pYio|8lS(?c@FFzZ!6E&g*Ow{e'
    'f*YJ*3J_FqXVKpU%ZaU(9jI<E>=1{Yvrz0_*<U>ffE@7Z3jVF^8sKNq@hDz8%&)lOQz4yR%`SIN7U!lA1+C(;PyrLrsVt@vWr)gR'
    '%;+7KMFEYnMP->PA<?A#hQs1y=Ox1p)_7aY8`0EdntR0bJe8^RIBzx%3!2J|+*$>yr=?m^LUFlH$5FL)SefnJ_Z+G~7yPYFc6|ef'
    'Nu07tH{@;-xdKy|S>C-v92Sp(Y#`M~)vMG;lR>k4tZeVTXON)=4<S}_c|MgWc{!DlDZH=LnD8l|tv>9z0VO(wlCAN1xdA?;zI1i3'
    'nCG^hrVl&4h_1Ql4k^ehG6g44J7fxLpvrjnJ%<8qFXePV%4iDbuTa`;=mOxhJNuqPA;*|pN^=WhoyGJJlY0?7q*9-?A+cDTFb<11'
    'Q2H51RRMI5BRWxq)t7-t=I7n_99n{jLi>|J(BuZ#q%z40=v1Kzj;8mG!_H$sF;6GouvZV?JDPaB7js?juXo?*geX%e?4oA2P(TJ~'
    'g@(kt!=YuBmBO=Px$#e?d>u_sfly0Gp*Fe7l)j6`csk^pTWEdK%xI-fnzH*QJg~WiRw`BR4fe{Wgcwqy8STTS!f7FJ>(ewyfM!2J'
    '?!Fp5f@rYi&Iu(0HxS-8*-#VF*mTris9K~>;TyB<2lTh*7+{!ZWDPP}?@erpBB+&(q--$gHZuH~lqEKlv=4H~#5Y48+^k2s9YB+1'
    'UP1gdg@1Gy&1b404^L9fI<O6_1QhBGD;Nl7BgRPuBY>Kgc?yz8AUA=>w9NgITtecl8Ju8z@(PGuCaLCU1H}W9A`XKI2~<|RR3Hz4'
    '4anogx3Hzo*Ypm4C5!Kt&)nya<y?X$Fmgii<$0+xS=59bqJfcygTZBz!~7Cr6C37L$Yen1m}-VvY;Kl@Mcqq*Rq_GiT5V>t$jvIU'
    'ymj>ogl98d40ecjFvMUM;;<@~aW|QWIxBvM{6@o8>Cv-;1}C($=-pI>4l?pM)x53hN)+s5Y7{GPv*sPFH<@Zi${{n#5tWY1EP#QW'
    'DP67ETWQ(=(LxS!Tm^Xs6<!TpP4>+xibQK-GJ(uEF@tcE0HWwLRD4zl`zVm)sdU(giKhaiqVluK6N?LDRmI6fWt6d1F;qZAQ$be2'
    '4ts5e36NT`ifmX_{ufI+nuU258xH+2U&5u8m~)r_HDPSCRzXBmFvM8WtIwe2zxsA$#mm<WYYKkBnplRGZ>PRZj1%_78(|HE7#oT!'
    'BtI+sh{zK89z0p|-HK{8D(u6t!?JM~L{Dl4kCWE3<@&B%%x+X^E{K!LIYYJSndpd*P3Yc-X;X(yo3pG{cq^H@ZLxaU!A!00J8_1`'
    'e$Wa>Pideguvpzp*x>Z{U5~x)B~8GIT9-`;#s&sGZ=QR`0`VwNRIySk%k5UvJyPMzf)0-sHEfxeQ{_Ns{<Sqh5pk8@o9$&`LfR26'
    'cpmtlF^;ef_|D1@`g>4p%_TFn-8L$s*=VL3AHkP^cNDX=LPkpYa%H8&f(fgD+GW2T+_Y$7ATZ`CZ!)tL9wdyo3j36s`bwDI*c^P2'
    '_p=-`vsCBRGfM#zXH1`FlT`M-CC`ox?5(-h1LB@ZlY@m+Q=p25NvaOHO~tAKzgR<+`KTxXvc%NotI3XAp6>NUsUT`f;8f}|$noQh'
    'dsVBh#gQ3ZCyOle4+TRkhIM$!q0Ews7lyVAaB$De6OR)H$s*8_N`(tZ`|3H~Ub~HU+<R6Z_{X7U1s7YJD|A*f{wISc1{@SyQEdf7'
    'TDuoG&7zb@flH|{^~MfNMa&4)-c+&oQI@HuwpP=}H<Nv7E8zw=Bi<kjddAfp0_k3+2|jpV!G0Q}fVc@fs})BIK}o*Tvo!_&sV;MQ'
    ';qA|$01LOB`Y^lrw)o);08v#ZMlR2WY!TH_b%Mp>60-zXT84Ebn#5-yYOI4{!n!d)MsW$WkpN`XU>R#{rD=xq{p?_pHkxa>y!M=d'
    'm+JCc=66+EQ!jfrnWy*`FyERkSk`H6eAo^<AX9*%qv0t5&1&8kXc#ZGo4Spy#+%hM>G_H6Yq3&RNCI)NU`h4;3Sm$Y0r8w>TJ{1}'
    'U9qcB&bTK6=4T`D<vF-8EAr(t7q;;L%N`B{OTTBC#CzCY&i9$n8HMnpqWuSqPM2U@a5W7wy^PGl#~aAZqaImUG(@~2$VBO2MNKev'
    'aQMsbcN)@`t9k3_1fA$v_gk+Hn%zwX1C~>#Kd^iZ#+!Q-v~<f&f#;ZmRc}|RzpEnA<e$4a3H{Nd6h9bd&sVpfc+blJ)QMp<iFdJ-'
    'PNG#TR)DN2NvtATtrRM{r%qo@!=Hk}3Sr*hvrJSicw$aAJTA1<_#PZx!<G@!z>{Mo^f5Vuai7^PzQUzFor!5-^bwHVo)JOKw`1H_'
    '>P0;<!9ncFX73V*Q3IfTv&)w}4dRjH>Vg1lVemW2vvLKh#U(1dPve;UxtzHgJeHcv@IOuFAh+0IL+XTSo{Q05v3ya8)_`~C2#o+V'
    'wEqy?Qm09Ur0b8`bKczZ*Si*m-wY8BL5-quLq)c<HJinlL_06kr@k{PhXu-5%C%8Le9sUkG*1puvI<QIAv!m)%oJ;gbBi0jFs!H!'
    'kBgr(#0fimAA};{4Pr{94jkH^cZU-&D+ovG{M3X)M^EK7+#eZIkAhM{vA#QjKB~|Mgft?dC<rJNlw+yrEC|OA9(h6=GOQ1whymd|'
    'Y6FG?aZn<ue-y10=u&3H>I=%k&rLp|pX3trjPe+Fpk{W-c!tJ-Yh`p%%d|-yhdogs%Fc5{fy#=g5t?f~jO~J4F_96Z105zRs4zTd'
    'c;cGH73~UU)~AUYq2sZ!!-<1oe<zQob&OIsvEfKA*v&ZmzM=mA^5gIS{{Q?<3RY8PB}>&Fd3|esDW?GO{)tjo@rh7&mEqf%1O9ZR'
    'C@4mjb#*sE#f$J{Qe%xQ<HUqQjdoJmtWf@a-t^HFf?q(^$@~SM-efWa5IYgRK(lf3A-scy!^NRFDR@OMYGU#&vz?S920keohn!{V'
    'rmRZ;JllCE|Lb2r{>6d|-|4<r#ZKGzJqM#N69t*_&xI^d97()rP@o7-3BF7#{G>Q0Dc@NmX-bhyQN9&c#I6*qwkR76v@d^2Jfn%K'
    'Kq_=ibl70u6m8mERIE9QooIoP@zs=A5k4OjpbD>=%u>RzZ0Xg!jAG_%tUVOr3|GQ-Umx;Y_D7`@WFKlHjR+m#9P5~HUNZ<J;|(*_'
    'S%W&>L2kL5nyL+R((h<xUIZN81-zkoy$zZ}m7`4(0pRrTNPUF1ai=;~xjNu@TGfm^MB}U>X92<Cz$&RcCEN5)N!|=T1#xG-PK(xA'
    'sr?NRXlyWs!`9gG+!Ryvl0PH&okfJwkRsyDwgOaI1Oqw9TyG;l&m3j^rzcRIrCAN7Q8MowIf777o0jvun4aX)v6R8oB7gzssOt$p'
    'v=Su-(aN+|gX|O8sfNr2c=qaWHbnFoJj2wcxl-n2R|J;pZS^Z+jb^_jqA$CtK|<C+?CZ-6>U`0Ir}lScOUyN^M^UXw-TzLpEDocN'
    'RW1WoJGq8ND{yn+Jn=;)9?<w%L`=9i%s^D(!8BXZhEblcZqF{%4pK8CcT&^AEK&8Gh3spJ&ScP>iF<+L(UNr2wVbHK*-jbM9hzN0'
    '-sgZ^pt$lePuFeI(l#ACsf$d>&sN3Q13iWK;J!>yESt(|tngt#hK!=lz?VhNPp%8g2+8YdS&4F5gXCtlAXi;TX34PBoNP_mg?N;V'
    '!i5lpoD<Z2G<{XneU!$v=d3{z_s0{-7Q?6wfv)XZhYadOL8dH_M^mULH@+5z5pPlkdZQMWs42hVW>ZnCG1KN_&Q-QKK>mfF4c!IE'
    'dZ>|4tGmuzrm3rRWW|O-R>7+~jJIQ30lAK?gB8fgODlS;x~RkIL~@CK6aNkJ_~v3g#4pY^jRpAosq)tNO~&lQ5R#3vK(E8WeK<j@'
    'c~q1U2<8Zlw`{$qMs$jTJA5RH@Z=&XGen*PJhkjNArm0-1^wJW$!Nf81tPF5R0t^%fkME&Cu*T}9?HwrXkU=I0#ICF*@gP6o;1LZ'
    'XdW35468VoxlNcz!*B|eU1Uo*Tn$a!%vY0yqJWVsgwhGn>(OL-R?T|}^_+*SN}<kShcuoOIOo@+evpiE$*%2~mH;w}MAZLk1#Mz0'
    'GgP^B#FHreBLCqz{IiA4M2~Y^gI~i8OWoKE1f{B1Fj~y9cBs&!@_s(5gE7!dg%2rerHl^`=ZM~26$m_jJu_*YsCWt-!MewOxN9(F'
    'Vd$b1>&zGkk(EM?aFXqi=Z;1g)Ptf%&bY*r(645Pi4m<C4HJ*7A1Dh_hccs4k@qYDk3TaDl|}|oB>X-Q1hgDsJgpf>7e;BtEQ2zV'
    'GoyLCpwiK^5r%^(5}Qp+XvG#j@6QSp5L?`RdsF%)vxZV{ad6J?6;b7r?l1{f&AecCq&U2pxrEPButUI{#q1|^78B>Lm}1n_D_lsL'
    '8o&tXJ{q&Y??*kC1|1f(re++|#bPC(n-XcdL)1TnDpMzbDHu8fxvH)oQp0oww<U*HC>;&rqasTj#heDJ)hAAQUIK4qQ6`>jLK65~'
    '&;tmV7C(7EXchG6CA$W0*)<j=VRQWT?DLqtfBx~~@BiPw{3K#Ip#-Vn{@&IjI;bk#2bTeE$EE{cwLfq~{{dS>wxbcJS*HbSsQN}H'
    'h0?3hO_90Y!6{m`d^qvMHf0$d8*_`S&FscVk)EmM&Qy+pLFX1Vk!US`Afq0%)Qg~cP*qR_)3FqO4Zw)PCe-^k5LCqb;AFMVf?5QH'
    'UTV9S?(U_)duj1rs=Roo=R5pgKmO&HKmYpmw_hOpyNxvJILo)Vi|SFD`@HshKkq&AZfbe&(N(J-U*koDz3=CFXYYG_-|hRp-*4bw'
    '9i9FU--_?|e$$Wd{C@klKg3-<J@Jo!`SIJ&xTN3rjDm1cy?wDI3og`D#j4OD(07Z9V=5;qob8$<y4mkRySHTC&SZ%BbG*xt74GVa'
    'F<*+fb;Y2s3EQy>lPA#giD#;9Vz?AxL8oARQjb#bMZ4LKFV$^Q>2hhw1;Bny0Q=}MqQN6xNQzAqin*w-LT^GdI}rLSfZyY^L`w<7'
    '9%baikEN1z&a4NNqemgbHEO`>FUnoy(OfddJEF4Ev+QC{zXaUtT0Sqz4P?;xScGa}@k7B;K2<NMQ&P8w7AEz5-07mSY*@R}`zd|W'
    '(OR5TbBfy%WhZ5IuIZ{i3AXE_q1vfwwM&@|P0CJ<ySK^zr`Sn79u3K!dl0F13V`>D@+0&jyLv1PtduET8{Mp764!Em78Q1;C|xcw'
    'MKBkZMbA{Y%X0M&3QhKcA_?^}1wHLVMW00|g@%_4cI|myfDdXRBSVSnwRmNaxKdAJfqAx}yu4i|d2JU*a+cyUD?cof(cW*L7l7Ul'
    'FGX47Yf<5%+uD43?nUganr?hAma17%wsk3X+buCnaX@Vr#G&3{otK7qjuu49Cj5A4>&Of1vcvtJRcLWq>+Gk(=ncDg=2*&X_!ZXh'
    '=hAd{4`kFjcoCZSE=t@OHJf4-9?<0`>L)!)bFM`x2z0d6tV?}jQ3wQqgFY1MNUKFf*(KknBf*58S91?kVN438)Q+?<e3QMLc|eoe'
    'CEj2vWfcuGYy35-vdzV6tT4`BvdSsaxu#2V)S}9kc|gqNsN)PX+cQgePI_m>AOHEM+bsSTU%Qr;@3dh~HMIfIt%(WTXjOy6>jdb%'
    '$tD!@m3UjF>b!u2Y7Y6MB69yq`97`WB3jZ0K$fsKo94SZC?pmT0EZfotlVB;`PFtmcJibpJDK1JDB(I>5^L4~mLtlMJLIy2!SXyU'
    'r)H)@Hl#F(jgf*OWW#Dg1dSwS(^SWb)rMF3>Zly)#YQFvLMQxY6#lSA8-X7g2k8HGGK+dIm=3_SIv@f<MJq4_;-7UI&-X$I$gH^S'
    '+)JghGzH$ncUmZB1=hK%Yf-CJaSRI$Q5>>vjSLN8Fo@GBotT&%RFTq>FSsB?-}|)C4S|J6_TEBaS`M`a49MQ#0-}Wpl-egp&jfv('
    'siZCE*PP=BWsgdf06#oHQevQ}9{qp~lMIL-sFw6ZI8v6+CSdZw=RY9z56u1(djG)j4@AEK71F{<u7*Tr%Or;(88fW~)Nxo{?X)P('
    'x5E2$!8^3cK-DUx6Pe6E1?oD20^u{tJ_hA)Kixz8dsku{x1FYlj<OG?SQwxfHHUXH8B;@k(W)x8SE2(d+WL^q*5rgnSKN`zMYN!&'
    'CG(*vRl%CVA`3Y&zn-g^UIE&%>Ds()#}hM#l%9xD?4b4zPYX1-xL=Uzn&)Mi(ij)7E2l#+3UXfD+&=k2np7t*39{e4jN0!D2X&zS'
    'yG>m|kt}o;9WRWET6siOeYis%2G=-{TJ9NiFM^Wf5H&>(fKPf$BjYT)rA%u_g+y+%NW^ju_GpSm%?1Vciog&<HEm>q2VF)VVE{u@'
    '&MS^hDp3XGqEcJ%NsYqUqt-T@y;jjJP!FwEBej6r4D{PH!+IO^4PZtjrZ?mv)=EjPVFCOq?2*DNR(Rmafx#`v)CES9LJQ~@VdsV-'
    'tRUFDH0Y!QqA{IdD4JR}g~G@ris~Xm`z4KNYp70+B~hi)SWO8sy@qh2MHQw4TBvA=fQG`^1(h;lo=TOW-wcI+kz}OEs`zfguNKX3'
    'LpDfJv}n|mqNBjJDBLk>jx6FRqzm;`3u>?TZH&SgQ6aPDilTXDK&h!Bo*^V&8I(RlMNN^h?n(|8?+V!xilqysv5E^8IXxrsD^&IT'
    '+V~3kwej81G|)C>0jcHOP>9O_l*?YAbReZib?=5MTQ!x6>{>WwA>E=8uGnp%`jbp(hst}_kXo4*SkbQ<;e!J7LdH~-+T=K8E+}7C'
    'RK3`>4#%9p_mCaMB#fgW=s{6HNJ=M`Tj-s%E0w{_Ky!xdj-qxQ(d2=M%V?)n&g@m8#hz4UXX`jnJzJEX=>ov{YRVlH+@AvLzG^;I'
    '$yRUz%0usso)hw}d>QIlH7-e{MhoVFMDNV5wZD6;>Pn&Dr|uuFeJbY%Y8X`AYhk0ZHB&pKmKLI9MI?=nLM<oMbO_pMdCb){niaMh'
    'ssa*XyqWSGZ9}mH@g&5lO?pvcx%Cm9V~H+w3wlwi(?C*yw)Qp{C@9&1f#O}3-=+FpzTYMNS?1rR|HB-3O@!BMcuk4dym(BG*9`eE'
    'O+L((*MvD{&1>p>m_M&c^qNVp>GWYvy(ZRUcD<(9Yo0wO+iS*sns&$B`!E3?v+%=Ie9gzlq<qcH57YBCM_&{5HCrE3_BC%C2Db79'
    '1UTUuK_gNS9coQ0y0HowK<lJ>I<1n~;(m3Rk%n@$ZI&8va?NyIyE<N9VI~5ey15#OE*tE{sV7)|p#6l+20)*$3)?EiY{L5yuE)-5'
    '-66vq??s)kd9~l!SzaJ{WOS=3PXO+~3`5e5-ZW>Nc|jvOpfZ8nG=)hidblRDg19VvE5vM@d@G`tP0>gaPLMs4g!1{;>K3OXZPgt3'
    'M0rHd22Bg2M|_Y7<lR;ab=S+5N3$IP<3<L~1m6nba?KG0#9_$5j-cjUtEno#nWZY+Aj+)NVeW`>T4Na+N@YLk0KzbUupaB{F%^IN'
    '`S;)c{%?N*ElBzM9{q}0U+H^ygdA-#Zj;C>#EmTbQUk^ByP<fneqq(rUs0Fbe<23ZUj=&azY=2B{)nlI8_}gBcY_2SsU_tZq`c$H'
    'h2%5d_dPWBMg>Q$Wo^(b{T>v2lUhVN^rHe3MUp<a|M!8}nyHfVK{K?T*3`DRKnN`l9C3??<DH||JqWNZgvasdH^NrB)LND@M<8cE'
    'ThVn!X67`9Wpb3*Q%D2_Z=vf5EvZ@U<G=S1x1@0BC2|exu3sQ~$f^>HlE7AVkRhW}8TYJFb*ZmPZEHU_%GPNCcqDv|X8Z*TC_Dau'
    '3OG!Y%cAn={vej!2$5Ox-7yIC;^QVGK{g<Q1B+obPIL$4AY4>%reST>5n2Kl1p!Ifq>K85q~3l3OHWFFUsl!DWD4z_qypHN)TVB!'
    'n7kjERXEdSv0NhTGX4KmtR+y<K!wuZoQh74D&Aw=0o|EZFWd|z0Fz#Z+JjU+xkN(DlxBbSg>%z8AnXJI`!KyF4<B#mK;^{c2Y43o'
    'mJ4-&qVW86Koer&q&5$kXF$(IhRxN%UnwPaQNL4A)9&CZI2nNMB7^p#HWO8t3y5l*j9j$r<_lzYQNH8?LxQHu1xUN7ob&>uje^}p'
    'U1os-6R>X~=4XJQzSRX{HktrF&SS<?``p}}O!^D$3K;6IP8Emv-!EhriazLRDx?W49J>4SE;ObT-H)w?qHqR~#znx~bE67zq6JYz'
    '5OK~Bw-X(h^#c76Wgn?oN~{;iGb{@0Gnw}N<!rgvniVaoKGK3{q0oXr9vl*&h0|K>=c}D-YnV`-mw=Jg?3})s905tmoxLdYObELI'
    's!J29$Ao@|hEzqC!i0_J37DrbAGMH+N{=*A7^Y_nO?y%1o6sDi>Mwy`EAiah6sYe_Mn+p7=Q7fJk+cAajC46zBx1Henh@MFGB}T5'
    'tObF7x%LUQuMA$8F|`Z_Qj4D(aJ#SPlII56jNt_ffRwzs+C~o=L>5Z1Y(<C~fx&^;tr5~zF$QBz4BjAYgw`loIF~q8({~4O?U*_o'
    'pmIoNqq6J*_n~mlRc92FUDeg@dUX_GdA1^xt)(Ig>Msx@bHq8|1a;NAS}x9>jJBvuyFkdRj?33ds5{Rn$GTj>hM@DQBfDJ9r2sFx'
    'p*x)7mePUH%84QkFVK$MqKxMw6j#7lfv+g6?TW62(o%}T^Y|5M3NWp5CBTlxna_3dnL90zM(rFh*9*CkLU(1JwN{#|dS@s)(?k~e'
    '0C=Sp{G6c~iOVI@+l-~Eure18MS2cY4Z5RVF^c;pD7Oydu8`faZwAZ2SgCIMWY4+ja~rS^y^C{`58O0KW`H83V@H!`qyoIdDhr(M'
    'oa`XB_dIVZ<mlPo4+pS<HO)IPA|MYyI*;Xz0K!F8Wa(&ibMdy$FhGGE(pq6N%6W2Y&{GH=3f?r4aW|qc?(EcpVxA<1O3qI+Mb1rI'
    '8*p5jM)5Lo)Hhwe*(3XqDWJRnlTg}yQ5{8P$K@&oZ*>W&Mutlh6jBl-n6bH@3&)+EueDNKoEYbYj1$4oYx2U72&77jAPBmJ>MF@v'
    'hPha&Fh25fw&mWpa5By?La#6rusA!!h0mAO6_L#;swc9oSlT>hAi9>9Q86TSyaVRUqT@ToWUzikpAlfcMmr{EL2n%-UeP*z85i!Y'
    'ocG+r`G!^%XXb(Rx}HMU>(s}}>>g*1Visj$1kI@$n`_Un_e24wAp|2xiISy8(U8oP7eLBaLYzIeY&<JzbOQ915NC(gc`wQ#q|pgR'
    'PmO^{#od^=T{85Ew%&~iSm22KWDEPinGY0s!%wajns>CMl0ysi)uUoWrF+WEji~g<lrjWNJ`Cmqqt1|J3Q4TG!&gUZS1yPe)jaV@'
    '^C)U&GOecNn-sUEA?J~Xl4>E`gc%@1*|JY0i-Dub>qU-Dk){xm>ioh)Ln{`dq`=vQf)T%B0;mG@@`XF-1I|(WaZ_VTNmgzfW_=c*'
    'u*58v6m<xbTGrS_U2<Y}fzKr=W_^|RHb)A<x6|BbUPXxOvVxs50%7bk2rFV>!`9K!5WA}7+B6<1s!d99T-Bz!Ji}^LFWpWki^(Xk'
    'vMf#N7?T1rS4TY#Yoy!v&~#!6HnA#P-G|BD>z1O^EYuNyYV13#Yo{Ij`ZTanJ4+%7wDp^GQ&-1Nk#Tle#|#LBjYrb_6fPq>dch`;'
    'GR_qCsM+ByYz3*fvGja^=*uiVNMm8zphBj&q9g~l`N9RxSB~WVgCK<6<SKn!P=4F4w-$jd3s?a8P3X8RhY6IH+5-5g8*eSza#yy%'
    '7!1N;P@wR0q;P~YqxM!fv*qEfQLhmAR&KfzK~UCQ@dcvWa^&@of@@o!T6vQ_06y%mGr1ghLvt05eJS|0SQ7=$n0Y>9Vgdl=)#62?'
    '+6B-<(0|R2EZZzov2P+C(7;_{q-v}1v~dPl*_XkZo0EayT>1Fnn}7jWUWHj>-cCop_HN>m10KfggGUNI`U}~&l9}#1RSA0<NFeV='
    '7q%bA#E$iYujPUtoHs85^fpkUc8^@v;~gx{%?>#qzZ1M<slJDL97qt|aaiXb;;kP9YYNffjEDyy5H%&xOQn`bqE1scy77NmXY_7_'
    'l*Uk{`XCt-!%ZbTafBj$l1#xia&Kz0zuD=t-?Y{yElGKU9ho%ic$(@NLhjsN$sUI8XsZq*jDw6W&ys~!xj@@zVf{hAybl{;oJHVB'
    'pN{%~OG)lb`2c{9yXnu2C_3;Wf;-D!ymB9B(aws#M&M}E0Xv-v3t;&6>0exe1=*oHPDVhAAev)Bi_p{JYDkaJLl<ExZ^ee-yeLpB'
    '9;`x{VQA0YoM&ic>~p=QyJ<iS%>=B-F)wx?!sgzgs|_Mhoqh9&j-63Mc3etE(`kC_=nRx^F_iNHX*76ZUHCM0XPOUVhenMLT^P{3'
    'PsW=>PuYRF=|Rxhx5OaGi=j;aIbZXbY)tAl(};N5n8N=-$YMpsBaC3w13MP8SkX&`tyANdP&;#w;=?h{JtF{Mim3sAC6Tf^U9Gs&'
    'DU`xU;STxL3gMhXW?J^7K|G4!PsV8==wcrhA~er}l|m>w^_gBAIssqE7{}pvOZYuUp|qwEc{+iCf@OgZQXLLRyo&lX2^ZuO&f?Y$'
    'DNAR5*^U;;SD|neR2!Ts#i}$oKFs7fW^#-WPv1CIJOu^|j*rfR8i)2xmRN~LFHCV@@G$<+PQHr6a8}FCCpHBojiXRhR(SG-smC{2'
    '9eQHT=E6!wUZ$CI5RLM(Vh7(7>JvmOvbvgsVl>;cjF~G=d{vhj<n#b1&_(PG?^99E%e7b;!(#a{bq{<xt?5_u5;BT)fimKg7X?{p'
    '`(Ci86eaU7)wN}$qiuw^>9=EHn&HY8y)#efHuC0~mCb9GcssnlR715Srf^e`5u@<!2o)XAmIwo8?<=E;FaTx@C2U!6+DoRy-JDop'
    '8o`@LBjJ5iP`a1hO8k+nowu`|CX?mnMR^7A+o{xEtS~9F2q-S<XMD2C+`=qfyQkY3?<{Lhb#N{(q72)UFrBNMrKd1aDJNyaEpHrm'
    '_T6;a<z|jioVA!TezKOWYj8Z%>Nqc9pG_s~juxg!X<?ptULnIDv&rx>TECWOm5Hzw77jjdqWCAK8CyI``mn`y*>nzQX_#QLeIJC$'
    'ToUD|IGu<*0X^295(A<6bVz8mC=zE&Lwg$E)~^81m6<i1=WKTu#ug@HRC>9soWCMaqP>}GV;YY*>@+ahGRbP@Ctf-n)XmNKXGL%|'
    '_2Fkx!)x_l@>B4$Bob7ES|C(+hM!b(4yE8p^@5l3beTHW(^Th%20^|^?3=N6C^V(6^%`_zal~Adxl<K(QKgfKMN*!ciK9>wpPABO'
    'TF9WEeO}DrL7vOFYq}eual$VOsTn1DnsR(HW+0&qO5QFi*v0vAnz%TGUozNo=MNpF(A=LLgy0)m{S0L)M?dJujz@66j%cv_19Eq^'
    '4rIBrS=<9<JlViz&I?%D_E#pe$9T7G%|f4^kWShxZVq`9S|XFBvpbHHT)piIyItZjlitQPslc0@fle?eQm<}m-(q!eHLZW`&;9tL'
    'u2%-QOUYGIWF)!~DXHN!Qw8yE(d%XKd$`B0f>z=tI$~!=u$sBP8>PvKFt>X(sY+$5l#c_2&$9vC_Xr@xgu{BqMGTXTVqc{<w=WCI'
    'i)Yd~GzwE)c0JI_rYC1+6UrEgAyCqT8IXNgrepH4=$G^XzT$QeB{M-{_7>Nm(xMhLx+I(z**sZv5mt^yXG!rZ3v8#ext^GAR~@p+'
    'u=S8arJ|XE!756nIH)A%am_Dy+F7JYu%kwc+^DEChO@6iwvjq5Av~DDS?(P<)Y#S`!#OP~Ly?7b+I3MJO;#DA>_|&Pn~~AGsV9LA'
    '?=1Q2W*ZlB$rc7RM4YF<IqZ0q>#4Pz|H>0$DIWoTfFp!KIfgy&)#b$)+Tbp2sY7aJOCDU$`8E5@LJVPq55zX(u1fe6Dc%*oE^m{;'
    '%YF+~oEA=@&g@nRvcNROe!NNQ1nv7kFU)>?(G;E?hB3<Hd{kxmxhh=uct0Yz=NVXBQ}_>?8&?e<_c*22zylPorlcge9@9ei(+wxD'
    '@+@Ie=Sf7A7c$6K@C1Z}Nr>fv$YL@`;N&)_$i7(9C<S+JgX(Mt`ZIIQY4!-kc`HV4CKG4dLQv_e6n7eeZt2;IQ6mt}1%+oE=BsHY'
    'h6gi~-XL9{$)iC&mj~txk4e+LTO0_^S;kgBCJ{WkH3^a$OMomdV-N<hx_m$iq+&&Ds>6&|kSU4Xq%M9zp_eJpfFUoo@^vW8h^=Xb'
    'sZQzb1&6BGMvfV$o_<x9*y?PmE?j*wPd4>DCMk7yZvrW>5Gq}8tUvrFZ_U=`WBMaui~H*)!-52c$AE<>w@E9Wmzs!V0>HoyAZeVf'
    'A-Eqy0jfkz{O4rS6}vp=geA@yGX4+KHyJ9XXjVmo88*dhm|T0ha!Jh=&RW_8$e8L2?lZ0FY;f8&bW0PE8v?i+#X`4q!Z;LriOOag'
    'x1q_J%ZTGt5ZY+r+s;VQTAs+cBx$ciNQMY>tWn{u&l&ckU`xxH!s&^i7-%z2i%=VZvF5xbh18MX%+7h*wu&sN8GfWjL$*|yt$uj8'
    '$KmxUkpjgyofI|DDE9OxfXrz0*yX~IUM|z&`07x6S!IR0OtfPj5~9m#RN~r&%qV^Z%to1<0EyM+h9Jr8l;hppkBR3uy-?CBVb5da'
    '_@=cCtcO~O{X7w$2e8k*(8W!s<@~G>Rgtu%J8{0<zRpK_Xr1Ld@{IICb6Y0npS^?5)N_hcQPsrrbH>Q6$%Ad4tUFBBu*YAu%^(fE'
    '4oh}^o9tkIBu>UxV3b#rGmV&23f>$Djk17RaM<!0dh{OOj=xzPMoizYfz$=A$Yf~O7QNKDjiR+E-UH5MF$p(WXETNq|KDz&d4huB'
    's|4w_-{h0?JfM*x!iDhVRfFYM7)_$c1FQ+*f&u6NZ;$LIITmNX{e0s+naH%9r6~h61YW!F^Pak%dw3Gt<wVrMnr(suohrSQPL6w<'
    'fQGI1`-svO=fmfTEqiOw3lM7J>H@C14p)RVBdRRLPsUp7DBYmu&`_#bq9{|_Ep5tn1qKKs7Z!}odl2~N+MH1Q;Mb1yE6p6<>a}Y='
    'soHn~N=fMm=N%k+SyE5XM7RkQM#PC%-zJhBuAGgnIHmB<UIRo3mx1>TL4ylU(9<;kYzkAqvP^b(D5+_T3$xWt1&d5UNC{&%S8WGP'
    'W{TVLY#k>K1NY_CNBVjn=sX|JW}F4r7BvgKuwkNHyGUHzJ}Q@?{@5Hxih$wkASZkU_fp0cug?zfLdcxBf^;b^e1txFi)cJFHTE1S'
    '2y))Mn(Pa9T3MD(<$GTP4LE~I5*L-)9L6L8-37R5yOwoPPRxfm7Y?FYJoz$A{JLEKY$CsFnB<(T``KaV5*+3ZU|RurnF^4)&mc4;'
    '!Jk5vz-gX}w@YpZ6t=Mp)&X8-Ipr|&{EJdl$ZeSJ)<ds)o^KK?=w}C`GMJ$zNGE}{ujvc3NEMT=_JivX)9N*?ibDKkKUag(3t;tS'
    '%@*E#50*%EKU2H{e1t8FIK4SE$-0orMFl$gGJqrauc$J0+7}kfi&zolmsewvQ60<Gc$479nz3mcTw@K&RAV5i2e?}6r?ZD_i{cR$'
    'MPQtof|UARygUzNQ(!gVTzm_{BNjwZ$m<-Y(TlrAHG}8l%2ZVuaDs!|Ag&cd&_vQJ<eJY`Q!uNO2a&X`RZix`iv?R<)b!bvb!?ie'
    'Rr|sTOxUHFCt9}?f)Da(V(aW-wO4}M=9cwVb4n|&8MM_zlX+rt4lG^>!CYn(CzNjf@?rsYo&rm}?I4oAh$2$|VeiFHY%=u9N@zlD'
    'EM|skG<hHZ#O2*;9ibii>*k*}o*B^%dr(xWAmMF7?aM6eqzSA&QUG$%qZH3|0;3AvcI9p$A&Gf*pv1q82RhS?r;$Tf9>kRKl)3aU'
    '$_<`kus|H(cP=}lnJIS%01;_>f9V#6y8F^>XU_Een`WeGAjQ41>9r9xpXE%t;I6BRJ#3i3^<f9ms{_s?G{Bb?d5rcjEjn>&_i@I9'
    'fuZSGWe`ALL=z0k#8<4{#zP8<+vpOY^nGC<p#B^r87&1f!R#^TP3$DI((<;=O1lWG!*;u$=HiLX7)_RmUPnYfdxoXYc)Hf`jAo&$'
    '<zVD?uY+jZuLSPbXTtZ~f0iV&w_Zi4La3ID8WF*)1eRAh%+&_$KZiw(PX3Vb?<fU82*!saAWiIm2qqpB?i&6Mu{??r6Td1Dg9jGp'
    'i?a(}5GW?cMG`t0#<5@zd)u%Vl?uHFJd5$-U*{ap2^HnzAgFxV0cLUzdR#qj3wRGuQ}dVk*f^6W1S6Lu(RY4_x=SB{XfK~2?Z8vn'
    '#Z))i##$xs?lYa!-Y*%SSCi*5Tq`ob$x!JYC&9l-MssrjN1oFZu{*G};U^{(?h=G-EgCi9nZ?fMNNLiJ0DMg&zjoXbLU&(hpNF!H'
    '&=+{4ZGSE`f<H4bNPb*CAD1H_?H@VxoQO=Dkxe5WRF*4vDp_y5jt2K9jp{A|VKA?O1=7craQoaM8tf=0H|i^$>jE}pyN4Ssnd_X5'
    'vCpq$C(8pR%Lj{Ad|&+>d%yT{pjzIY3!Ka0?bWyqW8*M(4Na2)4>k_)+b|WJ=2BCVrU+{^CojLA^PSH(SxkZF;f(UR#)HP|hJ_n%'
    'XJFrqaN;@kMBitQKw;W|po^PYOdJ@2f`%B2E=7gHenN_OQwM>eJEsc33U&hc+6<{mG+8HzQC3KVA}@4cxlp#*e9Nrz&SOyr0{wAj'
    'eIfnjywKQB6b{~1O<s{2dG6uS!kecypsjP?)&0TLsjEiG0c!{*RZphQ`Ff8(aUf00_O><g6W5s(1uZ<^{j|E@eZejaznEjsnFv2P'
    '$g<p9%EWkeOamU3Aeug{<q%mh)`aVTQl1CT?q6Gk&C0~q<!)<rL4fzerJTe;-Qr%3td=Z{Zh4PI3g^ghge^?+uG=C~KGTv525_*#'
    'Qix;^&ILo*2IsuzGSm;P$Dp%KPWr7|$cv^A9W|2h@Z~?v7PUst$sQ(ey>zo^%gnGI8B1{;CxUj>d17d`G0P&hGft^OfEUimp7@8D'
    '?^vQB@FRAilG=1&;D`Vq24(~X;<&{0h6P`Vj6@#hobPa!BUC`3VxYDFqF<C`z_{CVKBZ8H?5wnpcuGu0<SdP~1~Y~XQFvzLdCh<b'
    'Dgzw%bJ!#8)(&iabC%SD`TK$tu>fztLqtW#iOm8;(O~v-hSS?u2{*DBR7E^C&{6Ze`W4jb<-iFi{w~;}bB0HN@Kh~CCCW-CyoU_}'
    'AD(M4s%<IUm*fIID+TSasf#DKh!s)-VF^QD;#e4Xi_I3wewsPQ07%8fbGB+23qTE+_rr+s8RCT8cfiWqi4G>N>Y}YgA7r8>zfasU'
    'n$t}HG(JN=6<~=~!O--$#JQpm1=Q<Fi=Dg>mU}rTKuF}?yMkgqM@4xp7I75_9Un=|kILd+%4_c3!!q8R&~ehXIKsz6z}^M3rvPsm'
    'fgf8ci`Z8GRfR5@r~7YrMbbI;R00fRhUM4m)ov7LRYNRCzJLc7d5=KKY*AXEZ|4a2fB|2|r`RJDxsDSa^skE^0h>@Z*^C*}y&8<N'
    'sH+6iwDVI-$YDg}&KXY4W;tbS1rjskOvnj2LSCO0xUifXUI)(JbxP3QVX}e;j5;Ns3>+3yFusOsL5*{xiZUBCXxLsl7(ACk6+15V'
    ';PJ9SX#1E6&#lLkkcBb~5LX?zjut1Q4~IXnfN=fn=OZ3g+8Xi30-&78n*m(J*otJet`6`IgmgE4C=NSC&cFo8Ls7$Q$P2V%1buQh'
    'Ct*8P(57uahn?ze3@=OzqR!2UPReJEI3?HTn(lN~iby?@>l<XGT#-$rr8wv5^Pb+cakLHpGaxxIifkuBPR%a)xAUnXn=Zso1pj!x'
    '1VqyBIX?tBPBjl@*4?D()68;emV*_q#OIo~<^vG7_ii1J(w;IR;+n=s#U)Oc-oY)WSoS)|@aj5qdf})~6R5iZo*<YFx&kauFRaGh'
    'dh0s3R9$QIMA4T;LAaU=nd2VLEdgP-Mu_9F+>)Q%%^K+4LIlPrd4^#;Nu1#cq^oZc38G*P2>}NrhSgtzX*qZt`#v6539qFdcb7D-'
    'y*d)mu9{)DbC6{1Q&e`$8P@x<X&>&V_c?AyiJz6B;7EN&ds4#S;c~FgIpCg|M--IUf9k5f_=;anJwDQb)YXe%3TGE?b?zCybqK*B'
    'Cc(LjBjR}IzSamNk$qj**VBkH8^U&Hzf;I8kcM-&+QVhif@W~MX=l>cv)?(2kwD4R!oS12BDUcIe4^>hhR?nSvDfAy4G`nR(=3r5'
    'OOt>cH*)qpZ-|&vG5{(+&DkP8l|W<!Apz6_vDJsg1Y~quHqcFhHzOBnH`{8U5un93j%^y*6o<uG2xxXA@-UDAtN<b)z@1%;4^Rgj'
    'aVOAl4m;lrZq)J68w6^Ml!!~BK$uaaboTXP4B=B!C68$JipV+2Df018-Ve5^MP1JN((P=`-70X3pVn!dS<SgKPl}Yjj&-Jx?8&0n'
    'AWSHg(rhx1=dfq-ZGi+SxGV-t3rv5o(|bq7REbb@J@W<ub<eyl5a>SHqgzCPzyd@w01>6q`}Q+i89aXWeRv%DDe>iA3_-6BlJ0Y}'
    '1mK>1<BQ|EyD-CvnP{$4ykL|CnRqbF%V1;>lrH#)Gn^B$P#D&mPKjvVw$|HIE8?xO?^%k?K+00=LMYfmn8iVJc5Z;Ho_*s`Wy%6m'
    'whmftS}rT<D|qK%%1}-|M?A-iPI1L{ue}rQxf(dCz?H$rL;;A8IE!X_><e<R!%14a)<6!g$eup?K0UKOWEo^HDUyJt_%z$QL;c{2'
    'D4+~b2r9`n$!-z{dFq#}A9MCS58U5c7uHemPi|(vhxrtu#J<PYF_-|O#qs=ha9D<%58MSnM`z#Tzugz4Ln`BCx4F3dv<RJN_y_wu'
    '@}@+~kL0;gW6lUW6XG2lV|fr}&tY*Ep)W4W_VJ2C*AKo9g4Qh7jO%CL;~*m5WMRG<nT-UIaS7qpYduczt7|`^mKBCFAV=m`Fe2nc'
    'YR-PgUG5PggqQHbj}(hFj&j5s&raus5>@<)#^EM-z%%XqlAry{SHyJYm-WM{1T)yL89L;Bo-dZ7c82OGEKJI)hNDvBw7lO5oH!K-'
    '^n`3<Ff*_flu+MMw(#tG4h5d!ou1wWX6fu7JDfO*Bw*J3IxQk9Xd^gg0>jty>l=L$ADNF@=0X7tQeAHEUmyK9_(<U`7)F6Ylu*oN'
    'AVP`hs0#a?!gb2<%?v<Z%Zkz@sY6k^#De_ndtSj{sIK$Mki;m9T4)D6#)))3`+j_1sKxa~_+Dp7U9-ENEVGcm`<{8o{()`pT3=*}'
    '+c>L4iZg;h1(ZEnEn$BvGV%3sp2y7t@zO19<=XAX@l|*lAD8WR0nJoz*Y6yAri1muDhOzjo+9FErK_{U*`Jb$*&kCokStUql&^P('
    '^ZJNf7zGCAC^=?y5RT6dXMbYo5q_<a#)jX~iRA(Uw;H(?`#!GRIt~>cSQN>Ea<C(VLuHR0Kg=brao7iy+Alxd^X0c_d|?LozWcEj'
    'X>+Iz#l=7x4HZ)sCGdxaTv5VhsDhDVmZ37B0$;0YGmC<cLkqrtRS{3oTsd5{bGIf8ppCSVm^kIsO*!I#jHWzpXga3KM-XJ3iL)S$'
    'BcPrP)v2sw5w0g%Vj<u`<V8%lEX6W1VX*9+kJiu~bm(8u(;26wdr{<fCeZIm`@xJ___dqGbE?7!ijASL9b$@O!{RBS9zpJ4#~7M<'
    '&DRRgsf_oj0Z=4yFyTX$IE<fM;m4^-o$fOscMvJyRK+mh$Ei>VJ!)poy!5={?i8YXsvxjv0iLkAYE_<!k`<v_CS(h$Yn}3PSqi2c'
    'z#@3ftnl659@dmwUWFf<D)}P<IRzxB8f{anCo;2BV*;{_33a)W8ngA#1DPs2Bg;SGP88~LB4eW*?A;0-l8LOYZhTT(pe1^CQ1`k6'
    'RAZHuoe2FX!ZuOJ5C_|AsINEf1+jRY8Ov(w*BavO$bAr<pD;wq%$~C709#D4TA?5^W!oWjI~zQm`qJKS=ncWsIm_8Lbru!LR^Ep0'
    'UgP7mICHl~uiqaMQWgx>)p+A#rI|?n$gi5Hr3-uJI~57o>{e*jdqH8itA&KPGN!`{#rJ-2Pr8Nti{60{Y0OwiOjZ?qOx7%sR?mq<'
    'u<lf8D<oIuM3z}+2F|NXA`J;E=!7j*A8Nv-st-T4h``@G(Ff9vN9+D%b0I9Fezj((F;Aqb$@hwclcddr6ISq^+wkTa(NaI~e){^l'
    'gN+tS)%nC9so?8`#9XQLDNu8j$2pM}5^KPOudHx?2*s;kE%!S8sy?K$GT)UPc;9LE`I_}#=7Kl6?07aM_oraoRXBqw6?GV6r|^>='
    '3kWHxrgSPGT@z0<*)}CNrm-~@EkPjY{$VB9aPRLHt*-s#Gl&;=LCdc4=#}BDl@g{HGSJ#t%q55`DIgD9?@EmGynQvGw@bXeWf8kk'
    '%vnh9B|{VrtOd1<KInqKSzwW+aB)$vYN;NH`DLlPh}mb!yM(c7727iRHihh?r4>Su?4=bAQ)0n!teJ4h-ok`j>L9FqOI-?ka5d&X'
    'X)kqV;CmHX9lmOXpj2T5*Yy&cA%@uMo{QNA>$NZtDVqs<2~k>9C$a`d+_Tx5MC&C|HDSHJje|wdK%BK}Vw34jYJDzZz+4l~c^#_~'
    'z@l1`W$9cUyWriH)|K*nMb&X@n#ozeqJ67xDc7i|_$FF<4pesqrCy5_V(ZtYc|7R|t%J&*Ets&q?fD7|)^NRNX>ee^FTztQU|Q<M'
    'MJC}Qpe5UH37A+0bwI3Y_guWT6@C@q>pR}55-fhA4V(4->b9Vne1)A#FjsFa6GdUAEd!0!ZzsOTJvethoY-3-eo~NI1qq}ow<=qG'
    'MB$z4d<9nStzYS3sQ0NC>+{66yCdz)vxz3vtk<Gh$*iGbPs+S|fno}8S65Rqj7;8!he$5}Hplix0Txd7``AJym*`>@t|b|};aZmg'
    'z3U9_?5!Cc5dGFv53+vC8!QxOp)ZY`n$X3#WLp9sMVQsykbTGmoV4HF+qd|4;N6avcx*JwBX11fjE}WE1TB3JRg!FJ6ctKO)J2Ps'
    's(a9L^fm_ehG=I{I?xok2fhfIE0N$SyS4S<W7}RH`!L^n+e9^)@LFxQRkk`(5j=PcEK!81-NP1-j^*8F-4#O6S=bt82E9|X7UT<V'
    'B{X2X->5TaQQAs*k-XlX1xT%7eTmw)GxEgn<6n7yaLcRb2nky>J!~o+^*Xl}$fA>Op?~0&+sca<#Q?YB=%w?u6foB5y`Mgmzh&bH'
    'T}|fEwlugzgcS}XXtOI6gX%-@^&>$kCkGwzt+Yx}1lzx=b8Yzg-<G1!TfsPp6JndvjI!m6uxYzetfHPH;3huM)?p>ni)Aaz2;8dW'
    '#4Fk|l;CgJI#p31c#CzEWWz>kgZqO~K3dwQI}qTr2(_>U!oz#pXaZD5BQ3ovG!URnQi{+|qzoaXt~>MOZ%gPcloTYlv{`3hSrW94'
    'oQn5@8BKB4XxZz~%&_ncbnNcdU-vkNU9BJp&*(;ComoYexymzkL0Le0hcau6vZeQUm12NmN*g@^&F!jhFN(sg%`7E}L(h&{S7k&O'
    'Eintpa<@=nkOnnc=Y#oPS`J<-HfPb?zUpbI4=Ho8Af5C|Vnq|p)Lyrwy2u<WZN-I_1FH}%t*fjggB3zONAc{XJQ^cl$@T%Tv1Bb3'
    'K~dM@**KPwQpVZFT7e2f<}BH7($^4F&1;!lfY`?Q7$3ereFZ`Lmc2ICC~ZsJi@x-ju`*Sj1Fw>|L0{v#i+<JO{Rx1xP=pYua><n`'
    '>W;p9TD);pZh1dZsw{Nq-O`$;1;g)N85*yg%3Lp~Kh?Q>SA`XJFv?2YGiPQc_OXp?<alx+HkP0)aFHxQMvAbfOUp?SPGq57As=UH'
    'vGH@DVj<6FO-3?bff29kPM%HIRrJ`;!F;N0!x}u1Y(kkrMRw<6&Q=C@Y1%Z=g%)dQMwY7Si)0*@x+4@kOPvs!sii8}ECj}aDnd@{'
    '0z)Mv#gbc)_hj2ptDlx^5dW%PCkG+RTv+Nz(2y*m(c-jU;2BiexL~l5+q_^qmXVLuM}hNds8gc}syLJirWffxYW*OK6Jh28mOByR'
    'EYKIW*oOTAe3#J!phkh_1)P^LgUa#1w#}izy};Zl*uGSwk~{XeB7;&sSP3QCg=OBY2)$5d?pW}suuj?3x_hlI%lgi`dkVK0AFzLE'
    '4qM6KbFBXXmRm;*c!u?>3ju7)lQmlopvf=Wrbv;8L3Mdw_Apdle7C5BSEziG3K8OVJ0&MSg~Xyfn{C=e`7;oZs|a)(;VQoj5Ok7%'
    'q-j$D7VHbIz`~{jMRzAJU3Frb|0Y(8&V&(Ebd=cg<rgi6wj3#aZ~537ABrzcELeHTH~{y&GN-?~JXx8ga(HW8eLzB4wWSHJqlOHr'
    'W(U{}nFd>jlP8uN>Xu$XH<7f03ugYfZx`}I1sZ2JF_i)dV9?}BHjB_JKmgh6AgGZn)jJyk-7Oq{J~<&2xDqB6FT!9EV2Rd5s+au`'
    'A6NbDd|G3<Gfj$NM87`4c<Avad2t16Cy*?853}fMnzJT(V11uSm9)%B0zfdVP0AefWhQdlG94!YX;HxWe0)w2USbBC1gUxLK0@v^'
    'IC1F{PUw;N1PVby*TwmYK(7<yIun-C@}@PNV+$^}qBcdS`8YyPq~s&WomrUH6wFiQ1tH+6p-*skVxlWRK8ra~wGDF7e9k8;z(){9'
    'lEuWdI}t&WtuyssC>5vBQiZUTYGPOnrX~=k`-zDpJ7gA?8C^r5#v(t*;Yqh-Xh|z-CYGp7n2=-h-I95rK|3%71mhd}6mZB7FK-e{'
    '(g*Ob7#Z?fZYx{MLQ`OI!tF{(RZFA3jOOy9$DfI(RKLQwQ5TrW23d;x6PRKOnn%)iYN9EHu%n*RM`Vn-ey%$e)RhA+B*MT@kkL0-'
    'B(-1<j27j<91yPT2e3neQ45UR{e8#6z`ITPC&28?ydr6;k=zyCbh%zcWs~gef$%v7w|v=M`A$s^n4+(;yM1_^OCJxLS>vJ5NGP{&'
    '8L~)w#6EF=mSwq9)H)O_@SzZfwj`9P&pvTLBWh%eJ{p>BhdqFWQy`nq_CGtvk+Q?^58CrHE0nzmoGU>Jd>7u?M;80gw(ybb)@MPb'
    'yWmQ*g?-`xv*(2EE69Yb{&}_m=GTjkYLaA+0R-4{Y@EsK3End6NqEcf62-5Ych7$3hRMC_{+xPOtD8DZ+e`tbmU%#Z`xzgnSVm%;'
    '7RS1fo5ysOxt^S4sqT2W5>UQS@w&=a%t|NXQK=|QEf<feMa7%m3akY92BhNg7r;(%T`JP$ftTPAS8F|Y+PxBQrbj<pJb8R_r_@{A'
    'Pv%XV8}!^yN$umE=`wzog~$l32>W>#F=VFhUww4y6wz2-GUKOLm;9(f=k<NN-$KcZ_B|(06mrm6F2b;;fP!rU){~56pXCsQ*g8$g'
    '<C(if_?#j3r3?0<Vb*uBTo4<>KQua38}bVVd`=m$KLPqLWLpNvjZCVjQ`63LVK#ZKwgX0GmA~FipwSxLqjL{TGNzB08#xe-$xKI-'
    ')@Rv~11;Lj)YQOPyGdP!(!PTPpKp>9SP)O6y*Fuo`9`AK_j^yHqdCd0Opy8Ib)f-pD$BDvO}z)#@uSLDt=X_7={F6zxnHUa`Ylg{'
    'Zj0Q20hR-9pyB-KLKCo?qF-k~>OoQ4K~lBI&bfC$6hL}^)qrR{h3iN~Q#dHnDGD+VG>73l92D#nsWT3jl&bn_Q00Q7G2kvMnR{?k'
    'm0R!<pi45US+kWA1NMoW$)IEgI?Pc?h-%aifG?zV{dS6ig5$t8czxcgj|Hu^JowSf2DEivS4>Jy5nx);yfP?$f~s%0fYaxN)UhxE'
    'Od_O?Nc}&=mqQ`C%L{CCZDggF6!^SL^pYZkmirFziU9#a?7OLk(g&cF2Sn8{Ik={{0Udr3JJ)Gyf$fxaDvB{I^uvpyEE|^9L9sTB'
    'n;tEt(&UNdTa3;FDGN2gIXQpZyYE*gb1i3L6mo`fMj`Q>0cAK4+Oq?LceyZB`$CT?NSOzEQC}G$D51d+TbU{{_drn}A6E0<0Ot&}'
    'ZCH{*Q27D<2J$w~R<33)Q$754IdVT&!#uG-Q(=yx8bO(&0T7yi*BC&4DsvP~&Gkd^lM9mR0q6+U?6`-_WAd@&LU|IybRB)R(<i%2'
    '^#)DqtJ|5mVh1y_*0-svp~zAc<r*E>k(|!~U0xAY|LCq@V3y-#$^s3D@rrN`XNUS+n5J)CF?mnLMfT^gY}01$Gaj(<jiH$3c3haq'
    '{Q{p5fNy{~hjh%}EeincFQ1j57@f-RMNhT(hQ<&dHVcG3ER@51HVh$@B&uaZq!~9t@kYIRpmv5PX>8@QFs~1mq-Myh4r4h`Mc6>>'
    '3?FHXYN;z-m_qcklAZPMfBF5lfBFN_xcz<4<H~Zi`7MS^DcQ075i)K1KeIrz#9OSYpdu%u$tHsl7x_I@r5ec|^}nWUevw_!2;8VE'
    'BkYGI-w-D)NKDEacT1r%&3jEwZ?V{9ZuU9a#3LC<2FeeSIHd9@gnt@|tc*@KUX<=x@35$Jxhe7p(R-7tj7QS6oWsnH5M)7IQ;B1E'
    'ml1-@d&D+;K!hOSc15xhHt!}Wg3Jq~N)E|Dbts7CqO}mO+z!`xKlIQ!->^iYd|ACyC>~Lb2KYO;uFR~#b!qWzGR)KynkXLWE=X@?'
    '4dtdj1!fS1ZM@`$q>6tf=ri=IU@;PWZSX;X_tH(qu`*L(9#nn~-fsj!vqBLK9!*gWHRbH+#?2g2`U(o_LBh>9&x&a_NJHfgHKh*W'
    'GH(j`Wo49+X3WS$Qr@;(^d<siZ8-?1Yh(g=+J<gNL%0!8^9-?5LHe8GDXhb>O?N=p`1I_L(eU%%{`m3dPXZ5pD{Uyq+iebRp2y=s'
    'wfd@lH*GuQSs;jj)ip{%mGH=li>S1LrYTXy6oBeRLn#pkcX)4`D>EwP?(7Sz0OEmKmNOL&ON(l5X6N7goS}1*pm(WnV)1sA5m@JP'
    'aj0QVP%12|1wur?4rYm&V5XWQH{xQ24`!o00Hp#uP=d#Wjr~Gyf*N3IpVqK+3`}ya=K}-{@{N0#oGkPdVYsO9+oyAeY0?kQu*?$#'
    'GTEF$G?5QOjKO>`K$odB1nxzjt9*;~KE*Qc1IiAtS8zMf&o+urOuZv$-%@}Fb5>jDdA}R>I7A-ga1zdvn55(`L6lLA(}RScCbzD)'
    '!z!_$3u47yfC{78wu7Y?nRX0fy~*rQt3H4(5ef_hTa9YdJ%Dwi78na-QUA!=?EZl0)sm6L`Vc+Jw<F9^W(8a*`}xAqWB1UqU<WF1'
    '2+UCzL<h$MSzvr2@o;b&XJk2?S50!EGZfeE=-K$jFr&p*LVCP)g4)rPKi-3yA7WQVRt!DH>-{c-{vDPyfa#UWBG{(}H#&%PTQVc|'
    'jYIW#Oow`R%#pmX%_PZCkppev(6jF|PyG1%zyI~?Pr{LYEA*t!w7->cl5&Y>i3wjPRNWH_L-bfDN_4PBPh{;NP@E9unobJJ4%00m'
    '1BRjlcs8p_OsW`QQ;Mz4ml$6bqnK1BtYjH2ZG}mYxCQ!u5fRO>4{FK@B0}96McEF}`V25k_%K)>KoCz#rA?g7q-9Vo1se-~0lqYJ'
    'd5)J-K}x7IFPN1P05S_R@fM?Q0R_WL6`@zirU`0nGZc^$Db@wk8r<OMVwvFW0FvLn2XZQW0B@xT8f@u?qKfwbkJcn+qs)#-5VXVe'
    'oAf7imvNLAa`#DrE<kKT3#?mrvyi7rG_`f}VzmS;Unb6QPN)!~xvJv^c=FBG5$~=f7`e{Arw{(mfBV-Tzy0`0B@o}ckss~fstJ_z'
    'vAf_<<P|SzC@_Vz>cFf9Zk}<&Hrr%Ds)ePk9yhdxqVF*j#1&xWZdD?xr)9=D#k$prhftaZvdfm5Fuv*72{BtDsll(xYCue_kSzG$'
    'A)|b`ii32M3Ze9o=$c(obc&IO5VFgar-BDjw@p$FLa<Fi9y|p~q#^|%<Qxi^UDQ74g7jq&BPCdh^LpF*f>fw4TrGZ;Ta1e7!dHf_'
    'PoWcv3}groco?wksk0TXiMQ5eJoyOtIMFxNB8Ugp#h-=;(N$Jvkt+1kcs7LX&C!Q$Qeqo2(zGN_ueugQzc<Iqx{f=*9%q}ZL!LaT'
    'GKaHByWR8+1EurR;wh=hslEdt57|%I*=349;ZG*ovY1<FVJ#?#L_nd*iX%v~aApO|Q;sJRv{1AW`ZdSZV9MgHq4d!gc}8mADr+0>'
    'S;?do-z~TE1<^O@aLT)7#R6oxNJ;Vav3DS`Xpok=Z)6i&4Cc@1LkgI<Rrp3q+ZOjB<)tak6F(#d4Smw@O8u08R97mu`WkwZrqxiT'
    'RB=mu-7AiTPiqM0h`q)LAb!HtSnY<gny#CcC*m4EkqEd3Fh=(~A^iH=-~a8G>wo(7&p&_sO%cuZeIHDD!?!ZK^7JF1iRD&IqtMIK'
    '44pzEa?SDFi&UIgED6~mUg@}^i8dkz35ENWE%Skn1?fJ`rKg#_1JX75oB{0u??8Ea#vG2OS<`7E9ALlJR5#oi>_^F4%{<RdLXljb'
    'CLVVWwE&t>I$PKoLU>}3CY$DD6s8eOggYQpve+qd<Ic#@!|04Xl#yA<<odG(qapKNz8y<HaRwLeabByLu$7{~`drx0wp>z&QxCP1'
    'Spmw~VYu{<y2@hZDoQL-72<1JT8w?Ic|$X*L%!Zl;KeB8=VnUX;nqYA&4*@FrI`qbL;bqBcpMgI>6W@6{0$jUdLJSC5xm3M-{r2n'
    'G@39aKrs;eI}Obj7?ue5dJj8if$WP&aJP7ReLm@Un<xmKegDG^ha51yfuIXKS?&L4>uqvvN3txvsRW{<&OH~jf3jIhDFHRAP$&sd'
    '`$dE+!%H6SSsv~Q<N(yrNTlefr5X6dbiQvH&$*Gx&U<!wNF!;qpKW{ZwbuJJ(gke(@qUdhZ6C)=3AV-^RKDB{!@S%N;{ML_<9+VG'
    '8Mutt;}c>w3EDoN%LC^79fLq6M};xyygn57683YdC3BD>A!Eg15{UpKlVQBVbR5A!&(^s%odKz7f`mw6i<GnSfT~83Y-c2ZE_T_?'
    '=u4MxpfunWQVj6(hfEm|gUDJ%@J*V8gl6OzH0jMS1qpaZbVsjOX&I%1S%Wmgx<eGf=$T@-OD1gI$9w|JR%Q2!(u3!<=ab99C0g}x'
    '5cI9DxMLGVju!WIhW()OP`YN_^NniErAP3>%OWV^30ocfo_eBX_hoMSGGl$6$KJEsmpSk2O!#Gf{4#5PnM=RSuwUoh_iQY&v7O$a'
    'T4wG<>P9$3js)%adO>gdNke)<QJkNX;+EX8D3H76Xr%sjkGYFc{jrxtR;=ufX<#pXZ%<y~&nVIMTl?HAf>W@5&AESe8&0;t<BH0f'
    'yvxU4rreI(0@v%AwE~<MQxbH$?nycdb#CfJ)3$ZPpT&mR8L~(A*w~B|IVN4cySOkUo_9%=J6U#CFxhGWVar@H2N2ino4HpmUK|*g'
    'q{AlYo#AxKCV~X5wRLFZ9>>vk_AO$3M)J6l9>yb;Oa-`NwUsgTRGK##{@9dK%e_2#CwddWZAm~$9>^^sDaRQzOpajI&#mbv+dPdZ'
    '`x%*lk}sIB<hL=Sg-;_{KHozqQjMx2K95I|qe|2l@xDDweDmY+UUbMyz=FMkEQ1%x_Zii@%z*D{^V=`q{pH&a?18T1vtLVIp8k2A'
    '>U-{q^OC4l0^Uo77aeT77@l!DZg|Ee6g6;a4!uhreQUgnIuVpToVS1AvXf6q!%dIF8}_V031Qpk&<9$_t9Zt-G(WazAzRw*)S*@H'
    'A~^+d0MRUEY~R&&dwssTo8C?+9abY^re7UaZ#UK#=N8soB%82VBc^egp4TGSC!SU=z7!P@Jmh&(c(cU8W>K<i#>ypkf_Crg1H3-m'
    '&)@y{&6xWvpYuGNf}bX86v4NNI!ok2-Kd6eD8;ZFQFM~fIxf+Itrl+s=vBk}zQ6t7e|+Ayme1#>z*N7Uy+)X6`btFxvrPdIV3Ld<'
    '<k;*VuV1arctH~Y%DmQ?94dgNCQt>yQd7=`<Py1N4N^6;eAqoVFhj<dWwYMrNyz~Ykm3Eh$G#)jGrEhSd`;m8K=RE93KG|C=_KHJ'
    'c-YDc)}T1g^9a+qYq{ToC+0m_1Dk2Qpq!eJnfyw&*?r!=%D3}bmrq#lR_Hs;wwVTepSQ09-9>P%(veyeJx-6Y`C=EUUMoBMMh2I@'
    'nfn;Ono$QDP%!CTUup*=Ejhc<Sb2N}BQ#T(*#=4TYihh4?E-C^3Ee<x8%{3M<(rAdqA`=kJ^0nsZm@DoP_5^A4C)=znu*po{9_e2'
    'rqt#}k(*Hm(zs>y6#&pM$ILBD`$DgoV-3+1)41<9+V2NKkz7%e#2dzxk_C;d($}y1T)y!oe8RRAvH(Jlq|?{0ITsH>T1%X&oIXH?'
    'TR_X8CIwBA!`aNqyRX>S6YJySSKP`y&}8CQf{J&E>xnWQ*1q1Qi_-D&s%1YvebIhv8|)H$Yc}Hdulrn3n!;sBif>ae2%~~z?Okd+'
    'XRn#wP2yTJz5Bj4HIYEq*tu0gjZN|a;#&=!1IZei5IR=6#{O&{6IcZjmrcg2mR;wwi!`H@@JcqJJ|ttefb1NL$A<K^CG*|BBG*r!'
    'n(>|MQ(_(>eu{Uf1yaO}!m{(aBG<FoX+#Kx_cX9~axO@kUCD}&6u+Xm*>LE+uGrVt^FEhx1F5TOi;Zw!#W%69@;qj~`-*)%Urjg_'
    '8SYI+K>YhnpcYNsz%0Ga+gG7AHU%YB-<5OmNEzkLRjYoy&f9I}40x8iHEkg@&73LmPc`Moz<xEXD`AlhgX?weu89!}r9&k$mJ(6d'
    '7RV!Qy#jn;J`i-duh`G;O5wAvzDtC78=qcjm3+1PynR={kT4p|mnn1}5-(Szl&0`X>KtbvE?JxIYxkJ+3x_1&+Rsl`SraaS4{Db>'
    '&p$Uzh%5H>d@F%$IajC)_rj!8e5Jv9VQNhi5PDy+uMY>RvnX$pyIPw2UT^}WChVry75jQeh{Ol3Z|zm0P3~Xsb6pFmmvOP2Ygm=k'
    'Fe0$pgpe@TWd0zEDiQp?!t-`5cdxgmBLKLD16MFqTY}<_W4GUZ-aQjKa-9j!J;DK;XVl-n-aYKvD~|ob6$hJ7k&8N94VbI(6*Obj'
    '*nZ7O_AYH`mBJd8w34i9DG?99?{W98!kTOUz4l7rNN`O_=@8J%uL4cA2y}fm=l->8&w0!OmxK`2mBpv9zAGdVS)&MWw#%8pwWz+1'
    '34i4ouXevdo)64^pkr;sPIx{2{yvj+14{zuWEbX53RoAkonz#_-{;+P5%<5&Jp)R+;TH^?jrCcVuiOHGy&e8KQc{yeh1qUaIiR?P'
    'IkNBVjJWS{-|^UE2DC#)$}U}dMqZiya^5|9=yRoo?Bgyu(_VW-?NWZ-D-Eqn4vG-!7O*INZUlx(Kjf==%<PtxwGr@cRTmOKFDoD6'
    'p^EX^%!dJBy|2ZF!1wpM+ih=fM{O#(5S(wyy>e=C<=0ZeqaYbd)>>R5O`xyb6cPeCR)+1y=x`;*dD?~}fmuhXO?E!}nD1;aW1K6_'
    '>6|<qV?K)|W<h+x700HM6#G|4>nZmGPcK75#x2i$T-z2JT2-u;be-5D<IeEUn*&m6<A(FKA;y4p0RLr`*8+gptJU`SBv+=mDv?K3'
    '9M#xNxTl5q_9E1=$+y-sI(iWtUf{c4BVgZ7da7$nC%ws@-49oO6hn{eWqGkjU=HtxZ;FDwhz&yasjtcNHqY47&ryX<OZ38#MxJh$'
    'GJ*Vwez+1;o<*>4<614~LVZ=T9DlVP+CHyo57rG7?!dr=M0V+L<O?zEf-Sp`P3xo7sl_kRuD?50>TM~zCt#C@8?zD$R#HhCrcQ9R'
    'lEac~Ng>UALd>g+3aMeey8^cCxK(SF4~sdfI@^c}AwTL6f}9=95!i2&^ap;KcKPP4gqIi(ST178!?JP-pmrWKv+la~jK~wTOKeW&'
    'P=1XSzk*BII|gUnJOQhU`h?@XvtT~A53-h<DP^xc`vx~>rE#Y~Vxu7K>}UJSj`CMmzm|Pb3G}(0c>?CC^W5*Y-0x^5&AaWjO8_CX'
    '<qpYp)qcPTU$&4Xf8`0ZH2^iV<yTm3ugOFsV;wYYsv8;`o5-38!R9L<ftp&h$Ex$7rKvs1J$Nec#k=IS+9*DO)|NSn+MVoMU|)zW'
    'T(rtBQCQcuZ;IBRa8zwhR?C!F1&h1$Ccf+3??H$n@1_e&E3;_BVa*b8Y0Bl|f=!DvJ*bjpZe+X)IA5|qC*3>ybNf)hwAvZW;`X|T'
    '?y_>5yV+tjI0D-CS5$YRqNbWP&mhd2xu673au1)`-bBDg884Au=~qONw#P)PW{za)HE~ggtZ{lb*4<-sO#g+?eD%8)^(l+?N!{(@'
    '2=EHveCG3Kq(H!yaU_F`$T61kcIiREGa#1h&;p=mgFnsqC0F4tm2EvR&2>Cv^>7|C$W3PFq0i&G=Z)f=NWVWW=hO2p(NE1#<@No7'
    '{Sc7N%%5ZHo|OU6^kP<RVO<ae-91**i@O9^^z2^Ro<Vvj1jt`4?s=!BD=Ty_dOfgdLLtB|y+|+SIwLgzJM!Xc%uJYTeHM3atSPQS'
    '<8@!6V_iBNVw1TdB}q-lS}l^yRPG@S%3kwc9=bm#rKNlxY?aE#`XsRl>vJbnCF*Fz5$CeBz8^}iFi0fR_^1gI0c4WBLx3)TBv~^)'
    '@IZBmFg`{c^j$KR57GKeTENObM2L?MK<gyVU;3<p#FHySA_++tU}HZ_CLzV+gI)Nu&MA)>8Q>qnJP1&I7>)&<5uVnM1O%V|OW^#x'
    '4xrSKgZH`cO1?b(4{e!|c*4i1n^ITrZc?Sqf4_Etx4E`B!BDL8`Oml|1#}Ms%NnNkKPNv7v<G22Q|f_>DK8K9OC$h2NDgKB!RkI)'
    '0!%Jf*|ZAE-lA8gl>yH<>(-W*?OBQqGx24}@#G~uTA73>nHtGc(jfW_w+=$PHQ`iE2n+SN6;sP5I@}?gb?j)(Y&&e%qplo{o)nH9'
    'XHIvzd*6Tf;fL@3^6j5~{O<Q3*XMIRz~eNI&+BO)(!K401TVUBx(r6?oXl&7`Cym|EpZb^u6+cu1G*e*7wy%FudMu3OpY=!ngz16'
    '4>YNV<rf>f69Hlc?uwb4uYAdC_WaIAzC;(ujNvDgoKC^h;auuC2U&_b-auBs4ks}lgnk8>^eli#!3tzPG<}?}K;~p8u<8)T0GswH'
    '5Hf{^A%h7*Xg&`W4nY|H1Wts6mE&3SFq;Jw8l4<4C6?)6a%6Wg%sqQ9jDsWq2FD;%oAAa!It1f<6fs|11<HuD7vj^9BaLFhRV5Lo'
    '-vx&2v%)GpAxEm9Uq;Vq`wSK|ptOT9BAIW4*l3P0j!E{m!0U+n<-wYaiwtq<p*28}nWBZq*A#IA)5PC~2vyI4)vIz9B4y-m$ztI2'
    '!#H6yEBN)izzQ+U1`b`aDhGBq%iX~Iy}TD7NHfIb>!uzEfSwl&5uLN`BW<6I*LD!0u^!n*p3OJdgR%6EE(b>E$mOIra14>WIKcMd'
    'R3A8KNVOX59Eg|>*z!_g0J6`*DT@+?X(WCmc+NMF8-OkH=t_&M)NO2jbQQ#TH$Z9#X;G5ooeNCep>ZV>Afse-nWa6ZPRmK@h+cVK'
    'u^>S<A{EhTK2uQWG9eQ$UaL;exu2!^cu7eg`yPC>p|~2`lw3=q!7xjM4bWg8hb|x=jbIw{wgGvn8nScXJR1tebW^&Bj}wv@rl11H'
    '8#QVEHC@qw<&d!hXVL(9BT8T>blB~wye|v(P?nob^Drou5_w&?%7$tr>~Dv1COEx@`fM!m<1kQ%w)r8tfG1E6!zVCW$p;utvA&zi'
    'Z@!UI8Fbm9I0nfZj(OxXDJho8F%_pWH57G!m@PK4cpMSE;))&xB_@^VpFB&l_R?07a%Mey<qQl%k}Nm<3PU@ympK${)eJ0=X40=v'
    'TOUlHXOPd`h-nk5bZ!tqmn*#qLQw)2K#ctCE#Q*quL2(t*+6{}N1q#_LYSeipr%;{geREVXA6;txfI623|kV=a$i9=!lst&%=6ly'
    '8?qbrv08S0#6P>}0ii5xsLYV+in<GKQmgn-jb4Q#p;XEP2>XdXq3|llAm=GIhiSGHt`pc}EtrYY%3UiMtxLEL!+5WTn;UrDLo_9a'
    '57U!kTdt=4rfwPG4lk72+?}JVBT9z>`GPztGsuy~AALrwm-7U#J8b92>%LoX*(vfGAG;(l<!YHoQAOU<173^~F<VRIN#57q+6~^p'
    'yuo&toN|&J;uD*O=B%cIL$2_85|%StdK8|V=4@Ha<`}~ihp`Lo+%qb(H;G1?o0#5rJ*m|8B-x8O&BE|!mW}-MGNym~+fP6L6WG4q'
    'KA)E9ax?IG!s5F&N})4kJ6o#m;9w`w5me~p`E{m2t^8fWO@c|*h1_AUb+;@c`ceX`Rjd?KM?hZ575-VW8*n4g3Fdb)<@PC-u60+>'
    '<_eA6PX51JppM1YSpYgysP3)abj^Zmi#vr*PuJ<hhr1))EZcUwg*(fx;mHn6ycf@Fd%M$h3;8lw!JS-!4v|O4XxPW7%59bd-;{d{'
    '<c_tSfAS}LBb!M-8t{j%_be5#7qr9zkn`yH9S28u|K$~=Z{$Y9TQVn+9o|_eI!?td2orB;cfJ_bQiT`}UO|u}jm=X{KLyqyaG5mw'
    'Yfsh*rFL?qy68gukR1lBd_0139fw4p>OK@9m3%1X3yih&9DJ`G8(X+z$F}RkZ_c&DT`R9Zw6F`!BcrY}fI4PG7p#dUXeY(2lSz`&'
    '-D6+f{M%pt@cnm5cfOD;W0ZeRww2|b@0=MU2-xx}N_!ryW|ayex24i#m>7C=i%r3EHtU-kp+xeeMH;!hj+#rCxvho}p>I^1xj8PD'
    '<OE+E)m9>HZr+`Mqh_`;+^i+KBEj1wE%&VVEv8gUkzMiKTJXW;zMCn$El^=Ov?Qm<8z<RSxqIJjmK#}DW_!KGjNW!~#Jy~Oyj$|!'
    '{D1iA$KOAvzE99y<Pja8os5)gqflIb&9-8gL}a;0Kc=WQO)X#lM>Md_=o^9QDUhvw*H$u<@M@U7$zsp{w4Qt!JKM+<CQJm*#rGBQ'
    '`VU(}N&<<gR3XoN@BT2_t{my2Gh(|vgmL=X>OI?pEbTqIg;Z7MRt4{Fj@!QfLt>dwIf!OFU&lfdaQD0|zp?l30ZZ0sf2xA0K@{}3'
    'ieL9J)wne{+>0medv0aw5}F0`klR*6FQGl?p+w8wOl5vi&_q|Mh^?1IO^?EL?o@a6rA%*kjG3`6n5szTANrbG-4ib4Taiu4iF0Sv'
    'tK7+o8NE+t^@nm2IJ+KAT_ar)nk{NtL^F3p$4NX_SbHRp7~4Wo6aJdvU4S`u>mE8T%)dON1(d=mr%swR8OpHMn(+=9ZQF81mx+g='
    'SB*}U;ysf4Y!VZ=h)iLIKGS(%Cs@PQ7Ay!9hoH&aQaxtKis)t5A%zlIZ2n13G;uK5RzG-y=az-MYesfn8a^O_X&<6Yj%)^ptaI?#'
    '4$%JDh@I{a*=q}Mfm>7|vgb7<c+YIfH0Vv9OhGSt#-dDtdj#J@{vUt-_M>@ppM$M6AfwND2`t-A64oG@mzdb2_voT0eebhSw%?6D'
    'bh+@ttm?ue9jL2K2=TNO#Ei=%xwd6-8X1#G`k=x?^QZI4^|~-5*{*i>5IG`wb;)C#5rZv$@;qVL3-6_ov-rR)?S}-otmo&){azs%'
    '5;c!crwOE&0c{tGM=lAtG@Yk=0SMZ4*~tmPPco&jgv!do`U>5Niz@Ij*Hk<dcqj4I04V4%0l>oHl1>UhX99NxXVxbi@7fJvQ72sJ'
    'f-iRj_Q({rR#OjYL@wJnN~AErtI}DO1o@M{BA#DrJ`5!ZkXV<Aw&FV31x%4}i98F(iq{G17h)nErXv{w*qeHJm-p>?FZKiy1+X<{'
    'r_-5Ki7<7=0cdolZW@%8MdV%zTB{I^sOTFW5=^usEk-zuH|fNakhkdBW^jbHqInfN(3Q)L^0&|?Y?H966>R7ebJ3U^1q2ml3_~|{'
    'hF3KFI0r+e8c)}5Bmhr0-Lf}b{fT<Sq$K0LyKtm;LguM)H}?^ffH_%$fwqq{hZfw8^htbQ`~Q~XR?lSF^EX~k?dOLB1wX5onvZWo'
    'h6CzIbAc(~Sg;I~MwY=I!zK;1<<NReAa32njQPzpb$Cp9j^yF!9>Xk@&_K)Ed~CdcT^DxF6UtDon7lbp3LlK;Efh*O19V5{6qDNb'
    '5&Nw+4sD=6Tgx%*F0{66HMl_R@wxfjs5MGszn0nEEUP^}aEUi$Gh(!e8}&1ae`Pi9rhEHEm&*%Z(#-hcUR+`(nndw&zV$xF{Al8Q'
    ')m1DP&TyCA;tzM@{ou3dD|om^FlE1njHAUhpJVP#%-T<H)~D|r1SSoCUpEE4dpBkE^mH8lijx`lc$d%xycM>WBpVM>#k;uM9vTe|'
    'C%hlZWqFL}V`tojg&SsyW|f|8OHz8M&B!FBHYCA^E(AjFN3?|{?U0v>F0e~SJTZ?VMaT;27r(Zwzd6rNm|w?zH@9_Jr<oUo<V!Pi'
    'iSHt1;u79i-=%h+rym>(m&UyuPVu65z3C9%-!7-;|7<h3g;c;erEsa^QCHXsN4i81IJ4k9;7FJsn2vxR=$pI5%Ibz=kEh>`;Xe*@'
    'N-4-KKjUL(0vP6G>#HXi%0zcW{{{CO!ED1vM6j<o)qqe9$S$#%E#MgS!E{$12Jx;q|8wMUUys)<3hF7~obji%U2_1Dhr4fl{?<cp'
    '_2kpAf?ts4!3Te)(TDy#{2HDNeEKLmU{QNM$2*qsif-KTj#mWa9TRy)N$$AGE0Xext-PWw@A%6rBID?sJ4Um1mStwm5cxU1oOREd'
    'Qe<T+an{`Dc~u`Gx-chPFYn?Jyxg(pFwOsNluzBz`zWi2tQkea>s6U`4~aY~mfTR-QIf4eN)^mn&rfRnu<Ah{Ei>A)GlV@&uh6E`'
    'qR$Q0!Rhtydh_B_=I5?LbezUr1amfJ5Cw3DSMl1EG!y5U4r>?(<g5{<0fY=kSGWm6p)-Y$xghLf=n3|X_j3?-F7h%?-Q9PwZO)Hb'
    'w4sN{5!tV4imBGO9FDIV?h1!2DxN=AxUT_{$hwr`wOK}{Odh*Jww(g|Up3qn3eZtaP1M|7tQ)~I-R$l=3O>3-vFyuIi>^<xdAdji'
    '>=~jF1UI?}eK7M3+o03UxpS9cMxHzbYC4HiC(|yPnLVx-ODv4)sWUl?&}S(SYd*54Rt_Eysw@C((WRs@vSo)G)qs$antg1R0$IM~'
    '2&nQwEsr^2Q_{W6dBwRz8Fxmo*m}K)BL3N1N>*sJLu#Z^G<rNH_JI0;+q&-_#hhMtU~CJTNPb#T+4X^ah4T&Z&tv2KRRgQOB`GAH'
    'r!JVR0xDS8_RS<9+jN<5VM(P1zXFLjb+>8i+h6eBLjBL*zW?!;$AA9*yPtpl?(y9(kN^1nfBW^jA8@2R#{7Kls+*qm&!ZnM;BjM('
    'siDH47*2#UYz!V{d3&9(`%Sh25*(N;sP6Fyr<fhaWI`>PK~L^Wl87lSo17VX{@9(oE4$y}fBp8u58wZlr~k)afB7|e8|(Z8ASUZv'
    '|E$mHln2^QxE1cZIJ;ID?c%puvt$olKfa!}r`dJJ_H<Le&Jw!Rl0S6QWdnZ(<{R_$b<N*5`G5TO^Dp1i%(Fck_iD0)e;%hXsT+DG'
    'Pq>^RlMJc5D3^|s%l8R?U-y6itozTu{q2YE{_?NfpnjfcsXYhs`n(S10PY)@xUih(0RE_}OA*umQVDdQaNpoR|MbJZ{`lM9{`&oQ'
    'zx&^wyYJuruW#S~H9pVl^SON!*S<cf0kh_2tqrPuhO!`h26M)JAuxFc4=<yJEO1}1*^S?2&5@<hKdvu-xb$hOKL1ckS)Il&*WBp%'
    '@G8u}--{>BbAGv>2Kgv$e*I-%^LmnWfS|H3SH2Zm+RN^<w)-}}gu7OzrNAwzAezl$xvz14`6JY_!DwG2Pk-bZPaC^}{RI1wntIIF'
    'ir+_!WKi6K?XMLqIxenDe7O0Vk88y|*w?=60x5d6*EQwXIB+Z2KkSq4i(=^>;R4=Z|Ixnu(f1z$Xmw@2uelrcID!uqFE1rm;O?g2'
    'M6A8*V|$fz{`-0nzqzJ}_3bHd1owT-)pCv)@3_kEYhQJx>ul!XGv75xes8|QK78bZg@5=+-`l_36};M2IP(>rkr~Wu%0>3K;Om9O'
    '#?imee?8HR`!%<6V=Cp=ecyAx)P36fO+p2F<*DlCBjyk7!}HkhxKDe1(rh}<d@J{V$C+Tm&`wMDd7XCOuYDQDy=9-`(;5SN|8k#p'
    'UnkFQ8ENsc={o$**?Vt(>ep$n>oSb<_LAzBDf<8ZLlThpY4=?ApMLuFU%va{hwmQ$@a>1+{`RMzew2Q6j^`|F_Vm-x{Cs|1d}qn}'
    'gMw1xR!t?B0kbiQP?!a$!igk|a>ADIUE5HEF3!y$TQ*HF-h{G=V~bRn8g$CiF!NoTCiz*uvct!ullQ7dIc0{h7%OhN2AdJg<Z3x}'
    '_Vqqrj^ET15?WR1r{0Ik4v*&y#ctlkvN|kVj(dgD&a%Q6i9FyJhcRTm+aZ~Onz&j-Rc(}a8z#6t<3*5>ZN|5N8<d$i$NRf1A?WM;'
    'Tt6;Hk?JTTlT%>lmsGY_$?<*8z9$({q5FN7%gv5WS;F5uOZbIze`6(S6X)<MA;pYlupaAoo%<eh4-$z@d-_4~+t?P~V;TW5?_@eG'
    'YH<CImm0YLef)fvr*+=--|Z;xb}^*2!pv>kYd=*z>T`vISZ={p?tK^KYf~C?PU^Lue7EKK`EC-hDhV$8ILC9X?a}XT-IwRN$J_sc'
    'k@?r(ef#yVzx@2|hiB=#Uw-%P$6vnv{KsFgI{*5`qWJ6YKj!sGripHy>$4s09Y&q5tZt~Md@;MomcqUCE&?8#uBiP|<+pz+=CU&?'
    'z|W$h$V)}{I-()W32JWSTph>NdQrRRK7;Zc!*lCa?|2rQ=cc$`#co2Mw}5|5GMi42Wy)96q2`#xM{@3cba<{cjPvzmLm$4CA@{QD'
    'RsJ|l;<?3J;PTmbn^)vc9?7<J?dj4_YrAv;Bojj6ee}$Ym%MW2Rb1Cyg9}%uX{W1lhv{sRjQZH#kTfdaLd&GfT75B#bwD6f{*g|Q'
    'WJ<HxVd0&ZkIu!@C{>4wa+<o^u~-yi$p?7%2RoYLEcr?k+KIa(*g`zmloUv3%hZq7Pjyu8%i_Z@n%D_G59!(rT$tDPEaIb=D95<Q'
    '``o*)|Mq_%=OCLkJhbGq9-n1Q{K@IWax6+;D6ir!(F;Wp9cE;gE3}Lf6oxRIlNI-xrkwdfO+&-*ox1L@_a?nowt{)cBZ<|o0(^dD'
    'x(04IO>o1Uo=!veQ)N;Cj^I<MBqR-->OwwGA%-R)oP5aBW%0h^%}p^yBAJSN1P^iC$8eu1d3&GtitK&=<Cepy>$9%el}Gpl^9V*J'
    'BpKaIV(AsaVm`K%O+fhJ8XtG*hKV`nN<A=VU__Lx1;qzJj)1i0QU-_q7f=vV3qmMm%U@m49aqc^Jete>yQTBI>jOGLbb1EVVm@@~'
    'ffmgWmn+00gbR5D1>TvWr`mW-(2*cw$(Vwwj(e;~E1hG~X$phQF(s1JcL!jr#Db+7ipaS|i=yo3^G(~zW960yIycOD64k8u{vs#@'
    '>>?sVL%K@gs%y^|bvSD48DQ{aBB|4f<*HE_otLoCOkz@+TwAwYGE0H54Ew3BqXFeOww*f|49q1?5;Zfv>aM;Kz7yiJc}o(D^Bypn'
    'OhG|xNhStybs|;)(2g(6>5_UB5DH{CQ$0UsM6IprT!M?Yvh9>nUuAYRQ`#ZnKF}TV0~B+p4l`kx=t8M4P|a{BcdA~Je5wk4ZY_y='
    'spvp|zLZ!{U)JP!pMW<KA$llU0Sb+YUg?v(hlt$#3RcX4hCiaQCsX62fgY6Mttm->XiKgYA&MEIj7d`<i#TFaAdApW!?8I<ZwVYB'
    '2Slygsgkf?;gAf1ym_IT3>O(ArHaTgd&H4YB$ET}l4!H!2GE-dS~A`&xD!`RuD~C<uW)*l&Vfn%2ncs(Qmn)2NKjDOQ<>1B8E_{f'
    'Ss*8|-%LtoR|n<^!DRrNL~AeUQOH>ql1;#MpXXBv2W2r5kX4;-&0BD~RV~&wP=cU4rOz!Ab_jACPTKrRI}1{+PP*-k?&LfoMj(Pg'
    'ub?e?MdfnxX=}0Osqg?-Tx$!Pk<_$H{S3VCE)k?2TPstyaV<vp1%h#3^_+{8vDRXmQ*+K)e0`<l1ZoI@lUlY%Hf~m-V@lcx5>$D{'
    '`HClqlpM=Q6F5kJA*>tCn5LM6s16~gjK&u5-*}aFKaKF3D!(Eq$jaR%W*_CzR0D2~h2xeu>wa@}YlOFxioocRcE?ec`Hx_LKzjvc'
    '1mDjn7#r;uhEEF(sRrtdz)=MgKBZ$ZU10>;rVPmnp^vglYPHT081xo)d980*N=p{zErFL_f!UE!aR0i`Rc^?3bQ~*<m(uaEHGztv'
    '1di=9Dy4TR*QO?1&ol7i=IELMVQNZL9L^bM)!rruxV}oqHb`Ez?5A_oy?d*@upM{F`N1AsodNpC>xzB7Z(X+FU`FyTt?0aFL=i>3'
    'OSxiSud(7)8X$Vfb8NjzH>uFAM3ryVr5*CC9X;UnDI}l3|30>_c$+eR`mf*r9^r}4r`6UIjQ>1=AzrsBc$z|{*P^Hv`cOv$!H#-o'
    'Juclr>S0ihv(j^9cb&2dXa6gc?uv`l{LK6K@*I6eSUk^i^2?Uzzqwe~wbMLP)``MCah)L+J!OttIga>-EbwfBDaXI7xJaVo>{>bJ'
    '{XB_*%ap%P_v!LCulVSg@Oj5uQ|3$i#4KH>`J`e7?sE4Yk>etzbDeQU+eT-+Wrxq&+7`*VefWT)o+n-@lkKu}P^#Fl<PYE1uI=E@'
    'IdwaJ`6tl>-Y5QAaTF{5UeOC{>G=}{vBL#qTzq+#JIWBkP=g__H_x-}KUOry&h&@<M?R#l{Hh?W^`(xxC*VV({5t-n>M!Ttt)1+I'
    '%htAxwM9$nVL5E!n&Zc-Rn@RqrztmAdAIOBp08k3;#L1E&T{RS@?T@mYacptEQgNd$0QtcU-PPD<$JbhwGIC`KHk6N8@x_?U027J'
    'bbH*dc;BM!y5h;tf7`9UuX(?L*JZT3*DY&D0Q@-8)8Auj?Wlo{?<DEGD|ppt%6G_V_jtRb4R(6>%U@1=^(MX5=Wa2*%=>xLfnLAt'
    ')Al!CX}5g01TrLv+z-A-gKy;bf1SUt&n%M>np~BPg7EP!M_D}eiU7UOzppp<3H_>~-Z8!3S#YO%Ig{Pz-2GG!HjxxQ>-ir2&Zxv|'
    'C})PS51#GcmREa^!c3*dmw~qx{_;mpf4goytgqL++yMXQFMt02r+>z=+^)}q4axa=wCb+Vxxb-8f@`|L#En(ZSY#yR>KeJ^;3}cT'
    '5fR{U!_)vs8=h4;-!SEA?nZhQe$=J_G8L*J3ljt3LV_46kd@w~GVNS7QXs(2wB6Vx;7@AZ<X;{`1!@uvg;bft69!|pK3!!}hz=EE'
    'K0;i>ulKtMOKKt<B!*ClT)K)O!fZ+iWXd_&L8j_U`O`$eNkvwJ{cH11xR>5aK4S@r%$INWFRXisx=fCj9<B6{ft*S7XbL<*b}t2%'
    'c_8gH55KpNK&S{V4VCJV(njrLK}Sln<S|KRdJ^@;_Xq^&!8MYCW#&f{i6pvD8U-1ZNhzoj<O{Z8{R@Cez3_P@=~!0yUQC=8q+;Im'
    'b?0cBoT#!Y&R&y=3x<^0m_gK$h)O3l{}js0PTX5fF>*_ZF?N_D*%o(@CYM>g7dp6PQ{4q5Y#16H+qj#EdYK_bwuDGUZjtN=^Et7m'
    'yBcgwD3d@e-bGdma5S`|<P>6zILcQUp~R5kg>0)P4Wykyc0dq0kfu-}JR2kUM=k)4DpR6}=a4)>kkm$?&mm5d!*7qld({4qfBhxF'
    'pgKO$mg2PYb5+y%=4NOG6&F6f=Y+1vhTM0GY*gYnBfM>(&eys52o=C^doWHpyvUBz{UEfJu_?RLII}%{UaN~p2&`OcBrpwGOxUsx'
    'Sxng34;El~t%iJUIIf2LC}!8m*Cs$#W>xY=Mi_XkPbUxwZ?bOEMRy68r_3hgfeaRoO3LI2%a6xo0_90%R0C-?u5k8PlIa5+J)4N`'
    'M!Ie8zj1|gtb?-Gl|+4}!|;5Ue1g+sW4AM@cRGs(9{eiOoox7ut(JR0qn9UuYiy|5y?hTjGkssSj?1@Xi_RoDRv639{T29(L)IWH'
    'wxMI_{^omYB(7oB?G%u^7Dw8UahBen`!Jb}u-D)<_Krm6hFp7<=a$K}%6W0MZ06EG0N4(>Ul}Rs3pfDM7m5poY(1x6_PA>fH&EMO'
    '`+m~>+~iHkscMp@oHv4=b>}f_3>|<(+lHLAOhx+`c+ofyN!^nNr`lhA0}%Uo%JR;-<RHlyIt5vC_I>vgY|b|OvCOo`s~y#%al7tY'
    '=$hQ;z1P;C|MizY|Eyp^J3s4b&R`jz&+jL<OxD*gONP2)<z7bQ*;CnU%IIkT%Oxnx*)Tt;9Vp{ef0ZH@Q^VI;mJ^a22wAA62{|$Y'
    'aRtlBrQve(3z}HhFn82hGvZ(5KgGqamRZA{UWgGlob?KRgojw}(;RCa<Aqr$a9x!V9<ej@+x;+&kCr$@qXnqC7d#QRg6z9ceP?08'
    '{mr!eF7BoT(OCjERkDwnHq8B}S^H}P$H|&b-e`#bNrX0Osfryq_X6MLRH;GJ3Y0&C6%Ba)z4X18!}k*TW!Zc!rSIkS%aVI9!|$c}'
    '>vH{G!aXmHlXWlf??wNMA@DX1UIxS4i1=byyp4^w0rECVUWUrsc)5FLQgZ!vQ}(+Zwhq^037Z~<w!oYFynU4dfaK;Hl@W8BTyBri'
    'bBBC!eAD6EUY;k9)ayK+P7LoM4ZXDF%KFK?XOj5Y!eVFks2Miu^Sqwm-w28P?zg}Fw_m^g^3%`XfBWG%NYBs9DdAIKv!nC0C8snX'
    '`Co2i!Bq4)T=u|Q_!L|PNw(}I8mOnpD*V;Eo(-3mZ)|-*lHJ&{jWv5g&u%E%4NWsu9>5h=SC^Ry7gdJi3ib`8Sje*_52;(JWZ%#`'
    'zq9dv_?K^g`srs>JkQVbX-kP5{ha48&azcJ|4!lYrTh3r4f3l-<QL`0x1Qt|mC3K#labi$|5m56ZNToSNU8}s=MvFdQ(4O(;}+)z'
    '0-r1df${EVsK@Kca&Ck-Pr3fj`xZ8}0Gz)g?f*jM|04MRLi_)BVgS2!-qZd+{p&yff^mjW@Gx#RKanI{H|#I`)2yeh^!6%US#6s<'
    'messu!*Bj-xJ*AEACqG#8EcZfS5G2t;AR{{%3cX63Cc7{)%>*_o2smZjLcik5udT4lh8$%Q0Njw!H7VVp-GX3>5&IgB#mtB5T1&;'
    'pvfCNT}GlBSzDEjbdy+9qI#0H8^H%(urcl%&iF=DvB~TwwLdQ5R*w5u_74q=X(NYyAM^QmAn?JB5w(voLz?0&D<Gl>4cBo40@=v5'
    'Z`Gg?YZqYy-=oz$6~B5TUIA075lU!+*nEs;3-N>0nJHa69MzIk&in~eMa2MYOv;rAXEVhaw@zaMt{B11Bz1gIt!wf5YxrZS85caJ'
    'dDB&n=g_CB#9bTUwTgk-szE$HMKy>=V}q)$ZjbSSSLTYw>RDNEe9MZLwZQ{ZF@&1Q-ML2+=Gm*O^}q|J#N15Wx&rEK4EXT5H_<k%'
    'L~OU^6>h_zmRz)QVbkFZAiy*fzFTtT!*?q`KSo9+e$s~Ak=M=BTjhz=R&~1_g>80Q+HLaH@3HkBkIeDk{mLvy*7eh_)7|p<-1;cH'
    'unF@NGa=<@jj#n!5-3$)C6AdZH7t?4;T2W!J4;BH_^Zk-%kx0UTTAkY5x9(my;7a9em~eDBWbgd<j+TC){<2q{z}ORmhQr)w2lrw'
    'xe=BsF}uh^HKjMrf^-62tV80}D3X?f|2Q6Rk%sZ@b!9qF6LHQQHS=Yu%wuAU#b&Xo#be@$N145`U!Fx%niM_NZKs0YX_9Mi(fc{~'
    'Gn`3NQg>G9x=aIS(v+gkWAVsQ&~L6YhY-6QwV_;G2=zRgE0vj0Svyo6V&{Vh(G2GASrqU;R1`8}u6a}}k7BCECu+%C2slg)Er`Qs'
    'y(X)f1Zx7apoa4ZRbQT0`aa)nZYqIi(OcwPD^undnA$_Vmeah|sv^;$^Y?1^WhIrJ8$ZuwM^K;p1a*~E>2eJ=ky!yFA8XdT@yB_$'
    'eTxJV-P;$zhTAXKa|Fnj1w@r>0@!zKLh$x}aQFrkX~MIXu7mMxf0C`kz8T)E-5=GWw3~bBQo57^ICi=bVUp-O5uWvkl*AO^Hk~Ry'
    'v0PDLb)_B^>=lDzY4sclk0}7m@F&G87?4mZI=t(LgFBPJRz9{>YMQicT7|ol9e&DM!^TMKoMDKdeM(@bX}m|xr`VNkMO(LDKDlv='
    'U6Kv!ZoPI7%(8P)pVo*&SbZh6VioinE#LDx<m4b(-e-*5S5QfJX=>@Dmi?A`wiVYl(@?pjq<m}M*z2|Zcrx{^<bm-0_O<)zR0LFR'
    'Ob!j2`ckblI#F_W_$yySfwFzY{dUo_^I^xvSaUQ=dOmnid>`*~?_0C5s;|Y36<acscneKEdP<_?D2eKh$`q4xt`JI)#}Wm=nW8~X'
    'pW0R{p4l^&E|nceguO<_(xp_I&*<BEnqsxK#_?gTSngKkKJjC}z}*fk6HV)ryEMO<7+{Z`A!R{LYcEunBfh3Lc<sIV{g>bV_Rqim'
    ';Ro_pfA^=K{__2g-~aL3_YikJ*6}$<c3RhGdG5m6`l87F62|^2m;Dye{{Km9D=Ga|hWm>+_ZP|TFGAj5<h{R&e!r!_=ep&mi_l=z'
    'qTQk7C_E^|v9=C1p<VQjXjwB3lNZRuG2IlBs`Z%BOKKLC%NC<}DZ5MiPL*)wJlRLpg=R#QOVMV6-U(-!%awPVPH7?|rb>7=$;88!'
    'HhXTWY{}`lYK40OyBS66On)m}@G#Lg#pS+N9Qktl91U{Qt)n*X%kMA5KJs)LG<QEIuw@K1YuVid1}(`Bm(yxVTzK8BZfx<@3GS}+'
    'KiGYq9VvQbPxCAeq0f$I?t1HyvAfa^n#!DwU`_iOHxk8}9siB*jI?%rpHOdc`&VVJhxo7G|KX=!k(BTC^SJ@ivsFHyJ7rzLo2$)('
    'GC{nf4X+5qJ1X%dQt^&nydoOlGK+j+UmQVi=g`Yx^mZa)t(7VrMBnObz3DKL?JVN6D-;ou0mV@j7K_=VeHn+ANPO`Gd5^Jafh3xc'
    'j_CE4Z`HP+zWXs*y-y?Qvk|@m(ftXw?z}M~@DoFoXOcfbh_lJW0P^A#SJ>EQo=yH=2pUf9(W=~dGbf+G-q}ps$Mtt=P*oXFYe<`('
    '&3;zxCuv><-Bw=gl&wid_fvi3T0A;~_28(>1dVzsVp-{BXZ02gi?e1Sh51kUI`a}wHM8p_m#knEoCo^^wzy{%mV$KYS++SR0d2^r'
    '4vNbg!(1(F0>+P1nN5?+&KjUXflMp7lQQlaiZzG=p<%XV*OXL0$E&!m``w(oWsW~xf&{&sE<a)wPR*mLjC&+e*}0vPP8G7o@1oBj'
    '5D(|m8T8_^n52)&Wzp*8#$|=yjLXg&2YMD0UaCZ4V#SI@mtHn3XO_fjWg!5u9gb`3ww`iHi*-M@@+;>afuwPZ;J2WE+-5EN!t;m>'
    'G~ex*w;J<Z#N&1(m5A?lMAw?n61X0)p!4W=xciJ!?NZRw?1|HFVNL=*i1RQ%2ESOgBURDNSnv2x)^+9R`H}R4inDPAh)6g;+lGP%'
    'X@&Lzex?=P3yIhh^p`^+D89>Bc2Wc&NOlo%(O5xXvfeF`1?IVDFIKU{*O~3ebDCLU2QO-rZtek2aNuCrtTRLgg&Kr%@%`7CPWxPK'
    '*UHd)zQB^_M7)qz_QQ-yE$@Acb&id#TLNuepV#tZv0|<@-^Jv2fbv-4x(>;?g@>{4$+?xi$O<7kX{2klys;GV?NyoSC9I=M>K9kR'
    'o!7)7cb`ys94$7F?rbC`ycw|Jw3yGWFsqHqTWn&Qi_4s&Y`z&52QTx{iBBzanH0X*+jN<+rxbBaBP&QY){o~}YV&!Ey7o9;fz090'
    'TC=8e9Te;9$xMv7d<HW&1jz(UXy*zrkf`JuVQCwv(WdX5CR5D2|8BC1e!hxVxldqOU7@(g5L=QC)T*>?aAnW$zm$4acw&v9<)E+c'
    '=MBJGU4|dXb620QULwzcIEmHWS!1BDjWtw8H;Go)xsMj+>ebVY_?jxu7Obz;!$Aidc^CaEN~zZ{(+!Gw<GGKfM+LcO|LrbRL}r#A'
    '+kBxOGC-dp-|gZdowiw-moD9W`7Ue>H|)|UQOPJ?0SdANt#B3=(^@iX@Q^J7N2)GYJtnQyg0F)#B@~A22v&1{fSIu5qBMtf7hzLa'
    'x^xs7S}iO;jI0&`17w1mY(`KLt>#RC=WIU%l;Zm`y28F2nJi03y7GFf$<RoS(-QFdG-N2chasP6pTPEdHBpta@=KXvz<I?GSiXgw'
    'i?q+x>`{`TwNN;D3mI01H}>7SE!p*+Fy{9gS8gokkB#=eG5hG%jvz#zCsfBVp50Yv@Tc(4_rOn2(vCbzWoq752sW0C6SaHp@EIB@'
    'N{;!iHif`Yl4_Sbe)A#r-N<rUvW2Pjw%qYohQz{c`cT_*)euKAoa#so6zyC?325<;IL=n>p>MLNA8PoJ0hB&3D=`n~Trp8Fb=+(1'
    'moJ&FJl7W@T)FfQnG3z#n+mL1*^;IPpUjn^WL)q#ZlF^XS*?Axj!hE@K(OZ-03HVQs#bu#ns?ckk=nl8v}X3yjaTJi%VkXrVmLXO'
    'VFW2EnzUT~EB3=WY;}T?OObK&GFDML1JC8sF<_Wt$?i}IWV@qzs5-hzsJ|dL((|gi-hN}e%Paf!k(ak9m7o5DY5N`_|D+w{3q7Uy'
    'bNCb4S~}q>$ww>By;GeDS83Mu{l=s|29~L(e`&Z>vT)Yn2yo!!geg!mC)iNi(=<0`QZeQg4<vMG0bo|}v+HmgBt30u5UF0ej&wP1'
    ';gV$fU96x8KsaRz{YYhO0cKpEO_-Dx2%DFA#gGL=B9TLIS(mL&2rU3t9sJY0@QUMiA0@&uWf$E?GXkuvW{3y|%T#e~WMm!UOk>ao'
    '?&)VC<52f@5I%CRRndg-fj<n3O%rR|A*G^@8UcOrtM6h}0_F_b>S63jNF2m#07fJ7B0P>KYYfQg9==s!q+VvxT;QBwGHzqrG?j#J'
    '>=6SOZ$aX<xcLS2>E;Ki!gPxx8z|Sb$m8rH000v@jjV_vd?@`lrD|$J6rr~mp365@(RU6RXqmaDj7@5NO<9$ckrgENQ2kbn2h6mB'
    'Ja8(>#+ei7lXFOMOXf*90vg$@)7^Z-g)rUYo02dimLuLu0)@7OkX;@XrEqyS;4NBA>eEgI<Q4tgMQ@SUB;>f?jb!;L7efP`xL=4*'
    'Hl>L&L1N0&rDE599FT%S;D{vf^5UdLwz7^s%y<z6_b}B!$jrmsg(=H}EEkg`xrgNFqH8E#0b4v|u$V?rI`hHoSMCN@!^j82eQM;W'
    '?>r6C$dNI{V#u8_yD;G*#G~aF;CqB1v#$!tff`!kgdVcCPsQfVu9<Quo4q(S8d5L)j><S;<SoD<2|ibE#sHMO_fxw>u__sUo<&JC'
    'MMRTbAROvh<mFxcyHFMV=bg>!bP4S$(p0g@Bxy}oUg<SwXgFC@rz?6V{4yn<lCgHWns$L!L^vse>tdelLamI~^ctF1&*jmJK=mpD'
    'P_(zeD`}ghA)O^RrUtjaom4T&*&cF(PGd4UU#6~UeLZ&(EG&tv6g|On#q{iR1rHhi5<MiQyOj<7d+0)%!OfeYcSVlL)WXZlt_{t5'
    'NaH$!ya!f1uFzAK8r<Kr2$^fp^z@)%9IGp_vX2H;sE59`V5O<y3|N|@HLxjyc<sJ%4+h#OpNL21)Goa25U);jPz)sjzYU>!$W=OB'
    '3;Pgo<tI!HHVGNz0x2J%=hZXP$(R)$2)U+wzQ!t*S#mJo2AFa83*s#fy(uE`5(6*9(bieMko5-WDY<9RpSBrEC-ZQKI5Fv!=%-EM'
    'mq%@QYR>f2Pv{50XF5xm0vH!}k>?F305l0~$qA@RwmU@g*8m<t^V84lC9HkI8(|8dIK$7tyM++Jde1ON*ug_Pz#u!$n>nI78_C*`'
    '5g<fY?^hT6UXKthtb4QwlV=<FEIJBtyG5<aK{iDFYt!!Hv`DXZ#14XAQ~OtbN&%FQC}8P4RHbHydQV)hWV+)JPANBS5SZbrW1V-8'
    'qrlZP`DO^hG~g@mW*A=bIBw6z#t^5-36jmZ<U4z&-Y#!jQNPP`i$F;!`7)92;s{*%Sp>`8_Z!bwk--~2Kt|Fu+C74DmPyan>5~3Q'
    'Ef6+8XJsiI<s}tOeLtj>bGQt44FVlEkssp%Fds*`aaza6$+^pRI3@FYy|Hk8x5~p=*&ME2^et*|pkuQTJ|9O?K^jVyouR!mLMtfJ'
    'd)a1zlY6)JE($2OKh&T~z43S#c}UFzwVqx@O$m7tnX5GfhH`ql$Svdtl>%|fQaOqQ2{RwIZi^{xCVN^G(59{9v^_&jI5!gmMeQgl'
    '31eN2`x!9A@5>~Du0F#V3P;(zt2GF^_br&CmOPRm)g@whD2O(7gz1Z)E8p$9t-&F9eWjn#C*>C!TBH&h4-w3%Aw^%)Ub51;yoH>G'
    'MJ|XH{z!-c<TI@5zIo#oF*lofZ4LoA8qfRO7^hdh#n|TUDAK6-0fbCzE6X|;yE;$C1jq;AXs@qxbCjkwE;VY*@w(|GV@!`K2}0;0'
    '+qtdy(NbBtzl9;SQGFacVz?qQ0_dLSZZqySk9&qW_rT$WY@HHLmUd4}Q5kT=UYS<hYRFrebJGqXY+3gQkfSHJxcr)q<11e%tU1R$'
    ';a!>Vsw@&^Pp{d(S;G>vHA-{|?HakvzVjmd8O@EOt;1xv41=q#v<};$OHuDkTdj9xbaQUHk^6avv>xO`LQnm)?l(vA$sC)6J?_i!'
    'ff&lLpDi^cZR+JJJ#>VXyhZltb&uG)$U~NlLbugn4aFT|L}6525yKjA1GsY^#4A?W)77!J_5&Dz!Aff2+|l>gT&i}ArVqrCg$K(Z'
    'Gv{)`SvE=ODs~OVg(2+>>1D7<(ZGer=%Hxg3iBx@^~wsV+rns+Jg2K<y28!vD!rfK4E-sunNj$DAu(|>_6%&fI4DNbI}^s*R<|c-'
    '!0Q`d#b8GgPtITtT}dy92gBCcMM>Ys*nXoyO8(M37;sAO7a~J3uBm;3Ux+m96<*h&%gYf<>h`MkJ`A`OLRXy9@n{WHQkOz2gE>DA'
    'wh@<?MfN`{Y_cRou9{A3@$~44Bg`I!`*}&WKk!;FU@Xs~KD_Jon!4x)iH6JQq%6*><t;c@8da&A<9i+Yca-@Vn3hXzpF!gKGAufv'
    'rH+?XZs)sr>GTcnt1f2hCGNDk4Yj@;UHHLmGXU{%Qje+Y+8gUlGVN_lz1V(vOx7iV^U$BF*L^INZZ)Y+UHDc;f@YEo2Xo!Cc)d<='
    '6=5g5hHTIVg=CAJHPHtXSZD{m2NQ?zlFomUke%?vv?ET_6tJcip}A*~o+^Ri#Q0x`lRk^gC8-xezvae!`a-D;NbZ`+LE^i)xVyYA'
    'X}~AK*h!0&|7y)-V=#t{tVox$WT%_J&z1|e({v&>a`7w~Qm275*Y0_#{b2qH)83K*cp0gX+jkk@m}WV$1z6M6M(49>dK>1~lA2D>'
    '<C3v%B{)`0I^hGzQRhk!u(6~_OIAABiPloQ=Cm@E8DLFZ%Zw16$p*Y8%nHgvJ&X7~rBvNVr^U1G@S2vvJ|Xp*KLO}#DQ_^1I7&7<'
    '4PQz^JB`A)`s+<x%QVQtz@Hf*hO3PBA%gc|a^6D^;nGq_Lm5!iX}YvPoH&D`Uy2^9alEkm!ziF*Jc+A|jETV47rTr_)_yXCmL#mp'
    '$kn;icQ^riNB$-g2`L&DRpSD`r%3V1?kkvlSq4`UbpWmGB0jKnrT;SVM^nhDLQmv$y3$pt%^f`TWQ?DZw#ufN__6d}r<>wQ05NHR'
    '8iENs%SxC$K}igE9;O)-I(_GiD70fpZCpPr6fdW%>163Rbwx-ibGq{Cf;kUK(jl(7i^lSO28^e3XR2N16G)wI4USlt7?Wk--Z|+4'
    'vzY&o<Qyx@1X4|qV7whB25A492x=;Xc*f*a42<&A<i!C&@tKv3fbyB84jkWILq%A?brX!|S=|g8>~}-z@^ywme3*j~)5oLj=4wBy'
    'Ga?S{bFbazjKpG^K6=>5d7OFM_GfX$VJ$h`OAk2D&M=a*mh2m|1SJ<mJ9QV6q>(*uvQFx@CwC1P^e3*WhpH4wY#C<J#&;#!zUphU'
    '&8i-ZVOmN&Ijik_4gD~uri99e``olc)#Fla9sMu>k#Y+>nx_e>3MZU(-OjBu9ymT(5^;pAovz2`0`AKc%rvV<@61_seBgCINy~qz'
    'iFC#xh!6U7!B(~AG#uDjpp#_OdDOG%X;Ae`{hW>kdb&zCvf)%FjMMit&9<Fp9v>!H=7g!Ix9@rz1PXTzDX!NET~9ABpCYr`>5|{3'
    '=}$CNa}*jZZA9hHmNZI+v_8_tHtI}_^-iGL-0@`kZu`C(8_8yjZP%6Ea5(0|wC`piRQnv&gHWZTi+L%vq<bPAa#xR4I>RrARa!SC'
    '=;+?qJyGPNG;!tpkxy3ZaX%UEhjm2BK!qvI6jVpn&JbQ|fXC=un@M!wxeUMP>oxaV7hg)l)9s(>|M8~mqf&@wSDww#xLpTVhKeMC'
    '+ZR$cvFNMbeH%W>DjA#@s=5KUg&fP<5MIXrFzjiADK3_6zV`T$6H?i(n~Vh4DW(~t!Lj%#I#JQ#6X6|LL5cegP>;)jG>o?Iq9?eE'
    'zU$qY-o{eRa9OzC!q>B^`7|c88#OgCW-<R|jn=;3C*T}gQs%xYOWnOYh3Ndt_|_Xw!pK>r6UCE}nNt0vv+9$juH@XQzI4`*hN5L2'
    'Z+tg_FVGBY(8J!H{o*3tH_mg^(O<FJp?KSAq?PARq8o;4c9O{nNQjsC&(d99F%ow;cp4EkqZ!1}KFTDpqsYtPvq?qT#&Vq3We+y7'
    '2`9)(vR#fQKs$}#dO|f&?TtNZloMf^bhg(EZMGFMt}^`=OpVOX<uaMjyf=nnK7>rqSpv$L%tTYab`3Xu3+89I?tL-Gt|zT4Nmf`i'
    'JMWBcc<r0tIcf!TPcqlaroEV3WFWv4nVj~y_ucl*twlFaku^AUMW$bcbck12XEJZF`9kB4ef270rX`V$Efn_nyWYFe7>`qSBz!b$'
    'CIg5r9zGg*ibD`(Ei(#L+Tb(qm>TX153`}l<~3R)bV;Pg7@1RwV2!X2wQFyhiY%Vp&vd%;Fw06Qqo@hFu^6GzVz=g2sz8&lMrfe+'
    'jbGe5V;jn5Sg$oVjnYQ|T+;2oiVW^`ODH7s?o~r7Y|0y&``lKdj~4ZTP8TiXm&>>)YfM6~GeK5^p}U4XFHEWF5=b%9k^wFsZdie3'
    'eas2QGYm7`m)&!`PN9g;l)4AZs3tdD_ZoJ)e@xuak~4=c=iW8Ez<B=;yxRZx+uwfv>9@ZDLijwMJ8S|U_4E0)<;vV?<oNUS)EJl6'
    'gE0*RnPSg^XjJb$Z_5bbQZx;hY`v*6204C89X$+d$F{}&X}-vP#XVW`C?z`_LM6*h3ngY>nli_Fz0ccM?E4$>boZA~i^u?CDsDYk'
    'LY)psQO&PQipoH1=8nDGwd^`^v})u{b-mnX4#_TgJA*gwA@_N&RQYc|e)pGeKmYH)|F}MnqD?=>=W~Bb2Hln%#uGCL=@<O;A*iT|'
    'qyjm|=`)zb3;v<OuqZNYD9g;_kD(s%p#r-Kw=)XvCbesp6b$a12XbdHXbUF8U~Cm!6S)UoLLEraj>p$za{i#Rm*P_a8eu=7<EF4K'
    '4&s0%FDHf5LZ=JR6g!uJ)VE|#W2muvc^)N2yo6rymeVADH08*OE2Jz9q`3tE43Y?5ZDs3Tz8;#m1Zi0f@k0y}!A_N(#K^5(@JjNf'
    'r-i91>tOduhKVnYCw<@(!Mm>IfdM?UOZhbTJ-i%RBiz`7Ucw|c=**-U4F)x6l|iTWdFW@%-~)xXQ>q^}Gs}~k^aX>Sif?|zR-nKJ'
    'jVB~)ioEGTFC{~D;3+7Wi9=d4yeLD8LGl=e#D+Lo2AN9bkfR0|1tGA(d{z#bha^3KZ%FDzR`ns%1P<n1;!CNl%>*>a36Zi(J*U)6'
    'L`U70F{6A;!&NE3=rt#gd_<nDutgu_)Usv|45Z6%@qFD&O|AnP1q<~9t91~~W>p>_UGL8w0*AvB0R$cPEI=$+8-C+*_-&87B`D#>'
    '_q!9f*Fl+rFY2JMGDwAn4YzM_^F=VIpnu7r0#C(!IJ#od3<tu51<Pd*K8R|RXO|la-m6zF`xSgeh7(5FU9u~WNKE^@uk&)1lSFsW'
    'SQ`Vbsuau9F~^NAUQr1>2hzS`UvHZ;#53KDc@Hq&UmBnxGIXD}pS7R&bG4qqCR<qUfB(AAJ$75M3-e*Hq4F&FaRzq4>#X~8((Y?B'
    '#@!{J&vWNJ-Q4Hxufn9q+ECO3lu*S^D{H0`+*0&=pSRUOoHQj`IW4K!r;?+@El#eGzWCJA@i;?BV6v}$=~evk&k&fX3Nt=QjZ$Co'
    '^E{<?ADg@NyhS!~6DtYHV(_7&M1iVI`2o!+B?pQ^!lHIpG#^n9xs;}a>gC*Oq0$ZZQ%w#qPp8T&(J&n$W9dQsO1&mA&4rj9Jydl|'
    '&+BTK&LcXBYLm(^qH<|LO>>uy`w66S3*QGX3=?Hh{LK<cMhaYn_9vyO;AqK0*2Vov=Og_N>!>1}S<)0CsL}Ow5kzRToV(8|M<5*)'
    'GfyFrVMt+v-@_yX`9{Ve^9aOe2T#96lWU%I9COo-4E0yKze)wfc$E`|kiihpC%GCfB<YacEEke-MkFQS13x6}VX;)_gcf%vU8Hh5'
    'q{M+7%=1_iBb_>wF;y{zC6^9f6wN;7)_}}2L#H>`r%8QlLPMFhnrbjmV37hO`EObhfBR5+f&P-Vf~<2Ka_3rKSZ$9;t4dTmNaNRD'
    'H^D-}T9Yc^Hsx!9X*@V~h<Pj#M#XYOFqU73-<O{4*O%^UkAQpJ{KKFB^`C$K?stFs^>_d2r=S1y`;XQ>=OtZT8=v*r6ff?Z)*&Am'
    'l7N7qV+1yM=o&-Vtj)&C(VFZQoGbw#gE<r-Qv699G9VC9{MuS@fA3?b=RxEl9|m+5@QF8B0o9Wj3f&KhD5=l$ASoeg*d*P$pY-ov'
    '(w95VE}don1jEd7lb8GQ2(QO_KkApqf4|cEHQ%rPZjM(Iy_>B|4+^UYj+!PF4Su-Biqi=GHpy$6)F-ot`cv|dG>7b0ga%tzn|I2w'
    '2%AZlcoYtk5Ut2?a)hU}NeaT!+p5i0=DW#XP4w#@&|(bnwFGbFvo*!sHda!ETn)ick(UrN{*;zyA5|*j5a2HfTnWDgygh8G&M@m='
    'eUndAU8yooQ~{E1?8N~-1e8f}@{nA*ZgvWp_T?D~+nrcoH98FDF8Ki67P%>vrT!t!b2Ywgc&8eY-bamfEg8zgs{o0R;dW73qM)9|'
    '(6XNv%c}^<k7;T`rbL-N44>5?<Yi(Ac>xi+8lHzKHDGO<eL<8C{0lPj%b@}jjY;*3POh%VebkT>VcTj7UM|zsv+%p^q7?W#mKN%$'
    'Rknbo)axP89OfAXzu3R2YvTzJ-{?IN2DO3as$1w@S?+esqI8A5i!5XEi5OW`nC4;3Ojy^$_}adOZDzPf<_ckG52J))f}UR!J|~k7'
    '7xusERu6fntFQ-&%4<m{uzmdiG9nhEWuy#>Dtbs#o)b>>8vNYys8qM1=OGcJI)&Jb;%4Z|SLhk(;B-&6w+4vP&y4BFvGWk=Dr^2j'
    'ub2RWhyE|=c%DbY(CE2qt4HkNq(jy4gK&_?*&-mpleOr|a)J^FLV3@gu6(|V5~6f_-<jg(n5203^^=%M6~B8D64|fWZ@CJJypTL9'
    '$}Hr`nxtE@esG|g%xJ(xO^x@Yuw4dHZ)R7Ib4N_sv*?>hp#VE~&1Rnkt;UydUv~3>sJjvJk8(yh0*;>*J%fj_)CxfNeeQj!mHO!='
    'UagYwoOMfjgb3&^=3|5iD3{%@ar&WCe-8oK?wAJM4E)yc;oRqL1`!p4^`@L)w3JPF7Mbo+Y>xPD>pIog8&Z^$h8Z-y_pH}?ouFC}'
    'Q`lLR_^>xr8`&>>-*~^m<=fA>*@&tvJnQm3oj2|iUMs-QKYjZps*0bu+Yl%J1abV9K9DZhjT$L_9lvl4&>@$iD&|lduwr%nCz`e%'
    'V*&5_HdJgM(j_1-93oJU;hz|amCKbmRX~^h5E^4y7GXbDdSOa!tf1q8#IPn8e3H9_(6KUna2PAfz?NPzWV!tNl>VOj^pM~@?|OfT'
    '!*{(aq|kME0rZvTJOIgkNK^rvq8n;Fz<CeC@ZC)v{@k<+)_%v)FQ-Ny>VI{P$;rXCLO&4h=gw5SuT{T`Es4b3IGlTzUu1cVXHm^v'
    'o(s4jhcp-7Z3`ysW6L(xh#W^0<jcz4Xe*yt8A!iYbvx4{8DErjhM2VA0?ISXDCw`mD`c{SxWNs#te5>Zi%p=Hwbi_Q+30&y^rA`*'
    'f#F*vhba3=6-r(*n<~I5PpJ})Y*+|fg>BPXaCwOXC__pJ+J%SgU17-r*L1y^SHS?dHR#<<7BzRd8zrY^GcO&st;SZPyX_?5r_+H2'
    '@24dJ=k+crBiyPfO<SmhzA^p&9y2eY$<4kBG_O}>9?G@JIVSh65qXrbqkX)t*tGf5XcUj1n{Gp8ZZaIy66wut!#?kpo4hGE=>Ebg'
    ')GXPpdfu&npSQQ?j3zL?5}0(>pn@IS!PmTg&AIqtIt`mGtj-8Y*0Ys&K9HAnpLbtH9c)m#ZRnSnw&roygsTJlI4yi%v7gZI%y;3F'
    'bSdxiB_#t*kHl@jS3?^9b;XzKL9Cz~z55PB?Cv}Hb)PGh2|a`q@Xq2~h8$XME%&ecz1X~W=f7@WtKkgogwxZTx*J^Q#4ycXd(?L4'
    'h~l;D?vuN|-3OSnBwtyQrSOKNQ}wEh%l>uCiS(Or{*=7Mt><~O0uw&jqYcjBCin63S4JgsUX7H@u}c_ay~^0LYkn;szx}_zJol0R'
    '-T(A2KYh<$>oGn_d4W9Q6FtuB-ba9O5O_~k$q2K`b1z&19{d;~7xSEZDZ}$5Q4%}$BZOu%?#>4@n$E?CniYrKKW)Y_OXgbqV-E>^'
    'lcz+Jiu0M)Y|`JgJ!&MPQR0{mTvpH-2d=qD)MDD5S_-l#>FvT^C8MvHI2^~Uaf(v+6X-%uf`MMxq?ykB^yHA1Be}mb4&2Ac)Ax`K'
    '1_(8qo>)*rA4W&%Fqb@r_;1-Tk&A0u0Teb`@K_8cU66jJk_Wveu+!%&T%qY3A3>BpPn6-1pm>-zhEZIBv>)cN)g)>ee<o{;TbQ|-'
    'dzt7bY(b_j3Ou1m2k9B5PcOM@Bw~%*`^aMqpAdJl_rftFBo#lDHRC*KyTTN3=#=EVd<>a;Z~#@A6QA{{vWn}dCVA&$*sA<wG7jG2'
    'fPPRSNP2qir4oQiAIgX(Z$q^LvNm$lq4SE*dFeef$?+ppkoW~duZN>f;!FHnr`xJ@dkOC~abd3?*|U9&@czNguE`5gz9fOf*xVtD'
    'GtwUXFdLO1?UYCoRDj~-qID0=#}D&2Pm@@*{R!@R6|vuJ0X|HWJY~U<_>iwG-=X9_?gBmp4TAwYLs|*Lv*vLqpkXHbkaUH}@0pDr'
    'F%)0tdBmL^X+!SEA+YhiU>`iP$N#uYki6)kVQKs@_lJG{m3`FwxI(g+x#2Yn{q)ygzWwgMeE&zB8E5}I*l;iaKMxvYbGm6OU(!AT'
    't&EO4Gop!(Csqf6kX>?bXtJC9&E)<_3wp9<3!*R)%oj|_Yt2M#G5*bm@K=%(V)~nTW$AfEOnkE4m~T$jW!uS&Eb$#D=|^WiD(A^Z'
    'X+EEXqCi@Ogq%3Z&U|y;%?%p}7X^q;tg*}QhBHZNvXhm{8RW$2B%J+ZEq3ZSN2V<vCNq=UMIG~^<j=%Lj&!A&Y3fYa!HbBS+189g'
    '6T(h##=CZD#~4J<;z$IDV%rnxIiifxEpDlrXOX4)NDYQ-{5060#(SPcNy6Ns_MGec8TZsfr8!`X{&clB{(ou=m=$n>;y03&hAd|E'
    'oYI;(Iol<!?BCO!{#~BI-~RTOpZ@ssw_oztufO|;pMLzOpML+`Gv8EkkMemUFR^Yy?7Sfk2pRMuf@-58_o#%k>KJDjFBMCLM-e~8'
    '8P-aU5!zS$(r#gER{N-{R?aNIZEAG06qr0ia)knGWaUVgs$q0u>s=iyQeV}mP?2M*%IQOIezhYN+?vuWKZ{L%Wsj<yYNAS$b=gcA'
    'i&3^xI?}<Vo<WTiTvZy@8b~Trv-DxCtR}BSrOV`?;ImOTgrb5VRK{l6Jf~+7vgM`dnO;m4)Qha%)8r(aht*_w3F^$ix{Kx#74>uF'
    '>LCWH{t3kq1<766W>4kal1)Pfv7(d|OOz#_Mb%*>_|%UiBBDUJP{a09lwy(AOINFe^QEYNnR*YkT09qDNH_6{(1!#f4?}Vi$~^4v'
    'RW5h)3RF$y=!zUFN7dxCb)ZJCd5f}jA$Pq(T@04SN9LRmCrMG4Rrk}SI_W8;sO}v`{=Bw(_H4Iz_KYT^Ya8de?$H1_MwdAB!Sa=*'
    'deuc6d+gH1c-wM9&j{~J3q`UZTWuYiYZF~+Ihy89(pMDK6pD~COs%q0YTKfoq9nH~f+3GH(Ht}OXEx5}(UjmUj1Dam$N5!2i<>!Z'
    'IVtx|$RsvnGOu=Ek}l6Pk-3M78$$baQ0VJ6_NrTvsC@}~NFK%{n!>ag%&h_+W;L=MErFDOl#m!?iz-$d7mmgw+wZnE?R1g5sq4aN'
    'GX&KV+T}dE9;AUvWPt5Xl$Px6Y=$0g!Au!rr>x`Vk&UCQ8MM{7!d-!*K&EcmN{fy2EYa0&0k<7BV<uu-n~ukZ<qRvWvT=_QSvz@O'
    '&4;VepZkc?CbK@wp>Aed2tdh1mUomUF3z%aCUQOaOh$;CJf=#C=?-E;G<RT)g#ZsCBQ{|*+!2J2O#px{)$f79VlD&*c%15IHa-${'
    '(TXynK>;rNT8~CA!e2VK*=HG@z)f(*1*-mSL|96EdwiFw%!~R^?9-T>fb+4wJ6S{<)tD=<ja~~n1a9h?1K8(d_?ni=VlP_`%eE6D'
    '_O9eXqux?W3|$a4Ue3_!vftxJAo_BaG*rd5<OwPcd;Qk&XW8g%Jt?r?MNo*vBJ$Yr0HTh4DLvMRB$J5V82xZUG->NCD7A-h0?{<0'
    'aU{?vd`;p;^{3Mbl9Oi&B_XM36)>5#5(#z+#4M;sFO*sm3f?>mCNub8GIlFNt_(U|?uekcQsnX>BTb>Ws;}u9!Qz#bP{D$b_sv~$'
    '<r3FZ6-5Z!7PTBn*rjA=Khk~=ea0eln;H`>R3<cPun`e9ctnE1G<j=bZX^{|vd{<aRMpO%%hVcoHhq~S|E`7evT4frR3hZ~gzvK5'
    's!lFe4=pQWUcgG_Z++;%6IA0sGpD#|qRAA5xo1p+0ITV0GS?RgN^qF(YZ+X^Nx>x)qj(I2A`VtPJzeP4`o`v-FUIx2hL<8opE@#9'
    'T5-&XkTXLKEf@I>VFvS9I2xIOO^<DR9?!8bM&SmO2~2q=?@{`eez884U%;kB8K&yAI__g3@MD&d&!*}=lfuf32<-&(8(hPC*|T!|'
    'MdX{Vdlt7crfKHu5Y)wM`KU*p2fsyASRzj_dc>I&d6D}f;!2TQdhW)KxI(2Y?h4PWNZnw$s6ZueaWf-Q1)8zXacT4?$(18yQ_J?1'
    'G4K?5s_H%=^`R@)XyzG@trIt!jv5^;*Z4S~iwa@P^7N;v7raKL$?rDHt{GS+3#C%VKFTz=V7tQDR~3k<?3c@=1g3kK_ER8^&LTm$'
    'eVA);<SMM~CL#4}A$ct=uf^uI{M-xDYl(WTS@#<DTGU?4+<WDFt%mOv^2@6EUP<4p>ubgRvL1_qIB@HWanSYoTFc+-`<FHUiyiQ`'
    '4Q_kkZ9{yqE8e!o+x~dlByT%K$i{4u5X%)csu?Y;(+{1g8lVOnG&;dz6scUu$1uh9p<6k4v0*}O=AgHu;4eS@{M}!F|6F=L8<}6)'
    '&!4Lbluf*=hkBEaC+6&7dTj5q3X}9_Y~x{mRL_7+G5ZrkH+KzD|7y+=JbiE3x#%~ZDH_PgZSc$9=k81U+M5$8l-P#!IQz5r=%T-*'
    'D!xy!ZqxT=^KRK%L_gG{N_^h+zFX#ppZ@ZP@1@0WpQz%Y>-E_{l$X6ng*<&{IFs~EOY7`XlTKO#f$t^R4El?cw;)q^BL4WD^)*Pa'
    '8IKS1#V#FW5etaLr%-b7VFK?%9FiMcX(Hons-g|W5P_jPuKf^VI|8d6b)UCF+z(92JprE>S1~2>_1!y2YZ0dYdLJuoA3H?VSdukk'
    'ZOpKH$azV1l-&Wz`C~KMPa(4e^=1ixSL!13;`tSS@$#6E-E+pnCr`;f*3W${(~}$0xfJ=dG!&T>3{ASN^<lZm;7@(soduSf9dNR}'
    'QY1!zu);8N={30a6|QR&cBi^)uJ<uijRL)9W)r1NSW)%}8DdKC+-`1e+`CR0gjzpj3!m7L8xcSR`&j##OYWg~H0`$~rA2pZO<Tg!'
    '_E_w4kDbo?Td^Re2)v7hX9_KMNhI;5RI=a5baV?)>-p7ShVzrcgWz?LPzeUdNozq~Z}*@zVjiJ?w5>OIddns}*4)qY$nKFDG}z`V'
    'oBV5Iyxl?3sU*G=CsF$r_gRK2TEP2byS>vXx8Bi_bw2P4DKOm`hfLJVE@@<ys)(gwF_g1yi&AXKA3@1E8t&Mh5gx)=$79<vKFj2S'
    '+`>#RyTxic@@w|<Z7ceOm_iFLyc3#pOX<5J1aQxcfty%7^pe|VZKp0pSW2!ywJ^f$lx(IN9g-zUl$je+OQjm!J*F?|@i)7vs+jUs'
    '0P9x6+3GkixtzCv&P!6~C9?C9-FXS|+|oQRv7Wbl&r8tfCGqnT{kdg;&bx<x5evGqT{a>xGe#6lcNCZ-Wsf)EC22S&xkQkN&0-{V'
    '^fc*v!6B2pn5NpXL5H<V(b-ih2j^g(Q$M1<domG|-!t51`CiyRiD8g9>DwO)dD;#uGmv!VE%|sPf%tr0aYf~*-N(8Dr{1Hw?|p~^'
    ')i<WRea};|j_%mW+Hn>4s?PaJ3Xl8>MB?1JebRR2Ta;I+Ui;c<i&Nhm`BmPlUEeN+J7#}2SxuMN%iIlM-a2+()Y{ix_pg9)$J%cs'
    'EH0(Ihh+R~x54R%V;xA?g|O!iu0RTqk}{l0OB{PGg!KJ%(2!6e%uz;nqjxfe;0B`v<lC9bh>jhaNQciyOGLj0>-tLPCO+dZ!!3U$'
    'Dm!-!#jHi=yG3gYf?go1uASa^`~Mf8rDV^UuxL%DW7%=rBpgPsX$Yj5$#m?up`pE)w8<`>$`>*jKn;r3gR7Yy6-1N#$NH*4K$dTX'
    'OI4)RL!xs9N|}+)`vg-aBIZ>^W%=2d<Vx~O-R=odTP5JCVB~HCHex@?Xj23!lPPXqtwHP!K_zqZu9RfU-L}`A@`hQKG`FH>`c9jh'
    '<IauMRNCkLdwckQ{^_58{o&j9B!YkUyMOqzqDar?{cMoXlAw*ejGptoqqlwxVTvP8r6PbrE_RAvPB<cp{=RG@V6Y!s-ki!+u`D%$'
    'mYiQBVL5F^1u9RaWE8txIrn^>fRg;aY}dPQQ;k6AS7npX$Yd>Y+&9H=P3gRgs9!3oYt~^?jrLW;T_I_-moqVY&>9Q{Q>&^h#ne)R'
    'BJXqWOO1Po5W@CXjIv0wukYwSK^5iu+>Nm@PZ}mEf)NB8*h)T&Ox9fo2`(L>g6cn2Qt7zN7=_-AUBeC9QH`9PUpe%*+sy$qR27qo'
    'xv>}uMSxm$*J+YMCF=05;r^Zqavpw96*)ycrC9EdSP6x`&%I$?!+db+PjO|)sd(wpK3~Os?!FXb)P(I)k`B32G+&aZG=8MUmYYdE'
    'sqf3~TIS9GFNrL%v~rYOOor*}gc~!K)J$W9xG5CoKA5!47$+fQ?MsiOl<N|NLfSM=!Y3;UvIOL|@-6u$Gh*=+%_i<*;3&3t(X5xL'
    'qhu&EXfCU9${CS(Yd}x58vmTSj)jX&vuDkM8RoJ23?r)U9widhf-S7-x(nbt&GZkx@wn^lMuAR|i{==g0V0yCyHY&jp{A3SWS=6j'
    ')?er1yR1fWn;P}51Ne0JTQsw^qx)=@a?VY?=E_)EPAIP$s^Y*d2~|Z@Pf!dtHB_DG=G`zaNcT=+P*39av&e_lEq5e`J4K(Y85K9P'
    'R@@sMP|+sN-O1P#=}Ij)O!qn7!mM?}=J+Uu0uW*sAmDmfc7{7gg&3H;BuK%}?U+(x?kGtNHU@7NxE=BEaa5p0)<?Vopg$AiJwfOK'
    'k^ilTjp8=VPgU2IH0?$nQA^JI1x=JW&(C8$(2I*enw{}Nhg;~zW%$`$JJ}m{sq0MNAt&IpL!Fe!B7eTwMf9Zd!7t3)iiLQQU0X@e'
    'xlH%(*1~+0V{xT%9$(I1F{+GnUT!3<#Vsn{)%;4><T5cG@8+cei{s|PjH?(5yJVJ?_g&P?2PH8u*OC!&E$7nbC%a5(Cm-S>aA=74'
    'a@8HmkgF?=y~7s2k}!Lb<Wb3Sz6d(j>AUfK(-z`xqLz`Tt;<^aNPOios~vms(o6?fak;{p9HZNic>c9F?3AYAbK9jyllof5V#o!-'
    'B(F&7T?95OyMW6XB**AR*nk(ioPDgGiNRkpIR%txD`cFnX%YZ9UkpTJ7uR$-JV=*Nc9~XFxHu)3(B}(YBT<oZm*{K5m<fnvK7|tB'
    '7f7dEt9&&3_lpe5dMN~7O2nWqC?zYqql-wm5tS?Q(F=>!6^JiQw9F)m%lV!a?8328%{*f3TuE5E*w$7){Kc48RqbM`uB2&Q;O-Tx'
    'zF0<T@|aizu!Sxt^i&4FNENK?tgge*7{8`OA97ssY0pdPX>xroQ!^kr!$Bd9%d_Oyzkt2-EQp+eSgm~8lTT6EJDuc+S9VO#Uw!69'
    'Nr7$hgp}V#ZOtBQT9$UA=f(X4w^y7#><2_jsV9fWywdefmn3VMCS_b^Ls1J1Rnx>2p!VqzkcsJ$=grS)P?7Tp3|AuVr>hbVbh2fr'
    'pf}FpH{#t4ozfG{JgV$vPZuNx)5*ZDB<Y`tf1wWNvLY)RzIywRX;(E&HM9%X+!=XrLwQx(Z-xx5vFT7_uE}Yp7CSDOGn(zH)oDY*'
    'EWie&9jd$~Xt)*F8lcf#Qp}AY;}KH5GothhhiE%gMpOCyQaSC4Q2oOB*;F!PKr~sAPQ~WoZ&gLTNL&LFP@*Khw37A0<r*u-Vt#2Q'
    'DrsBgeL+p!uzMB*Tb!Wq7&XxsDCHJnLco!D4%BE9S-1#UHf44$HRa)*xzvyM>BUMOmOwO?`>F8w<Vhv(rJuI0zA|s5;7v-t5yJA_'
    '=WAR6s|wvh_~W80qwx44T1RE|MYnAdosv4Kp1QnJ`&W>e(!s(U(S|fTf+OJuqdzxgbZ`bVcn!DT2d+pqxeX3(xjGwYlwDGljOeuF'
    'rlhO284p0cOE_7lGGvFPO>8QCjcliFDJJ%usT(QFQlvHpol;_L)%%KdYn8Ro`_Rt3;Qn<gUq&sw5fPGhhDNm$LFZRmT$MP|3YvUN'
    'ERef&P*U`Y_ESPm*y^eB6q_yiG{xj9RMyOskXjamts;41a*2H}qxLGai5NP3OI$jK??$Abjm0v<|4?*{taEa_edZn=x55>Y5s@-+'
    'OfyyQYYA#~AnPJ5bFKIvrq@NNFV|vYtNEdeqmvb-d!);A%DUgxd^6h}`${ZX$L}Ufr<{O}qmD7PP%G#q<98vd$_U_@tJId&67xYd'
    'VdIMSVkg}|W=vH+6l^pv|2)k3P?*tDJU2+LUK|qzV||fah4In|P}#DY%4wXW`HAZ|57SiCd=yQ>Jb4(u@(N2bXtguA1=Y~SW8_JD'
    'm_3gmGu9<W;@m`AqJz#O*q0WK&y-AAq065bm7PSAvjk(*B26mN;iMtNhnz~xdaZJ@@;_EFT3z1B7W|pClYK~yNk5p@0wRY^m8rK~'
    'hEae7o#@1gEs$SHyxS%7RB*D2W|0?$>7lbrGxN0L8#M8>>s{lPorIT@uxVfV#UyQLu3O~uFyBwf7=|o2H53u2_%kv%P7yamfli#H'
    '9n0txt!4ayrB{lGQ+x*o^C{#X@9ueqRHBeRy+Q(Uk_z5k{X}^X4?D4%I~Lw~NftqU%*tJYv`bnGGL-QYsPl2Da!;&V8>BDs0E8;Z'
    '<GvV79k9iPSD-TwX~}f>Yt-TH%wZP;N2a|C4}C`xQeAQRv@Yd`G5D^-oF(k0L+l!GhC@_gCCbM$(<81T_6nQ7h@`?=G6RZUcU5~6'
    '^@+G*!n>LXBK85zY$^Z_`1Lw##YLwU3v@99iw(QPAu<VG&@gq-0GHpFoQP`}Xl0i$4478!*p1K9E>;%jT8?h97hS>PmAzFbb%&ka'
    'u~m!@wp~42cr}pC?KtrBsq6zU5Jz!S9i$<5ofil0G)YSDVn8rFl#Yu-WSgFlgI4U>AmY&N<TyQNGAEmONkt6lwj(t-fFiR(dow8p'
    'S6Vm8L4uuN?99UHrbh3g1U!jFUs702k$xVfUOWNCl-O?aKtkAZ(3F5oQ}nY)WQFFROp8=-ZU|KHQ{l^k{Gm(06^s{iLkXXXD4L+b'
    'aza3)I?V{Q_N0gl4>k2X7{wu>5Sx=uh!&bIiq)gE@3b<AD?4WulSBH{tL7+`r^&>QK6l;{eO|vXnfzT`Q81%Z-<DudY-qf7<yAai'
    'HDLeudCvO;C`0yTE3Q9d9@2#|arBimDcVf(se(n=mv=kI9{O?EI`)e?=Uog%F_e<az=w3PR;mtnCZrN@?Mhx1PqvcA#fQxauE<Mz'
    'B?pgT+8{)uOb(@F_-leCg~Yl9$fyu~VX{@Chs2GwnR?_N?3Z4CUr0}GVS^VElyb@IqC@hM*hLpS6pl(yq;*0}QzW=LLVq@H+8F(*'
    '>P?g+XO_s5Um$+yTq+-{{ZZ5~9Ywb-_Wg)!UplTD-W<qVh^%q~Xe2pwq_Hv;taAIIFR4nXqzA9E43BE=%OTJycRSYWs34`4$GIq8'
    'cV%WUNf<Gho{GMsVX~ZF!_mB(^irn()D5BPvR84gj`QwOI=I4wgUpPgn7Pt4&mw6BVolZg9Om9tCS_%8iauuvrJ;yx;H~7-6O|Q}'
    'c%tT2hxrs^6}Bsw?-YA~zN=U)iRWx;-QxH|hMwJup1{xzR;M3pSi=Qe%nv|LgN)NfQP$A*?4xEBv6QE4)bQ*0Ldj#LN9k4R7^}t2'
    'wgOT3j^T{*#wPw=excJe+Ht;$HZ^P%{j#Y|3BGK}OO_sQOj>ZiBv2w(y_>91xlzAr62xV3TCyn}n~tg-8Wwcgru8R*)IORFDy(L_'
    'I3!b*Jqpc$?d-veS+SiB_j6@E?s43RDXIMREIN*-S$i5oOZFzGAVznqnKigYlN3wDGigHI40)-d`)QV;z}C~uKc{VW@I_kbObWYL'
    'ERtWHhZOO<k|u@;yyTMVnea@kEShBIwrkb+n*G>j2t_7cAH6xJ4xh|9)yEOGJ$(lDH8Ofrlp4SjM9oX6MPgu2-oE5oMlIzBl3%ze'
    'YT4z!+eIIzYT+I>>ZNI8wwv~`382MW(EHlUx*;8IY%sOy$-(-o$GKtVMVs7*gj=nfGiwHOH-)j`W;SnYpNsW!pWtf}^>GhD@rL;}'
    '*txp?`9?sfSNlVbO4pU*NSABdoN3v(tQ6h3*i(<aVId#(*r<1dVqLd>y~Mcp3<mDjJnD_HOEOxLgKLtwDYa%z$70gC=+26<$p%fF'
    '1WTKHu%h>@8;((iR>8)i=c|{@1laDyyAh=6CSwW&>c);#y6uK;1Omm-sw8;G5l{XvyIJqXgx3#eIZP5oE=#iO52shM6Z_$$z$X1J'
    'nm^*qMHXd0g8LvB{W4aJHPu`ImA`H-mX+?=<oT+2wc6!!K&Yn4`K4-h19b2#9u^Hvx(|2eVms9`G0{C68-*P_m7%Vh^nMp{2N;&5'
    ')3(#sWY%bBg0iH<;?5~@7`4+!PYtF_TZ3TCDg#&MKK9h@^2-<B*d@sj@RhJG*@vvp_t7)KUb1HrC1n&Iev9g^e;hlG+mDgxu4yG9'
    'x5JTOlbHN??8^9V_SBM3=-n7u`LLu3^g|X{H^wrv$#~r)%`3X-d`OLvM%q6`-5_OKEA296J#dl~q3DuuP@}9}nz#C}t{VHy%PiuP'
    'lIX7_drzYm1Fo4JBx<RhsTc2&aoD1(EtVL`hRO#NX+_#frv0V2Nb8CW@MYqJp2p?lU}frn!c3YJeYd!v5@5zd<)X%Ru}Oxi9({8a'
    '(Kb@!wn*UAR(c7ov-~Ww1IVUo1h%c@0bDa4t%TN_=B?(8I=e-RqP8@tn3L4xEr$8plA};sw3T2!eMhR*6leNQGsW9_tHtyvASL1R'
    'e6EvsQHPy#3B+4%#wY`IA~kEpq12|2&dS?nFWW}cM_8n{YEz2|AC5jmirm5>l46Ys$Yu4|#$F32Z;l>m-!uM5PS9QHXQ7{Ek3%*<'
    '+j0MnkFy=yKyD+JZk6cHBH36{Z4^DX*?oELE;7S(O?S-imdtEqQ59y!D#oSjh-C(p!)~{Z2#m_0YSkIhtPZItBurBr`YzTacyMmX'
    'WP_V|PfU2V9gwhiwOE5iuXl<(FXOtHw9v{K%N*CXX&j_Bv;a5`9q5-&hDToHK;EL}(P_St_N9j6q>CHAWjV;*lDsnYyw}%5Chvi9'
    'aby}aVBiFe&DD-E@74*2CfvC#7AYp`eYEUILXjn{9eXFy$X0pvz9UIOblOeVS>HvzywjCaagBNV|FQKZyPEX8mfo!xHU9EPG-~g_'
    'fZ>_p`N1hS1sG+cQ1=h-^*mJUJccJ5%1FHmg&K+?xfbCj+e+{_Yj<<&ZSlP~j8edNZ+NEglNd4csNN=Q0dHICc415RBbMngFb5Z1'
    'tJypi;!4|w-{9G)1Em~=H)uD&&Q*6QisM;B)8!0!)^wFD5zRLd7aaEWLCd+o`R0j(59(E}$656{e0jyyN=Bg=WG-I4UN;!iN5ORA'
    '-D?&>y}4rxP3(qzrZ+%hu52OBO+*FP$!Am`PQtV5!cTFX@SNf1rRau?%X!Rm;yi;}a}Gz+HKfA(H8u(R>R-DG^M;ezn{a-1KV?XG'
    'I@k3L`kCOKz_%YLzV95xCvXTvb0MS>V}?-&_kIN=C18P|ygOdo6PuKXf&Pv<MV|h=brwD-y^v`=y-uh2#D#C<DWq4VAb48c{p09I'
    '8sR?&2*MHii10A_jG@L6-#COVX5MZ{+Zy%~NZSYjybU10ZWP<<(>T<ud);N<ImIl3m+r*w@)h%R%1NEbgDI#i75cFoJsQnoH$BDd'
    '!dG)hzWBxXGKiF5BnNN9f<br>S9=gXPodvB@f8?q2y0<BbEkO<3U?2wHDi3~C%`HpVh<EQ*ldj6fj;Xokg-DxXffd@Rsh<`Jpe+c'
    '2de}CA$U4NzTHpYRwMApy2dL}_dRhscZ1BY^Rsw%;V8Os*gsf(z%gPRxD5s3e0vF8zMk_`?BB~c+=ht}#boeUlWp{;2<0I}_^;jB'
    'U$I@MUBGtrl$C1W(L)PuiEvLy=#;|n&aLeqq-o+nx73d*-m*)E<iCr2PZ73J!bjIFf!_^&kl@E#eub*g;CsX{`kr_&_?pptQ@kqv'
    'M{Wb`QSkU1XC}=-6$u4md;y@|jCzi%hvVxeGyy9L5wjc`H?gs3W{Cq#k#EbD#dxK}aQ(wso;W~QObH2udlM~_z$3u3&@DaM+Ng`-'
    'ZTKL^qCthQFGKy#QJq($h1Z`osY`FjNYF16>zk+W;Un;<dqp{gVl)-<lvrQn@1^lOn+#U!E7zt63(*zRAFA6=QPx$SX6xNh*uzPC'
    '!)w3i8HWp@%Q#$wP2EJ4C7QFMu3+J)B0nPlYeR(~y+7j%N@^Q>XtC8~UE2^Y-bT1r^XdCrp@?Y1_CRAVfzD6GKwd=rqdZ*;oTWt='
    'M6KJ}iCkRo*M~(+8vQ*1k0*E(kUOShu1?%U@gKm#aFY(;_97I;mlMDlg|{$E4NN_&DmwbAsR>fj35!;@oaYXHlEH%S@a3bZ(9Mg`'
    '<Yr6}t_NC(#N;-7apNL7_d-LVzZdEbcepDj->pG8ri=4!Fy?Z0x&p7Wa8F&<a7E7kK830mi3Alg!NuR0oe=E*`QQHi--4UI-+zVS'
    'AF2DV7)IVNDan5jJb#HJh1BzxWIB!hkN@|7{XhPP|LNyX8gBpk8AU?ho`1!?5_?EB{Fki#TLS+jmyy<pN1t+mfQlAV&26+MyHl=O'
    '<i=^Y)(P=A%0)>F9>sr`c=or#sSnAja+F9+zE?`x*Y>Zff5^T6%m4LX0#<wi<b*5fUH`h1nfZe&x)T08Bqjg(hYV;p)FrvQhYrnu'
    '0B-^c_}6bvpnaf2m<5Bm*^~*k2d(~g8BEBPXTbfbU%$Uh7xfCt0>5oq*A4pr5tRlf0GKw@T!9)H5j}sP#|(Omv`K|7qy9Kp!#|h='
    '#I&WIR(QkYZR!PV`zeD~ThPNS6V8GnXJAYs&iyewW~A{l$0)oo+xl>|U8ELWVDB&U`@k{>kKK6=x)5R^l;~<lhUZh>Vjvr3P<&w0'
    'Kv#hB`y)9n$!`BV2N4J!pWL~Om^{v7g5n3SHYRBMr|sYJ0&7S@G|)?+CjUHzc)a&-9oQCNY8hmveHwU62+%OwrT@U){`#(F-z=|I'
    '*v>yFv}R=e^K{3m%dIymDefQ4hJVcR{)if7{2#LsTW7SBAUK>5>$hBqgCXU!pVXhuwYlJKa|csLRO$^*u&{{mC5S+5vISn1#`lk`'
    'OM^6R&ju??YBhN!T*IIrQU+{C=2ceCdx?e3XuOAck>_C004Q?hO}YO*mnskna#qwEO-00Q*dr--xFCQj!U*)jy=<PmS1utK9*KDS'
    'V}|!v_#rf2FAPS)EQxWez4;27q35rR$#9&#BHmc-?+c!aFMVHGCaLp2jsyE&e#zzYb@q73^(!~?Pq~n#p!1)nz@>kPeM|pSr-7XE'
    '+k1WQfJ38yC{TC~f4+TErhZn`+q>hIe)j43ko5DCFDdPiy1LpNwr^rJux~Jcyu&#Pg6|#TmM-dd%VrN1xg@KHWnh^1hc$gC-z6sB'
    'WRKRLLzMM|AxQqtx00R~Wc7tzZ(fWQBxVzd@xsJsPRw`tTfPy7r+G6D8$=(xAlSgovHVl+W*?)D{U6j>;=iGYWbMf<zzUh*mz#{E'
    'Fj+NuNTKF#mkFU4@ZI%WzPq6LFc|Q5k+!X{U6b6f>@VQ;O$aX3?WdfPY~<se{HnQbA*pXBN~k*1GIh3?v<sdFvTDMP@ZR(5!z)6K'
    'KD>?<Nqt3cR@rZ~evMn+`+{x+q21nTeoopqgLrljaUo|6AxxNLVs6bg>1PAC7;{$mA4UP<Oun1%wJd5Mritx5!V+PMa#DQ4=XuH^'
    'n$vRYzN^a!N`FZ6+53ljhMoK?yN0teTi?ImMKZ^ES2p1BjT>xVwKj(HYd%4cyhx!oT}#`fqu2K>P}J2Nks*Cny3!`E0d=~0_~5ue'
    '7J<-h#d84j&ypnp`qRZY*v^veAm>h7av%&?IQm~@)-71y1_9wj$`QF%K<;+f8epbnFT-@x&^jiX5H9gNpRm^o?53T4AG~)L`1!(m'
    'tIVf;67uA|i}EPqh!C#!^S!+|;oes1bA?`-KR3@wJbd}>WMY)x0S2~0ZNjaFG?<(MT>v^PK%JQdxS52G<D6M!F#dCr6^r^eC$Q%%'
    'iLK3Hh(6QJN4Uv-ZI(iCvHkIk7-U5TWsB_XBwm<t<C_dQW$Ji-kyVnM&dR+`cIFAxwTu0)w7>SBRLd_}P{<|G)lD6fSGmDBxW^qg'
    '&Men0fgf4tyi1v|NH*NAV#!D!p0oG+Ywr5_*KZ%57lQ`hxb|OV-bIFU&)NGm5%l=x6wUzB1fM>T;(=I}TB{uvi%I&Kt{i)|$;(;W'
    '^pjOo^{ah2_asW-#P8^SQ$9WWV+wQOw72uZdV^J}%dCJrA(*9PU4Lp_(Wd*$_2Brk{srrjn&H$ZfpW?4S_5(^yp?i2-}roEXqmtB'
    'HCg*5IGOP`Q|Ito*i)`U+N5T^*WW%ev)t?dTc)4&TQ~<GlOaYad)aZI{=UC*Y^Ev0=SbRWi6a_&{4pcVdsRl&P(1LBtQk&|Zr+X8'
    'Ig&D>zm;}1{V^NsI9wCmGt_Z23kf;g`|a7sUN8?y9EPX;8vl3vu7UTHvHg*S#xiH+GfpP+Q?7ThKILDivTKYC<pmEnu{V2`leZVs'
    '1zgi+6=J0k?3CXkyO4*Bb~mGu>p=pZnhWRMeqQaH_o_4d_gmN&s&B9Ijdj)R18W<euzoAUeV6mTm7mNJV_s-^6TO=`=z<&@nJ+E2'
    '^0^Jm!0+iBd8+{@{FISJ2{1cn0X$P(wxA{{_i~*e|IzDSWTWIlFEZzvu9@dMFJ9m4nEPJWHe({AsAF=uzmTz{?XEE?rUAIy>$ltA'
    'N;~k<=Uxeva;_Z?f}D4JSFuw$hMblA+aqnuP5{-5=VOztujVD^xcX{CH`wmS#raF$=zk;e{DecU$-}^Vi8ZfzY%m7`?@0MA{)7DX'
    'I%D3p`xL${^#RCe2{B6_c;0^dzuNlxFF!@qw@&jpbBb=TTr-w0*ChIh>lQs>x%ce@`Ziov#EO;vP<qZg)PI~)xxdEO=0xVpN+#Du'
    'gXw)1Z47l-OMdTk-feS!YqiPV(aag&<&B_C*Y@pmzn69#ho&y){G$8a&MEDibPJtoRDbTuNG+X3%JUn;A$0}%-a!7dr|LP(^9?Rq'
    ';6>3tdGAI?WvzDK#XQ1;(qOEe!K$Ww$Tj{D!~e(h->3aX0)l<`J-+AD_pZOnccm`-LtJzE@O*l5m*tIlw+tJdBt5OO$6+-r-P>{V'
    '-U_>M>ow*Ny889)sot(56IxfU?!VgM?w}6uzVQBQo8Et$-^(4`y)d5s?J1ulGn;p0w5-;V`N*Hv*L`FTd5rR$F;PBq4;t?}GKAbj'
    'XMGN3)at)w$MF5m+1B@8W#yIQrreY9Zg_@rXK9|B{kIIeUsvDT@9$W#FGz?mx#bbCm~z%|zuk}HyJw-FH)eK{sjq3HKs#!!d*+Vs'
    'ekU`rd_DF7i7{W~&s%p5m<JuVi~FYE;=Q$v@4ax9dG3?_ChhZF9l$%RXZ?gt%X<Mbv-CQ~S$(_e^!w-lr_6o!DV7)NJma@cQQc9N'
    '2@6(3Nw2(T-SR~<KBb4SK5zTO_|*B%{S+$%9CpV(K>D)=V`{ytp0j_wv+xxlLW+W`)xCx_hIXp{){0e|`(x<?_RTxj@LlU$t9O46'
    '{Z=bo1$lxsrq-KhcL-bmyruz!-SMq^ANIGce*VbA_?$j~KrEU6dOu~B%jU!Se`lTPlv!8d+KlaJ{XQ>#9-iNaSMaCoyB=*&^XR&-'
    'tWELN9-9@ngR*g<X}@ONfaCS;bzjh)wu?zH_P2I;(cWGhf&-RK<}qvdb)Cy)i*(YaRK5glP3I1!ZSXyJsQ1+tL?6h+H<^XE?j6{*'
    'TyM<sTbyo;=k0xpcim;@Nu5hO8}oIQZ3G)EJWG%sN#sA#^q#HF1l#h~7KD8m2II*JF-5@)T<ed#{o3>%*%`6RJySa>wqoZI>+a)x'
    '$xb)4Tf*Yoi)%4$D(mR^W0Dj!v8U5M_TJ~%Lb10qr)X%8{9@QnAMeZ9il4eiTYkPfH~|Ew&9d<_4&GI<{o*_S%Dxw$#igAV!n^CS'
    'jNgPa@q$=R-Ym3Se4T&!hV&JU<vj#{&q8Nyup5#hg<Rgp2GJxejMV7OccjPNWHZd@DRY-@k^*x7Y_1RPs7>1kcU0Fd_Nm^qlX3@W'
    '+Q_(HHP<Ly(omvK*dDv`{kEU&v)guP`JC^Lc33u!=6p}f{kQD+zW4Xu>;3CFcdD|FbEhi0#6=ms@WN(Pgo(o(OV8K->pXFv^o+T*'
    'iMZw%f3(~4F7}3X(17(BbhO5?mt|}^)4n-J%!|I0_dWgBKi3nu6x$H<3tK1oT2B4Ff4_x0bFE{`$RpNYe~a%@Tg-KQf!ajQbNl8v'
    'IOhJ=d7*D_?InI&?BCiEzWq~n94vR;T)vLUA)C~8qfe+Ed&=0%(W9S3+lS+T?WxL)Cnwq9)}bBbcQ4f5Kx^)3vIB`un)vwW8fE-='
    '?2Rr{>a-4iw#%}<M9P<bGu9TqL~a#bf8BI$^fB_h>tj59DBtdjZlDc~8EbFdVX=ol@2B(YKT@ZCyf46E{QPx?(kI0oyqVWneN|pL'
    '_1IfI8M|?v<J?V5e$1>k$>-GNihYwh($8s|)pzi!=Xkbq7tlUdzxA$-zESoi?f>`q^at&7-#FB`co)L!4Ex?6@O?+W-`a`LlXl&0'
    'xMQ>LxDT8%x}EHSP<WoZF84iki{`z$d*|T}>>Y-2#Je^V1^tz|=isx1TkzPay$kKS7yb6RzuvdG|8P%30?)3EoB8hE`I*1&J^b&o'
    'bfc^T$AlHyOne0)X<O(|-r1IOjGtin?Tq^}$8)zl|M_d2Yuxc{H2q-Ys`<+Nrtk0DmV}CtK8@W`cfN(40Pl3&>8tEmoc7N=YqRAp'
    'ppB*033gOG1^Nrh&deI?uj>$7jeF!(W~1{oZ1S>+Hf`hEIln)7)3oC`2B#0FPt2p=>itvBXz$7T4$&BzyqH}saqr=b|J=XhFBfMs'
    'yNX1!WwWqP`d!{bI|~NyuRc_t$|kLyZ*kse>ur{I@lr?D|N1e!4{4`w+?(}>H0|Q}C6E?nKYsUj2kPf8<@xu1eihq!eV4Wq{RZxJ'
    '-^-dqPV~3#2krAYy!r(CuZ>N;wdNcSU{0O#Gcc?DRo_RJR();u>fL92&KzRjyR~ERZ6){0wy7NnJEZfbjTc$R{OD(y{ra_wl60@%'
    'WuLVCoHbImRy3^cJ?ra3oj+UsgqR6DWS!Q__tRviZa&8+gpO2k9<S$yoYm|1T;lvidJOnlJ9qU#)K16RGVtVea_WrbS=a9q=h2a7'
    'gY;mhyls>74g03&B%Zu_FA}LI4{-E^>#y?Qx9Af10FZKZUgJX@+z@&%zH90~?}>Fd@7lj-HF=tPGh5Rc!<uJ(KJEJa|E2d3@$r@U'
    'jlL=M+SmIK_4PHCF8F?aJat6<c(Mk&zvR?@)TLPS&-ItzO#c$H`0Lwu)?@3C*cv&}$`*Qkc8?c%)3iTp_b3}RHWvLI`j;h>UD{sO'
    'XaDKS59}WLueCvS$*Ui2GVQZrHtoja_r0D8O)6~8*rtvBy*@ikEC2iZz02>?Cf3v+I7`)kvC=f<j&`>8>#rfd@?Ycg$vl3q-QjU8'
    '^jpwxWs-Sn=67Z^?{7rR)cC%}0eF63zcz;8G0#N*{4228{Po^*CK@lWr2M|V*a1edRdP<hduf}j`serW_4zWGkuHBQyP<s6?Ar1>'
    'bo~O&T%XV9FF-@>mp0hb=6T0$zKwQ%)5!i)XY$s5%GtZ0UpuV#@Q(dRU83q-_gBX_n6AxTyRGf}4d=c8p&zsQvA;9VbgA0K)oJ9i'
    'Mq384ea_mdbJ{ub96Qxp%+2>_-(z<)&y`lOC9LgJ|Lnc-esiUbwbsx2YroZc7{7JujAB0Zmz;J!T>-z`^Q&vz>hJpZ+$;EvZwci&'
    'Tkgt0+`l@`?Oi$|zOp`!-M!jpUG<`tp8C-4<8wIW)1Pnc;%|DdqqB5wUFxvi7^GU0zxr~&?!%Csgtpf|f3<VQ(>ZVqjon%6+vlf@'
    'Emd8Jy<)n7Q~tFN8h`M$dB*nkeE$4G-8+gz{rCE0Rln77@Avm?KfnF-OyYLNuV1N`kH$<d?emFr{frg!^L{dr8S>pnmfGe$Kfhbo'
    'SEW7NwR+xr?x6T(du7Ss93<A093#DQaGW~z8zMLBj2xUV=H<Zp_&tBVGk73Z)D^zpo>*dS1Ezh@Hnab-shE#M+r$3sPTqCPj$-Oz'
    'q?7mLvU8xPOr%pcPcinlQ=dD1*FI(+%#reFPujodrS?GWN%fpFn6>ux|BNU51{)uFKvE`C3BSSKI_BeZ#`)AgVc`GEzyF{A_MiXL'
    'Kll3AXmY*uq#*;OO%V8#qnd1dk7?Y=lNbDPGRQtrNbe4d^2v%MM>6T}e0CpV>P`&$ATt$6sHE7MPX)${cp@tjq|0Q};<q=szHSM1'
    '?-v&3N4|@~9Ens4pfDC|358!0WS$&jPZ6T5`URZVK$jpYl`lcMXei}*CJ;7#5yE@I{l`3DD5#{jfPw>?y7!@sh7K685MK^S6#%UZ'
    '@WoSTr3>a`Ko+oNgQWr768xelA;_@E`4vo9s8@jghMf2xpsV6^!n+%QQAFD|sm!EshsPjJY><W&a%Em*?p*Mpo&pKMZgMK$MG0y;'
    '$5?(y_!;$@sCuZ8;=&Bh{tikru)jOo6n3Lf-A<BLzR(8#1BFriz;`G@5{wVT4umKalR>N$!(<gEIkv*U*|V6Sj!WjOaLgv#*<}Li'
    'I|~)whL|+LlSG1VlAMLWI8Wlcg$N0fcNp!XJl+7HKvYHtxgj5c*hreKz-Gun32X?u5g08jk-yDkJ=3L4u2J%^vAV$Z-M|4!r2xt>'
    'rkop09(fX8I7|X0l9t5mU_8q{7$m++W-&6=(V@VBa=ictS&7_&$;rq*prRuHu&ndpK+Nw5X3zq6hc24|zYnK+Bh~FiD@#P>CjtKd'
    'QXu^M|Mp-1`+xhl|NX!Hmw*2^(jPq6zxD-0V*G3WEU<Ob8A6h@V2*y^v<<EFB2mT@ZF-Tg_c0mOlR=aWI!Qd%EobzEssCQ2StnI6'
    '7}B^hz>2~>8Zc_e#s*N|A1O4z@&2r|xRQYc$A?aUKf-L1u`hJGA;ikte{*y~BX#|i$MehV`J3zW%l>)tfqof6e{+VOETUgt(J#~J'
    'mwWU#8|jyy^kgjka+v;RH9dJwzs#q<xlk)R>X$F|uT-jEa@CV=^-I9|o09cQ()y)s{Svu;Ir>5_671h^Lcf#Z?@Rg@OEM`dzr=uF'
    's=(jmfnPeo$|8D)&;hXvzcI^tlWsKYJHL9}rqR6Er7`f{9e$GLZp}d5(K$(t2lLj=>5|D!;iu!B?U_P$M6P;Zx1;G;AW2w6$!uh?'
    'xpVajyB#kDOCED%xC?KW-}qa8eG=`+xKHZ+%E3=EezNkDo=2X3684k1zf$<f<zGpCWcMe{KS6&)|92Stbv!`i9n@Vi&*${MNicW+'
    'b*TJx&itM|zvt2K8TFl8zvtUuXWs8Q_<J^b`bEY-_Wy42+~D|Owtq>+J4s;o1j)D9Mxk8|j4rpGW#!O)X3r^RKA79T{Xc*H*Z=Lm'
    '{rCSAm*l_3yQ4|}=r~u*<VQ9IO@tW58JeC9a&VC(lXP_t+#jNxI2;H^dSIJ66>LGp?1=;5?joUBY$T%v!H%U%;5&-q^CHOcl%Ahe'
    'E9_B{sy`7iKgFwHP&aY$Pb0@JZJ)aJQ<M^t-<h_-yq4q?9DtX|EkQKCK=m#ZEHe_)gWncTU&w1|cHQ){*@4cXVe#_(5ae}pUwg<X'
    'grV<i;WWr+jHdu%_t5_w;39!mNX>iTt%h@;>PAY1`Qf*P^T~_=s4M$Us@g{t<)X+C`801_d_Bp=EmjL>5+gV;b39~XMc;p3H$C%t'
    'x8LGZ2|%lOTeOAW<@jI!?LYt9e+o?U{?`ujXI%e!A6SbI=#mfxZPYiE;@YTx%Asb&+VoQn*C98|l*1zk>e2e#-c!z5HGCeXKq7)~'
    's9dyfr<}y87w<~oddD*sinaNVzS|p`7;W8QVM{x?W2AoCe{J(hn@{yKWQKS3;rT63XcGBt7NWk@8%~=}<n>Q^L9j@j7`TnizX|8i'
    'uUyaZ1pbbdGnT$~ITim%6Q@rk<*+Jdqom)^DlWe*&nImqK~g93ISd6bD_ST0Bj1#~a9=scDKSpDYrtOV7~OgPRQ;4ekG=^y>14B#'
    'Ezr2`n#;%UeAGdld)hp690tkD`VdRjn&^C^V+ReT<KjF$&TEZHjnNb6Ovkfd1d*Tt!@jn{??^dYTEF2bTOXeBZC2Nut;Wm#lp%RE'
    'uj{*tojK+3HHwB+`4J@xDTlGGAzv&0XX=EK(KcU*4XZ!Vmq3Qxh(djNKV_az^BuyfxU9cmjka%aL_hcMw_N_VLlxO6^Iaf*$pxj)'
    'G0&UKk|~GZj;Mo_!=w6GXEP3bS6HTCtxWmhx#Rr8TS+?u_G&U%J_A0>`YrUCtgR<>B@KLCSUz95G-(H}*I^yz$okU3+J1Ngi6Th('
    '_jw%gU-_X<_xX|tmgmuCSx?0RdRZrH&WM~RY_?*MZc<wG%W+5o3*)(O5>h}LnsRio%ap!R3;X5wl2j1u$!Xtp5(;f&$8S=EoI2r|'
    '>6dAQ*IR_NDL-j{*`M&4UM9Uf>-@-u1o&usN;zo^l_vqx_HkSOStfn3c1c0e$wzyW!@dvq6P&bzyumN5wL{sJ;GI5N$&ZwW<y;;{'
    '|COJ_8s89Ybz+vYchiTG$?4}*CPPCzuaXUUu58O*>dW`W3DDg?gvoXH1~Ax;r!Ax|Kkc}_-zW{;wfp8e&|RZ%jGTQd^N<DGz6F}s'
    'x8IVm-tPW?vljbi55I$Cx__71Gw(pRR{lXdTjdZzpT0$B?n~JmwYe|P8c1fhV;;q~Z)IdYuWx^wz@NUYpDjBs$137yT#ipj00xmd'
    '$rzs_7Zw4M8Vsy@KjkE(ypUflO6qw|8IN7+-zodvpM50??S_?b5La<SHAn{C^#5mmi$&lKA*SBM8&bNlJb!Y99BWkh{YNH%jDO0@'
    'c}Ys;<(ZJ{$*A#(Q(`FN-elJp+cn(ngXQP%auO{!>;G(VlJl}QM#}UdO56ys3eWjopFgQTe17r0!}Uhl^>*ibk&o|dAEsyb_}-g9'
    ';u_C4$Dbb1oMZXztn-c3(4J{;`aDaA8DE_kNX;AU**n00gih$^qNU7w!;3RZPDLCVxpkAY#)zvB4dprGM(F0eJYyqHPCug;jau{e'
    '=U0arTRD*Zc;+h(x`BV_2HrlF-;!f`EWah{&Z!?we1r~^aajMzG>~%p)=UDSN9~b>ZRm|o`T57_#`H7g(n03$_(%R!n&f)!T*PEK'
    'YVQa9Xw;p0O|fFUw{zz>T)*WQqEDR!DvIU|zwPs_pO65II-jyMN1Z*{_U39%xqZ(!?}EvLR9H3Fu-Vlbcou2H(TPZKbmnd{r;l<!'
    '30CS%N;tGpdU+o6(CcrLe#Yq0C*dcvbRg^WIt%ms^=x)sdG3yUi@MQpOsF3s+fiqHr=OmMoqeHBQ{}3Y-<rBUu!7nPPr~d@`>{UB'
    '9wA@2Q35wMCL<lA)1=R(>m;!Xk&ZXlv@<?&`qX}T+}Es6^r}HT*$v&>>1ShYI93UMGaB;d@?#$E8{2>FQ^I}8@%`t9VC&3d>{I8v'
    'F7_$o6ULddcgS>V1ZRleBKO$K*nnz}aFZ?Ol{-+8l7Se8$p`=E#-LOCD?<Mtqe()blRe1Y<YwGXIqM3%)QwSz4C$03Z`o5^;|8%x'
    'c~?w7`?vmNF7AYv4zds57=8!XrW;4Vsl$GOJw@6knO8!ely;cwD?>2w3y1RPr*A}K{*)KxhjR8i9<net-Kh`hPV^yi!v6{7RNlqd'
    'UeUwGyvOfzNqRyG^`rZf5q!~SWaE{ejZy7HE~%?r(aFZxwnLvU>*TjsOOmfMzhy6z(et#Q1l8xgF@lZB_+QZr$9%0nQlE3=ntc1L'
    'Zy9F{Nf<ijw$5ZO?9a;lvVN|?`N*2Tk`H1m&h%>_pPB0`pC|iExTiAj$iOQW#<7s|SB&dM2%S2^_H$N(`%n8>hwPDSu$ErO@TGnG'
    '&zeO3ToW1oSSLD3^@nd+(ki7}1FT-nb+yC%F6Wvw+Uf~cpkZ|`T%7xB*?-QC>GL=5LO;Cv{oA+39^ovz;_j58CtP*xi`Vk_`V*({'
    'SjfODj_xz&vCoZ#o_uk><JtWw=Ub-hC$s=#vNx{;37vHwReelk#TDIq5IeCqR@QWXr{e<z#LnGu5NNm^o4kKVH8enZ2dL;y+~G$V'
    '5IpV4KFoUBG5Y&D^3IH0WqyiuXj9A_#pe3cZp0hgeJ@(rbSDd*kKK!Ljm#kBq)UN(mBy=WHW>?TXY0E@Hd_)p7W>O;vJX%m?Qo>Z'
    'a<sEks%~y_BeZTmMZWmqDSBe&aF5?1$<5H2BC(_sxRY(|{4B?c;~QH4k+*+E8sm%kiouv=%kGZ!;e$GJk%hF`HyA{B{3rwH&<-h+'
    'lBbGMSa#ru1I*?g?TDa_hKr=xNqjP}^zC@G26&^LYxR$4;b&JXiL-;(8(2Sg+%W?KE@_FzU<%Q$k%T+32_q)o{H)iy^eN(fdjdr2'
    'Z;btpBn|)<*Q=<Y*w~HvS=@7m+V7V=9er)Ts4NkYyK#z5+RXurGPvWM7%-jf9N8XuTg{rq71xWjV>=!VnJ{*u_*6f4%;Rco`$fTH'
    'Vq;4l+?jTwL_KbI%;aD3Ed^x9H0ZsBl%J}0JHcp!o7|4qQ6+dcUeOuUeblra<E&gy8#e}N=&Kl+N5Ut;L}W7+C$y6iVqV=fiOcS('
    'MKCX?`-v7a+SeG{Kle)6l6K-fBtl{t&70LXlBMouprSj>q;ttK`cd5gWU)La8R>VZc7<cHCPjhm_=6@i0~^mH5jUH>iGUDyWFWcd'
    'olOZob}ZY*P%;5R(8!r8N^K_wZ{7@QG^OeTnI56(*)d1fx5XYkFJ2MbwVX|x?2*80cIX;Jm7HQanb+5--ztF_lF%4<2ig(G^I~-3'
    'Vw69OM644+x*hAccf6hW=~H~ZMesaFVcpMj?!x=#>e+emyH4F3#g&f_LD4C;bCue;oz?1#9$rf>VS?(Q`=UF@9d91Uotukj4$*y~'
    'XXdQm7dtDe$G*HZ(Qcf+9F#jMeQn<GnyCWr>pt2bX=4Y#`<3g6TbeX)?%1`*xpl4Wh8!u<Gg^qi@ES+nImLDrGeZmgE5@onB#TuP'
    'uZ3p&@{~h9-?k#sW#n5+S)Es)ZNaIx@>ij|212({Y^x5yg_grnw_><dh%N3Y_6+fgX?CeO3@UKQ^R~E4>*EtKb)M4ooYIR636g6y'
    '?Gl5!84};dEhy4xb)^w6X1Y7Ye#|Q?A$bD}4@m;0y;}*$l;Lf4WE3CNxS>+v9mjA$?nCb~D@-IVx%Ft#NFN@AW3xiqjeJ-xWoF2='
    '&a@;7dSJv_uIkte*3I=l0&m@DPwT6|rG4EjF2?L;_;U4Gw@3asZv(u$9*_L|>}f}pTu(b%AL(>GyI1@xGCHmY6)tDi3S@S`&alE3'
    'G_pR@oo|&S#(O~0x$aMEUvb=c2rsrYC+`a5%)m^!BAAS|TfrlYC<Z^po})Q(XK{24mRv0V0|o@A1ww<>h``7$TpTF)*H+?S>cw2X'
    'Fmk%02Fs{_=e)~^bAE9ML!ht{JTmb9t;7I~!EQ-N*udGj5@R`%oi26vsldrQ*ex7tRzMjBIJ>w<#^|2Skf%8V$g*r~;Tl|qsz_Zn'
    'SvJmKVh1;*_Iyv7fWf!10|o=d6mo=OSzf9$%x28IgQ)0*_DU!_`6>sj11mdp5Lmgg#0Gp0E0&Sb8K_vfMrWY1X9ffjD|CDocbxm9'
    'qkhG!HDGnv;hi1?Un*(gO4#Ad8nHK~-Jek>%5bgU?j(WkAhLO7A_rliRlInb9e|&BB{9Gtc7BDqWVBx~2r6_%Vh@~*%2a%kq~09B'
    '7*@g+1{@MAwyc4pa>WFD>gIZgdvwJRVY@p=+rW{zV;Ua#RaZP8gJAs?0Nuco>Tv)xVJ9?f!2h*lO&PDJNW==btDQ*C0fc@>I~m}S'
    'b^>$;V1>;l18uqy6+ryzDe6p4-a#1fj+K89N37Py2kiuc4OmWg0%-hV@^lW67dshg2SLR<uGayx&5lQQ5Quz=oZY!7`W5jXju%ZV'
    '9?L3*^mBkm*ho2?up8SRXhu?x4!}-3j+X)2dx!gHJg%tBCO+03to?vcYR8)~9`0GuVa8KmoS)54*MLmQfqK*wvXlW<Ze$;<4Ld<X'
    '=V_!Td}ccj27t_sZ4{2I9ryQudTPhLJ|5NP$X{q8cH)Z$e%c*x>UbRLh@kM)pQn5#Hr>Z*NgtrL+woiuBD8nrf80(>j!+fDc)Q0v'
    'y*{L<;+=0J#CKymQX541hlFm&;XlCY>_*_)@S+gkMfX}VQdBvO7lV6167zPPdR0Wp$=R*zXXnP>IhuEHB^^?u9k4(L0ksK*sv;hy'
    '>6KuoNZp?=?#oWbRp!knR>Zza=}ilM@lBD@xp7gs`r^exmkaM(pQU@o|E1C!H+(&f2g@sBrmT-|opm@LcQ^<-_stb&Rp;8bVgu>K'
    'd90;R4&@0K9+EbKQ$*N`v9&AD6451gYX$vf7<WD86_IKJrB~u!#Dc9%YzNr7Uh8tWC#xjE=_{+x`0y3;zOb&9IAbHrQAT>d&XF&_'
    '0E$3wpH3iBe<gmvAdO{A0B3B4ckr`h?F93~;ch^{%0?H1zAD6kr=*hPZD9iK(4?%~f8=so;VI}4sjNg&8`XeTAqm{NNcM?TCMhSE'
    'pcVLgC!U?O)*UKavSW9cE>;AobZ86js|b&1bX~#?Gm?T)F!v}hicJp7+ah%nARm>l!?H0Ab?ml~BB)z3odQjp_zSs+&A1^t0r)e('
    'o_@bFk-w?S)1Cf;^I=6vg&c*g*zIB~?1I9V@f@2L61E}5`J5G9GP8X;V$c`Mu@gq<E*<(NDhDJGofIGvJ8<DoH|avRMWSIYzO7YI'
    '&`g#Z<6>qk_r1}{T+)h-g5d)iG($bsiBFr!mgcISp#|t7K$60?OEwiZb{`B>3`>3V-y#TwC`(;}t~}^sm}2|s!hB<1>fC&063cYS'
    'JE8^mB`<vDO-NOOKlBXGg>#x0r^${5V-kLV1!EE&AY-aUKxX=lEx!G$(EfUwx`5RJl<4xv1sgH0>Muox1Sc_|b8NO3m>4c%Qs7}F'
    'zp4-U834SP<%e8&xPhhh3nAN(jV~9v0hK3RAFzo`%#WN5Gj_Nk3Mzv*puw35iXL(s#PT?V?8ZttWKzlp4QLu?O!4CZ*B^GU!9Jdm'
    'eM1&xY;DiI3Z(A|3FddWz|oaunG8Cp>T=@9my<}aQ8dcDO=PuopCWwn+=K^eNG<;GZ8Bix%H_8uYINMiN!wGriXv#-BD@)gA`4Ap'
    'a<2@2COr^!XECqVNG1+!U<<-9<8ccGP)$G>yN4u<0sM?xyiYWyqE3$rs0fO<8kb<)+~=?2k;x*EUWPuHH6V)CUBtKitQVKaz|MtJ'
    'b6m1`q-U3j4F<6Ya>&mLX~5}O%k$#Mmrx`@%%Fww81^kCJ<$>}h!B7=s1P~1eOnH|1lQWLQ%tbQ7Q7rOx8<Bq7!3LY*J#xPF7164'
    'NeM9<qxf99Fv*MUKM(UtCrE2>N-sbDq3){G3GR7pG}Ug5=xP;sBl;5B9nrQ#g;ST&K~CZtz?0Z#H5LhTnHeE3z9lMIC4%r2^<`>+'
    'oQ1109_&(bFrs(!J|K5677e74EyRzZXv6K-%|Q`2S2s1qv|UhY^Ls^nM}6f;;u@`*#GTPG<2I{A9En8l7!pPVay6ElSzp}>?TlGM'
    '&pW1ws=%5|6kA`clI~DXichJ4BkrP(acjg|)HiG`xoUO{Z2k-%e?7Gr364=`bXE!EJ;n|(sNE1DUQDD8*&nPM={dSm*%nTS&&X+1'
    '>o?KX-2@3yp*p_s5w~n7<YOeFZ;-2CV5D{r5yx^dSy5yMq(cMmFG$aUQ5}aq7sYOVjz+r-Uto+K@AilZzq*t2w_{iuvGjIAU$~Zb'
    'tV!IqJg5e$u%ly<GzM`P)P8#zb^1#@7d9p?UL70Lh?%7m-BGW&Y%HTgu8Nm%EU0(_e5-lOt>de}BCvRL`#}V)TMy+F)87b{M8_Uw'
    '&rGv|PU{wVQS%`&2ol4Xvysr0TGhFTRo#b61tbSiFuOT1UbmC&q8`u!L2EpX3e_=raT?W@j^@?z{Ek@UJ8ruXi+ne43!%t!Sd20<'
    'XqVtQR$r`uecH)Ju~YX+F`dkL(Sc2lT+BNRMkD5PL+%C<#3(r<ifhN8HKOG9!GZ}&tYexQH@Xq4I@iZe_&L{7<NXE~8$$T)8G-%w'
    '3{%>;i7V_>h{iOkzj<nZvs<t+jo>aicCJ&T1ujf5Hr>39wUdjvYTbzP+&f}BG;<vb(MZ5shkz6Hr795Sh>fX=Ab7cbLEJgw*y)|F'
    'qvqGsOn~TAf+?ZV85b{RS`v*@R2zxL>zI(nB}9i8d!q^(HufFwSv~e%QNW!;jh_+5C?=PEE3~(BsgP%xdqCCYGeMWfH`fJwN2GVy'
    '6#DuuVk})T!<`UULE<sHR^8b6JCsTt=$LFwlhA#Dym3Bvd3Lh4C2Xr#fXL0Jl7535wJaONL>ipmx!B0`9YJIf`6b)TxVQ{La+2^`'
    '<NgPitCwYeaRWsDh8?a$w1wa__8!4$2tWF+k^0D6$<aFSF_~-@q4oQB9N?(Qc9(Rdb4trt9#=+}5IP3z`pn6d7Akw<biv8@<fHEx'
    '!a7l*Txoj@jU6jhM|GE3Cnja-b^4$#a}i_gDS|_QiC}QKlJF(+!4`Uq4pRAa;^kp7>0ES|Aaw3;?bO9I)rp6dojhS8{jsrZJ#=tZ'
    'yu|2)7<YIkdz>tW|Nj)zt{gI18YDr$HX+_0JBHWPp4^ySu9nC;$rF0WZoEr(u2Cj02o;=c+&%o(l5V?TM775hY=}QvW%t~?sEeq$'
    '01f(-aBj}|I2U9>7~%ZsxDJ(`a5avHr)W%xax@}*yaZ;W8^s~s&2PmS>(QJoT-^yDK_`H3bYF}Xl=a|?bL*_`3<B;|v>syPufn+-'
    'RuBBWdGaPmP@)e)f#eSxi|~3ROwK-8K8Uex1km8LYHC-BqHKh2ppLi7?i7H*lI-sdmO)M(pb4g>2y$Q*Q)WOAbSG*?tDI#sKVu~G'
    'Escl-*fZOdTrh~xd7oTB4fEIxaeU*vz~kJ8q@h3SwaLvI!%Q2dNXU>3v!DbZFiZlmDc+R_p$%zueg@{SEhk&Mq!2HtRC&{W6u6>|'
    '2G45d7l0oUMMOm@e&m>~T9=Z87K*j{r>-OmiPtFJnnVLMX)Z$!-DDohK5kI<&A5?@?<7NeikSdez_RR{WLcTh_G(QE$Q8n@8C|HE'
    '?IAL?NR4w*Xc(ziR(vayR~F2p4ev)PH|SnhuC(pe@07;4v97FjnTi*~$39*$8_Ru+=}%f?ynPKGT)ce}0%F6Hzr_k#A|t6}^j`U_'
    '1~zX+lsQ@JWeP9-Z#%PzbBrnMdfJ$I{IQdzp&K{FjTgp6i?<|$3ZFb)@UOW4k)>_N64H!$%IWW--7fG)SJ6|edPgB@h8_i4Ud8Az'
    'zHP@MFU#9z9H8^s4_ic&cvcDIU-&Ot#o5rwZiq#GMi4<3NaJd>6aCpLxP}TiSwczFDi8-qkP%DwD3p#xgGrz8M1189h-n(!dX=<D'
    'IbGTUn(pU6KHmnL$MF4?5l{vuKLUtsKT}b_pI6Y(^8ur{9;BTe;pV%5kQb2Z1z7Y2=YH+1)Ncm!UPmyd{0kb2i>;lYUA}St1=<pD'
    'lwI|`H!|k@cFn(j3xKHV03h<WIXw$N#9c_R1;0HYqj=j|1uwsU0hAiwUV+Jt=LRp^{s&C!6_m}DvI4rX%~wEo1E6#OYbme!``c<W'
    's`MqO%HMWDVbA@6l?22IIO>9x-UA$ytV*z$NjMS!KEHOjwAXJFigA8Bux9rHj$I#!DgI<hvQh@8FR&}O3KDgk=Y9)TgKLlP`F(=Z'
    '-?Enf92D2@w<jK;Uj_Kmj9%a@6o~?1r5%A+gXi`w46!C+Nu8AKC%~?|fL*ld+Dp7{zg>H=ewrwyz+Cs<ABgNLU!@LJ`_Eedu696@'
    'g48fj=DY3#wtNO^SAF|#zjf}noESGe*(QrR<zC>pXm7K|3j2KfZT~!vs^(mylAgZl4iS`q3u6j6p#7^rYH<n(noD$$o|Kf)8Ve|w'
    '=e_^dah=4_bl$c27T2x6-IiZ~tk%aZQ2+};w}R%-)B&A7dJr5FRK?eK>3SGd=Q#jkfEWEO5XQbs2T%LaybC;$Z+zQtKkGTy0s+C#'
    '#EJ3Nccq<H{rsyO<Jnu|Fwx<w3|QIk1<)+GZ_RUQNI!Soh|<se)_L5|U*Cl<%)V{z2!b;<fNAH^o0UM+?e_|Hxfm-!fp04}O~8Wm'
    'N3cuwJ2#?~+W~Mtm||N22Q&VJCFHkUh5T*q@=fqf@RPj>in_QU3IsuO@2(~K+#PeT!h%fW(&*he;@aB<q6H7-eY*PD3-E-G^8M@i'
    'C~%}e$pTK2TzcZDNfO{0CZ5UxJbuc86rycU)ZNrsDGPi^c&0(`Nw1#gmOIY@V#I4uzYRhu&jC<`8+jT$PtN#h5PKF&%G8fos{RBi'
    '@y`@&MVj>~F!<|#eGl2#=c;VQBmZYmdybwds36hzKV{Zb5Dfp|Ci`P{cDzx4%&0zzF;6o1@OgC00V82y3I-(gk*_%@QyzZLNPWL8'
    'R3d^pf*4Pm^*rsc_Ji(9-~9eW)nYpP{g(4>QYQJ1kOTdc1=IWt<1~>o0GcUi;|`#6jsmr$uR|b9EJqUtg`Xk*5oh@OD7T-oa|J+Z'
    'I9IW){gi#4px)tJMT4Bfu>hDgoU07y9FA#7W9INXqa;p2CwHcMLg;YAIWq}K`zg<~->tt=zgcIbZ|tKxTI1>GqCb9n<lY<teG+{-'
    '1@GQbt>$p80R*h?+O)~OyQBI{0m1i%qY8k_9k<BDs1}m2pFVeG6l<_6mtr8B0*;^ff#y&zx)~pJn{XJ;u1=MZ!8rtV<p`WzKzELZ'
    '**(V}`evJt{aHT0Pd}k4tNz1uQQr=^&>Vugh!=K_p5%M1F@G-R)N?I&_p?so3_9y7%Ww+fxuZYK>As)$?>da7=Inu(LcjV6g(l%s'
    't*hytw!Z5v(~ST4ueE*TQpCy}SAl@~Z+SjvJ^&5X*j=o}z6yxc^4#*nbAdUDsO|z-26Kt76Hcoz_*fn(FF{6&_Ftc`GnasjPJcS%'
    'ehI)C*H>S{Ac0PK&E$6dQ>M*ZfKSG{7dR_p+Wc*FEH2dfuIvqH&?B~}BXDeFt=ONKHB2%0Bj<izBh}>%=Z=)30jd3r2?VkMvi-F0'
    'nH03}Kc;I7m@e;f4~L3Hs!_AlPcHP4&uM21f(r=OKICN~NP0aBAWIt7$+z78mObG9d>E@onHSmQxpn6PC_TN|?!V{E(NpFAwVylm'
    '?DZ{?HQn{sSPy@F7l!7aa=y2_|6|EGeM@GsnhPjx&NsqD1~qPE?)~)#<Utd(81W;8=?wwaX%pG%xqc_AKsxoGvfpkU`r}@Tm1}zT'
    '-`Vcc*WS?7)LMnA&Xi(#SH_(ChWm}Mr|i$P<6I}KYIcE>u`JE*JOV6rkK7THm}*iUta<yO+<7b`YpnkugSMc{?2Eg=o7oGG`#Ecp'
    'BwaW3UUSsBg)DBKRh~TpIgztfzV~;o#0qfQ$vt+CS_8;>*Fbn(=5TBiJWw*>&efr0bYA=qp%N)Gr=EW|GNG&gupb2~F62Wmz-IgQ'
    '@@5QI=^pdj?=emu`4Ei5lyu@s^uXyedc;s)N-)AXla#-$+Ryh2LX4?^V4btahhRz1OTl-@fl>YJ_Zi!3F@}N@W8Rq(TwI>J%A6x('
    'kA9KBH&Z)BF8uV*yHVI9{xMZ(_S?^Io&v|_n~8fWJ?C-H#QbHdvdjT>)C+%>ys1S~&U58U+*zg{jl4)Zu8;TwP5YJS>5uCmwsF(Z'
    'o_0=o{?^Z)@s1dQn)V%Q`l;T*d_1M^_1|wv0EcPRQXc=966=OjvcB<(6W6r4Ss(L^)rCIt6mU24+WVjLaSPgv+GqOgfJOg$9zFXm'
    'K<o<wsX(XqZ#ipS(xEFcNi`;(8IIeP`<dxkGsi#U@K*h-<5$pZLMx;aPx<{v>hU^X-vYJ%EaCaqSrr*w^^tKBF<t1JZ=HXQF%ejs'
    'J@B|jb5$`t+)p`rr6fyN0~qCa#)CHRUS*(MmVtHM0#>uWwS`>B^vk^$Ta+n}e#%+5m+y+*PJ~*@pw-&%b9DRb1!cv0xNZPzf_QI9'
    '2bQ2#fL4NCul<!T$OzEmt_!4;dcy_rfpbdyz2q8~qztV8>jsdoE<zBKgb0@~74EoKmeFa4h-XPjF2OLas|R9;?zcnq;JZW!p7JQf'
    'Ov(Uh?(^+MaB#i!(>vY^ae(tn=$AIj>jB2Yo&I|6gSW}JedGdHICa1Jv@clwmK4A-gs9?nY(*6LUEs7_{{oL4&jxmC*BapY6!3ii'
    '0;GX^2^tqobh%a`CJ-b)82<u|L(uT94?w<K0n5iB&OrYwqK6CUL-iSu_AdSG9bex7_Mdwx{Xb*@?*W`6yKqq<CxZcMbsv6zpZAYs'
    '<OT{_L<xX&{Ki`n7SB8(Q|c3)wV{3kF{;EL_k@UHsUJk{Nu+Y;+9+JG&U<aTcl_L@8jG%$I;Go<_xE|^gG5j_fXK@;Lg1(0%iYwy'
    'c2C;${=0yIdM?CYVM??-2MFrD0XeDu1cFqV0EeMz{%)vjO?UBA&i(Q(ov9;F>TcTpk%BwzzhC|M>rD$-_f$eMbAL_!x|>Fx+@6CA'
    '1gYPMoWDIg&=0hC!m(;v`MfLVMG;{{W<KhpF^8P<XnO9{eTZ{Yy(eR?J?~C5zNu)U4@C~E9|uY>wHJP8_c_bC&o*^c>^jZ061&+~'
    '=FVYXInSE9UY4P$eCuA{{?b*ocR{FWdW@{OoFAU$QD3L_v^x+%(-LPsa;`XMb$&Rep;3L@CvmIHN4+e;S9P{MS(2tGNtyBW-kovu'
    'TznFiUT5l)<tFI&XNdwVuDdY!UF~Tx-QsV>Uz6Y7`dM;EzRTaixqIu^80)O9CtQWI?BuRF32)$r^*qC0Fdqrl;{TqHHe2-HNBaQd'
    'R^L@+>%Z%qkrKc><x{vAAAKL_9NI!TBh^<P_O0jflbE^rYCn&&Y`*vVTSvQbu7Nc#Pvg8V0zh;E-}UlaeK#?Q&a%g%r-Md{A1C)t'
    '?XEX>Q1>3>OzkOK%J|mLS)V6?Q0QOr;)1hV&?JH^TxQFQ&bv(hm^hqePQ2X07NamS5etv+$hol4IY-i&tcQTngbgmJ7{P%pHr;rA'
    'N=O5*Am%I|zR*-h68$ZdQzG&fy5>l7s)dR*;z(Jh>$`bMj9lzJ>mm3it85X>O_22><<dy>&;m?8q7$8B8bjQS0Uz*SNWuur;sf#w'
    'AoLFuAnP_)Qn+$SxjEw7ue!OLo#IMktyKsoo^aC$uFC-8(C(T*BihZn%)$9H1j|_XD9#Io2ri~&BF|j07Aye-Xs^rg>iIprpf~>O'
    '>2>mAPSd&v@iWanmu)(?NoBL0&4tG;W3Q8ndTz5|L@(ba5Xu(!XHiP0Y2h(2n-R|n-Q;ztK)iSiUUQp%CKqy)6HEY8TmmEj=9J-f'
    'N`xjh4#Eq5in${a%{j9s2~Y^$AwRo>GsI|37K&WmritcZK&U)4k)o!z<MO<|Ogz&_>sV&MmzdjbnraTk*=%i4B$zNb)`BVgA_k^;'
    '(%9u^Wc-BzVbzN<hlB92_!fX;f=r*>4@__qx0I&vddi@}&@~y!xCsC!{9#Tf)FCUdd@YG9XQ@U%7M^t%ZUQTCQ}<_Z9T1keZtUfc'
    '0HibqN-Ca{Iqw4rvhL)rJO#+`3J@M~yib5*b7S;oF0?U|$i*u#6&BVAzBnW?(`@g2risTnl7<*3EFCvr!OtF_zGauLA{caT3BuHH'
    'l87$8v15u?kEkUt{lWsAP;#MMz8GA2{p`UkPh(ibu#$kBo+78)b8IcEV?Ln25OfR>U=V2xr~$NTX3sCKo;IenAFMb=D^cMKn<pQH'
    'Dq{<a3UCK34Aa2<7j`4y(xHp1&t$TQi(QHJm_pa%8klk&<S+Q0^dZ(`k;pn+fD~h!W^^$3;<i~?6*FP|!D*-6B{(|^W=cc=`8x43'
    'dThX;e~We{60u%^86D0p#J@XWVXT}6>d-SneG(c`tJ=o;p-7m5^%Lq|Pb~tVtHc-l3<ni!ohe+R*xj*AY_^lM5*XTui~&hMAqMi>'
    ';KW*Zb(XlsO@+pEzvmU0ya~l#Xq~=|>BcEnMPcp;CI}9gq2h<gT9*_e?%#SEIOLQy;=z;01fXmYOhc>{$hZttL?%H<xyenrBpMRY'
    '=qIJ8tNJSWyJ9Sz+wNqya4BBMJ_#hU#1>&BD~>G>pA~!JBJgl!A1uNVS3JBPUn`2|BwA%fulK}RxwB91$lw}JF?)y`^$PB%>ZXMS'
    '0$~Mr=?Te@)Y}5JUtE$_u3U>)k-0s2kuY!SP7!u#VI552x)jAI0-QJzS4$LSoEFP$ThY;>uMpz6ZiV9$RYE{zG$vc{oHs^}^O$+d'
    '2iYCyH5ObxD|i0|{0;2tBDdQLCG8>s5{p*#U?tvqL9_@DN&PIwU=D&zOX?!!d^%xEpy<welNSVx|LbSUWU{L3E24pe;HQXQ!g|R4'
    'e-ZezHhRfcMnz*{olVA<SB&;GmKY)RsMQ%AigZ#$YX}b3EsX$AB_hadS|MO;=VIkjrTPVxjB~MK5m_Q-7cC$ZV*FoRIuWFcJLgW+'
    '(;{EmPQ<PgyW-r2+#^rX28q36bRaUyC1goM6s2)-DcQLit55EW&qV0P*s%cSt@zYAb0bK9#_VC#)vMWhiI{-}CEGtLHXn~hYK6|*'
    'w%jdw#a?Il#1RL72r@ZCeU;iw@Z)+Io!n8DL6wi~vd&(LyBJ-TNaV!yEG|bYK6>9aH=ntD`gQU~i<eEMp|4-RMKRK;z;VFvp~bI2'
    '8tF*%k~}4-GV5w_`&u`u`}}qbH;I+74=)OH(cdB1`K^14{jHePMFt!Ipj;HtK)mh{=sTzjmU-|B^YJoSLzOj82^F5{@2f5sjUThY'
    'jCif5?{jW5n7bD-Pb8c4!bBcxuSTaxM!-zKTM2b(49w<_m|pLf?yP>sIy3Y(Ar<SbfM<%J2kTbnC2!*sg(!2QMRPpEt^i~<F`=Bk'
    'M2g;=vobNX3Y&oJOi%L0IY=OeL}Orq_CWWeE?zka1J8@AAGFdXe2BS%%ojV6x_Ue}>ORg=f=sz=q|b4u+WlzKu%B$4xHxWiLL^M}'
    'xT7@b!rg;gFAQS?#~&}o)`MShlQ*VIucqg;9ZoeE3r>+f;3{o0<fR2eWpR8;*YNoSVWrSDDsvg}#~Wu3i7&o!I6n8>&8Czhv6z~f'
    'bq9JiU#RaQfz4~U^$jo>xH&QZY&?%(F*Z(4GFI-1eMr*7=$V9tZ6ZvtJV1pwUl|m!HNXLeS@4wy5^n;$92(CUZ~YmUub}vzPO{B*'
    'ZWQKE-uP3bDhU^lZl67|pv~uWU4D|O9g1lh$7Qt>d_74aAsL<gBoNmjSa#yDCs+QBr<I7njpGzCw#R1^+5;1nT!}Eyxt1V}b5Ntr'
    '-43cINVq8PRhG%k#pISoGq)T`NfJaqB_+|)*{yq(;cG{yn=+9QE<*;wz2VW*a(!vR?0Paj%iBc+i&2gj7M`cPBe}^KnR(q<rSu9A'
    '*`np@LYLOlc0G94%EN*6Qx8za$VA*}=d)a0<u&BJsACAVl&|l%7hWYJJ}zcYD%lveT{o$NC(Jt8;BR?xdi%Uti2y`${tQWOaIc?^'
    'q4AXsHa6?y2n*tFzB)c`4V*eeG9j;lbPVA;>b2ByXvcjFr9XApLAD=Z_SPpd4CPe9Xexs$Rbb%ZzHNS(i|{qWBOQ=em*LsgZgc6}'
    'q&EpN?i53K+r3Q&U)#MeJupI{qr+hJh+7Cb6c-6N(3^xK!LyUn7#(3CKNr59Cih6#3CONoMHIXn_9!%W{9`gVC3LE|pE3x={RDG6'
    'IiFD~LF{T9kQ51BUuvfn$NqRd=HX~UTZEg495+j<r2Q7`qxyhPMj|OFf`G9tJZnt^NXUQyudK*N37-}uo4z81kpmeh9~oG?1tBl1'
    'Oy)ceNICLYhpR)uL<<n95}Uhn&90}=9}6eRO?j}U76iBm-!<T$VjQAjSAT<)8)SuEaKJXT@%#*qkQH3e=w2+tc16Ghx71<?%Xu`q'
    'Y=iN*_3`_(-^HC%<Ys4a)k%xDddPfBfF06tV&#CstBSDtCNhr*InNFPwe~bNYZQ0x>|=gK1n?wy#bbmP;I_i@R!_fZp*;z1x%)Y1'
    'x?I24tk9f|eXA;P(t>ay=8-hP&=gO`Z3o<4Pjj2AfFXBYzkc^@1$(hOyDZE}+vsteGN-il2~2@3a}(Q!&<}L{gr5XASfMM|!||<q'
    '`{8p&xsdx8&K70&gtI;O7v^)_`>!%#M<x6T7cz`4_Y8PMkZy!8D*RC~3`5h5<<}B=GfIFnug~+j&quj_34Jsah5n3^q<efJF!%hz'
    'e{h*sxambm$8gkc!h)d1ttgy_dDs=av^OTa^3C5TW^$7gV$|VAD9qEo+MZz-5u$eG(#Gb*V#+Fkoh37zok8(A2Pz%k%O7%(TqoLg'
    'FM@#}YCAjQ=w8^B#0gywcI6r*@&tNnS@xR}dNL)=Cr&^!a~4I+L{2Pmpyiu>FrC*#&8sZb#wVQL-db;*G~%|4yc_$C&xwJN_=~Vw'
    'VOuf6Od(zbTFZvLF#)M-WDN)h!^<b)Y-3NlK4nD8?y#YFO(<n#);WfCuXb5Z=VcgEC|%}qmtjaY#QY*uws_AUdv*T~U4=S~FPz;l'
    'l4ZC8p(e%WuC=`tulhdW>6g|p!)zLDx;VH<tO_+DTwW!Jki=D?DepYt>=$md<$itj82B!x!+n-J^gQ7Z*`v!4w9sdMb>{E#2#o2s'
    '92=J5Xb~5)!Z2{p){-OavO*osQY;yMV`zeA7*Cd*#TRxjnG@qIyQE8`*mrR+E<jQ%UYIfy<<*Oj97rEw6O#&ZUCpQXq2S-ex`YYA'
    '3MeXzb}_pwo?`sWU7>7x3;UmSbMd{3Uo$V(7(;h3b1wVvmYo^yvJ~?a6T-y0q<Mt5VLXb%fa`qD-*zmM+%|Su)-x5@(+IBI+P&`2'
    'm^iQ?wpFH-dYa>AZb?hAK|br)jv-}PaSfDFH3=IO0FaALVq(_BqKqeCf}}vznb?CdhEF|CSh#dawc<A}vux;h6dH>}D*i0nXv!jt'
    'p9DYD!cW3VUrP*!PJui{J6SAGT?On$CjiU_<GXQnareEDMB^?t&k2Lh+SsIT8J%10;Vm-a)iBn7pPmiBLwKNp8++Q6CI6<v4OQX$'
    '{p?YANtAiE?$*F9RCT;H4jX8PjPI{o5_<sywI%D!WO2RLxO^W%OJ{T6tddvy?fw<9GdJlHGcBAV(N3p?*!bq9$qvQjgl(suHgT`l'
    'sY`6K^zj`xsYZf+?AN>Dt&}C7?hKjw$Egs|g_d>iuUn%AdkENYuCj|TI8p@Y;f1dfE4OTuRiR;)WgMdCoS#wm(S(xy_8hy3hVvb*'
    'P>=NPI;ZKGc4m6U=0yZ02zP<1#IkgrUdPD3lR;HOG=r*v8>apx7k%tH>tc>0Bejd~_#HGO?-vjHeHUTqx}_0~JX1Knn4DS4k?89#'
    'R)37Z@N33`C?9Ap91B`Q*w(A=k+davo4!fjD5~d>oa&q0c)Q^Ufcn2w4<a2^NB|RABxV*DAvkXp83@&ly50oM40i#(KJ@4q4XVQ3'
    'P6bHP4a!EKt3<8cPFVOhqWK<Y&!Gu!e7`<L>qS?Dvt6_eJfE{@iRZZjEc~v7T>E0LA~_N;ea5nlQmpx%E5Q8f_TwX%UT(n@fyG#h'
    'ivfZRbWzuzK)ak!6Ez*_C8VV3Zye`mxW)+5UuBq+lKm<h9e1mGn*sMi^|{{AUv_x00X(fD^>w3p55xL_a*ji!z8FVxAD{o(cC&B1'
    'Gw7`H<1f^BI(Z2&z(mbesP*I5)(UW8iuBrjw=!E+-RpRgG;vNNd+212*x9^+6H`{i3ketFgSgl<MlxiGf=m{dkk<5_eNREjHJob%'
    'V>L$y%nGj%vcl59G~A|F^j+rL#b#*Uz#GNyaBX$Yrmk0Bur2znXiRTb11-aB5)AiUk7r{x2k;NZ7ZYJ?I?11}h}D-5T6WT-@jYXf'
    'L%Ifvh3FR(N{EelaqVmK=ttV4moZexKSI{qI8@FPmZA+%k}=!2xE=E}<Yy8w^VCERjHJaqj&b{8ammc<eOOtt&o%mK=gPVt7C5@!'
    '!!=cE<ZIVW4VGl%5%$YXN#0`6B=R*p2gtEIp$Xc#*Ma+y6vhw&&Ww^={lgn>0(q2(L$u*_&Mm+bTfCM}e+4PQ0<sPhxHmn*O6Zmh'
    '4yr`2y9q{2=+pGC?>bl}{4cwsb8w=0P4M*FZ@J}7J7SKbfy)9Ki+P=(g*}OC*ycET>$0QOg1gR6{7UIEXm7ra03Ed<Y#H_pt1uc_'
    '6}!fcaV~AFOVt`8RZUrgp<E4zIL)O(pf+^<^E)KRnF44Gp$;Aq``wNc5Ym`9CbQly8R)5Smp`+74M-2E&beb3ANzhQ7;%H~kewVD'
    '&+BQ~-a;pAEC$;7A8{G|tXPfv$O$>i38sB2pUW%^iq4-$3KH!6#o#?FD07D)XG#*@fR@NRf*l+W1R+;eNG!TBp1T82;rt}-1FzvG'
    '-VFEOHg*=Tg5k3CjA%+3=(+j5^Q1AFuRpsPUx~jn0cJGHgf(@PnwDdTBV=9g#+UAs?P7*cviTUdkB`Za{B5eVZTEKtqVE`lD;ZKJ'
    'i51R^PQ<$!6gfxKFG+B!FNNG}ZI&a<q#1i&Cm?=LTA*m=6k0HC8ZTbvf5B2m_7=n(IVZWXY|;xXZ9y0P#h^gq4vVdQ75b<Yt9t{G'
    'Fx!{OR_93vW<sU~@_}8x_*om@Q+HVJl^b)DE2b)t^^);jA2!2pgA5aUqbVTwB%pt_lO#;_oIDKM{AQ$iEc2(cS;4c>N?S4bEhvT7'
    'ZxLod=$+t%ydZKSVm7<6p>RNu+b}W00a#CS%W7nl7_nHk;tNdb?jeI;xQsBYL0I306o1KwjWKORzxQd6nXaMyLBiQE`a;j#fX%^1'
    'citAFp&}FvEf@aG0ecg=9aM)c9H-%9yw1lb{I+@)t+NrJH=<iN&KieAKE?~eB?qd7p$p2#+#nl?R3L+vmJ9a0MwA-_hz)zBKO1uD'
    'g;b~BR&!BP6E}SW4m!d**(5=naBSTlomo|Y4vN6sNdE?sl|+%PH!B=!uBsq6+Q<M9Cmf@8lc_Nd^#(SP1bb~trJJ#slBZ?3rN4~Q'
    '(Wb;@p%WtHvWiW{v{aD#ENoBk2+Z918gRo2#}?+;>Fi6m{4zDvrR%bw%{(Q_A=k7b7`y1YLB}lIN6xi8mD4$y($POi_|;@Jhlfh8'
    '^S{W*AVtEBjy$b)!*Fuw%ou98#&V-?Sg?WfHT)7Y(3S=z0eQk2wqEjiHFOGc%C;#_bH-sBIvutr;u9ceZc`HFyiK;#NGzG-oC_r5'
    'h&Yr~ibOW;Fz|-9NuJwv($?d@X&8h@!mpaFuQ}DE#TwCe8}f1_?!L*$Y!Oo8MCmue{6-v9jod^K>=yrBMp=3=g2mgqBqdH=`Y^7K'
    '-Y$1e`NKxw|MjydanggqTHNG!%{8XU@0z<yBX1*|!%cqI+}#^m(ST2GN_NaboQARIXKdgck&R%`0sGfPFeG5IN%b1kVIwu^$l2V;'
    'H33y|BRH6t_%@{p&d-L3fsBKsu8s9T&!QwZDOsgolkPB7gF_O;{TvYxOM&7IKcV6ce_oS(H5ZsBC2Pz@C2e0M2l#}Cx&Q*A%9JR4'
    'V2*<;`hmrdJs-ur8iObhZ1@N(Q`Nl@AS{Ad9F7D<m^Q^`T6lk@*XxmRHrYr}Uh6W?@-x0TMDGo5)xjQW!X@53l|(tKx`8f;PSv<t'
    'jRf1a216{oWL(Y$RD=F3uHM!}Erd^wi`<A_-de5SG}c-%J+~$+^t*~*a3r3mp-_(8^qMTHEUqd>fV?ib7m~h1Qr)0Ia@oKADLln_'
    '8V<Hsh)Jl;aQ(v2-QX&P((*jblEVW#1|Q^_TznA4&5$fA^z%W9CTd(UalW?39A~C$Bx6iy+;{=MwB|~UjN+a)8b>3jkVJ#E?n>n;'
    'aFmAh^eOA%b|iD`yJ*8($aNq3>^i96BE<hcUd}8x1BVaiVPmDf&)KFfBX*XHi~S}yFnb93JK`=cg#=zW*@Y2IDlbjK#Jd13t-DJ2'
    'Vp@P?(w+_1FGbdGdyY)?<E_Bx3_g=pokOg3<r#a^dE+Y};Eo<}4(vM7J>a2g>bHckeN|?Zo|l^gr9<EjN(XLcvy$`<jhZa6VRj<A'
    'R8Cp~SY@>Z(@c7fPCRXwVeYlKifR>F=??XI&ox-eJ(#Oy+#_F319;pa+{D|^Gb|nVNskWKVNCA>{zT}NJDhqbXNYccvv*b??0|f6'
    '!w<q|HWby)zyej~gBm(3EWDB&b{07nZpNX3(&-_%Xj_uqcNN)TXMbWN?;Q%kc7Mfg@%=4cUxln}T%0{uutsn>WYZGH!X@3djJ?n)'
    '!nfOCEM!m{pBZJV$3*oNA)2HJYa`UVk{&__4Us2y`u=iKGxHW<0+w{~q6i}=!`GmygdF*WeHM0VSXXi3bcrzsjX`B;cka9RYvDp`'
    '0D+)3Yux-{Jd;|pzi10g!23~ak~c^#gtRpcD**&^4VNd}ZmI|{HcJHxTU^5}<F{d+fG<rApt{4FOu(S$H{{JNVH&CMTkKa+<aQ`B'
    '@d$TVI<a+hs8&HabhuanX!Klh+w_a?rxer<5htWVJ(ul9WNClk+nqcezZJ6NP;sH%Q(Ndk4mJ!!9lBSn=Duy%-xn3ZIo?IU3PE=R'
    'm(&|dGPL8IX7F}&PBUOf9br1JZ@Q;zN2=gK-Ce3uZ(L893&bXxrfbr#KsN!2)j9d5l+?U}8HzYl{ed;EVgKrh>}cd;>=Cq$B#fO?'
    'XS(NkF*i09D0ZwWSZ9kczAU50sJrLid7g5jf`w63%%bK_Ct1H7m{q}c_g8^28z|I@LFSlc6mwy{Vz8oE;IV+U$z;W}RjXvFUQ*<q'
    '!w%Rq##d|`gMR(x#XO~jZ42<%-scr=5r~QZLy<l}oI7QBXU9rIpQSvYNzb6-X@!x#WYpY;owi4<{+=R@-eOlTS&|LdwuJML%X0W+'
    't+ZWK;)g1hR4FpW!sT89IxAi-Q#sc!nb<iFij@73my)5a14$t(Lccf}CWo3Q4(YRQW^=6HQ}3;0T^%M=;!r)`Ze=wbq^<{&i}`f^'
    ')*aOKm^qxI-*H>9j2L-u!`KLdtrT$Jx%;v7pJIYwjkUWV8Cb_HBeny$bQOt`#chBoj_6w*1n{PkNR&iHarc}I(n7lgQ^<Mpd~hvM'
    '5uo!P0Vu0g*sL%x5*zvopx>+WR3F6XWyxF}LnY2I%;iKvuV^%r0DiGlGGDjnV8mB9mii>)oy79nGcp9M!hw&2Ya=jw;p4^#OS0Jc'
    'ic%aWIGiMMnlJHe8tAI0%DBgiFmQ9k5m;=<=5ykB^#?f)vD6q<yomJ?q;ZYI_OmX?Z|E<4W>R@e*JFTa4D%0|GVI6q7P)nzm}gWY'
    'O|t}!OtC109zf*{bi=n1%3;sMkyeF^&}RwB@0v|Qd?0E!;hfzeW~%BgyWT4cnx9D+R26YHnuKe@<xs^|Ye%Rm>_)T5BuuIH6n7GE'
    '3Rel>uv5e?O@&S$EYcXBKsH0=4}+`;#52ZdzYP!AQ?TKug3fVT&Kr!Zn*pa_bYwu`W)-V(a3Rtd=67;w!Zbfk8ip6;p1hb?by?}%'
    'dgLWrZzXS^%{=b33voj;vssFEosG_zwRw`l<IR&`XqxC9@Ex{%UxBgTlz>D-`I*ho`RjYycd_VC*pX-8a7NM3m*{+EQ|bsS5U3B7'
    '2Xtv=K{Kcyv(x>Yc5fsX`0B{fdhrc+6zvjtPZR=!C{qPkg=t2*tk+Y7rs6ue(I$z@=<2)xP|>bM;^}M`sI_%3h+p<q;QeW)J7_I0'
    'Qxm}GQc2lT#JZM=oDvA&q`9=CJ+LwTJ|&+V;qj)}gGTexMhHVwWot;gm_<Mvb3&0R3>(}Z2y`aFV|WQ9r-8x;vt$}eT}T&TZ@Vgi'
    'TYhG)u?g4HRh$asH2Xt7_+k1rGBR)TNM!ApTMB>$_ouO7xv30f&*=jZSeFJ}`imVcHrKH+1QEOqbHSMJ;fx|%*)T2E`f{D~Z4y9{'
    'z(W2f5}NB_An`a>$DhV<Vzf~&mg8weRno;nmTj4ts;F5nskTj`kt9pHLB@p5>M`NQRir~>$uo3UQnsN*a>1nJ&q&D5sLy%$n?~ZY'
    '8Sg)2M>n(jSY)o}iof;tFuo?Hcc|kenx=K@MXYp+&*>!E*oa;@v~)7mt3Lj&gzO|rAKGAHbut%L^cd2Jc-tS#i$6WnGE2MHTN&QS'
    'FDFaKFzLsjlT7!(Eyl-(TX$&4OmNeHcbK%{13F?1LG`wHKhD!d`uC?eL58{K5|=WxBu6R_Gi1fQJ<5U_&ycT(?St`-v1Fa@0QsoK'
    'E_!A$FRr9)`Pn=M(gDaGFvm{??Q`iFqYmnUI~+-1M%iHF)IzBrxd;vY4UMmOk3|O`h*rBXwc2H+PS)g0C#(3Jy01UOa`^HTd*Vgi'
    'V}=}48*G3e^E14fu6{h^X5YOMcA1!-Y?edo@XZ8VC$ZKr&lj9sL~xV+Ymk72Y-Ogaw*_v211>iLrbZ5%p|2<>*BF{uGqIYF$=eMi'
    'Vgz+vbxD3*#W(_oiPm_rHrV8*9Ly$6aN3t4XY0S;*3Cc$tOv0Yt!-#Ldr0)o6y{nxG!#P<M*s&Y>xP!(B%3-F-uC@0lcn)22jJ+H'
    '(=NH$%}BR-dW_&zAtB8}lNzLMI=)01k#aHD+R@GU*T0_$J89#r^{)NX&$I&h>{~elYDf|dgdtY@XN_cZ`(aA8XbLzpYPWRvMV7sJ'
    ')bL*07#kAeOh#kq>x`odv#+@x$8h@x^yDfTOcr)ZJl&%RebO^!RCbHm;3n&gJ(LwchVm?j1l%1}<ST-ntYw5u9eM~^>*le{kPJ^8'
    'hPveM`_ItH8u$Ac%Jm3nKPb|2=*NgKtkFU6yW|oeUjtP|o`NUB<W5dpSY}6pjcdUmpb1DrO4LG}%q9l@2!Ww>I3`x{L>`M;k|#JN'
    'Haf#c3U9@B8ENzJL_V<>ncYa*0J@aF5ztNf3!rPR?<rdZQQVY>IV}AdLRarb3>vtF%x{I6`VZi#QAYcWx-Jc5s#g8ZE0A&>_?W8?'
    'WY%NO%U-}XwuH#H(Ll*nF)**Kqb`^+PKE|LOm?e?3V>~HoOi5~DwA6sokA?u4tU*UV8+IAl8x7$bpa|nez*Z2s2_A%NG9TaZ2f~+'
    '7Ls@lnY(h6z>Rf^rUA`9ernA4*(?vs`B`QY?56Yfdd5CHMMdE=>k#gzJ52ZMddMPx9}hXsNn!r5VJf^Fc;hV}c`CH0)Vg{a<%0yE'
    'ag5&u#2X}Me${L;EiaQrPG|7;!{p3%J;$vdPtUl-BN|8!srSe%%k0qc+{b$~p8FFkw~f6dh}Qp4g6Lk%T_%VoZe?IH7_syWG5<lk'
    '4%3rItsg@ALntEja3-i~n5jH&LB|X`hC{;TWV~is1~u8efE;uqSGQqS^AJ5ad~hEQkf+2w4Rf78F6F2Ayhigz#U7&FfYrYu6}anV'
    '9LygUOy(fk@n;xD!AqxGL~<~2qEm9P?BjlJq3+FD8*&s?7T()jit{tBT<&>p`7t(%nuL@E_gEzX$CC#sR<*m>vGQYO4>gIS==wf1'
    'V9F?1^G7%BIma#c`B>ou-vTxhaZ3QOXY7mxYyd_inpor}Fn=6ue~9EoxBCN8z^M>X0fzvT<b&W6fQlTR-h}II4Lqke)+~Z9bs8{G'
    'vdm>mDAy+mc8!_HMB?H61Eai02e@WjX<mPaiR^&c4p&L$SPHI??+qg<9#lZ{jA#3ks1>V;${rCK#IyVX9j*=zXT>1jh;PGN0eK;*'
    'p>wr=zNcJ33<4p9jCn(Q<$m$A&Yn|@`(>>D4GEVDKhvjVOJAR@!Yj<oj2N0m@GVm04;=oIw-Ts3WRpW%fdmN#h^otUudQw6saQK;'
    'lU$YlK+jd398_SV4<p)O3dxQw!aBqGSVj@skl_-3?)}YF@wlu&TFQP42l<dcuZp`;_jwDoJ0PE?M6=j623KY6Q8`7pyw1!iX~k<h'
    '5&bq1&~o_2@Gl)QR3M12B43kekenYtx!{bN*O2usjI{=|xBD0!KwMt(Atm0Ii~01H&yZl3!(loJ6Wp-hZ9WrJ7d8;;LdG~rfCjD4'
    'lw=Bg`ZnU7i#Z|NK-IT~QRZSpZ}XhUJ~9dV;`K5PTSM+KBtUmG3uN{lQv^I#e*7YqnVe*cbQ?n2CPoJUbQ5B1fCxkuUlOW0e1f*F'
    'vW!UTCIpWdbora0h{g3x`al`C7Y58*b|B0wd>imL4e%5D85Y^NFQ$x2Kyq)n;Gjn(kUN%;DSgm}tX%lVz;ZeYJ!s_Z0^e)?#x2SD'
    'JeZHCkRb0dXs>;kWmmRD*`R^lz77fW=CX&)!(=LW0~Uo9!HHxjvN-D!V@q7}BAJJVa;RTk>BXRLUGv}pIC*tXJfiopAp0|H{1TMG'
    '@*!j&17^{z<>{-j{w}koorDefb9<clPZJ@Me40F{W);UHZ@>obXD(Rzv`VTP-L#o=9UtR1{wXdwZCwN>nH&AVy<}(+E_+mu_(F7R'
    'e|@MfB?zW@CrM~b8wUK4sFX`U_<fq`uAlp4GX1%NJm1rPLPy6#nAP>3z$fo!4-tc0Wk|SKPl?tuYdVcxR`>!{@ehwLV|g=;ws})E'
    'GmwPU3MAb$bNh2w_^P7`<0C4MnO9+*XkPJg@el8_#1V3NBdlgP^6&8)qX?PY)o?!2@C-e{CVZ!7`#vE!qZn+j(*o{wZ;dZuY^JTR'
    '2`EQd!qtZkBG@o=!&qbsJgVmXP+;g<Ns`V%UgL!s#m$-TQ<Gh&ZSrby<Hx12UNm~qJhZ?H$Lio7i93EgO2_J%9fh9`yN8sX0}8R3'
    'kc<JT7p|B=oIO~@0sGm^DjnF+S7KsVq(ukvAQ>(`>a}hf^Q5S^)yuQD$9j(=0}F5T#ymx@f$0J;2YGBX#laan@F3&87_)Rt3|=~3'
    'V&y{l%2aEvn%E(UwVe}f2Ym@60JK$;JH8yO9IJK*=KAVn_4gY)JaGLui->TW6ZUX4)roLJ4;r6%oYIb?6%2mH_r<ng5)aaa%)?f8'
    'Rr-&~DIgKpA|VGAVEOdk0YFSw<#ipNADwX|p;$l;U-@0}IqHmqwVEXbQW3T&303t(#vARofkiJ7je~;gD@bxFbiN^}ZICeb4mt#;'
    '!7hngE;9X*#3qhy{iHV;5^uTRS|t;Q1Ch{_KX4$LqVa3DCL*!b9hM2$G6{3|1HPt1w+g7H8#--~xaRY-(@N4u4n2+W&6CB7tw5B<'
    'A7I8rTU7<~K9f!!Qw$7`ozOX-4oPu)J6(zfW8D?4q#LXqMW~ZIW|B^kw8Jx@Sp=YcEim87-#$#roSqH!vSdTFE=d~^=`|LwMZfK='
    '+<m&gNLF}?`Kbb+(;y#}$bsrz{j6pe-z^(z!@@Gfl>@Z`9d(<y49E~<vjBczpq4Z4f>aJb+yx<pR0iY-l&JtxShR}sw!`273%gnP'
    't}aLMl5$}bmu<489D2HB|C&WLn>xFQ@ABJ}&DMc7GJDZJq@Bu~LN-HbQCj~puw7)ano?fKNqdif{_0zyfgnJ}wN*uMf4OkTthzU-'
    'a3Cts+=xpdr{4aHRLb1G*L59;xa?b!eCIiE@LdFB1^kY7E^7lUDNrc>cIk|!4(iJ7(GH_qjaspF!KYwegvw9TD~HDkN+`mMqSNYf'
    ');Use;8(sGKxLA85R36j6oMle6UlFXPX8ogmQMvoYvrMXbETV~1Rrk^@!$|HlASbp$aKJPn{Vp^k5;+rUh{KF76RL}{b=B=ubSce'
    '%QMy9FYn+*eu5(30YW?eB0f1~@^BSVO8atqec$!1YS?Rv*aWg4NrqlXtEE1JU%hfok-O)V$MyXdzyGLwG?4wc!cY4rZv~#zZvF@;'
    'Bf5Dvxw2Jx14SK`3lcISb6L)Q#=~3{m9K1C$DAD1{(0ZK4iRPYY#e^+<lnf9l`?tC-okAr*UEu&rv57Wn&dx$Mg2NCTTXt_(|*10'
    'O;}1N{5|F5gM}#OP1JA4oSgJoBjx*_zv`dg-vm~%US5wd)?k$(HhOPz9P{m8{qOzByZrk51QJ(1-8W}(e|MG1vul1_a`bL<%_jO7'
    'D|LRG{I!>Ls?B=GvpqS@MNF>X%8N_x;`-jS_15?140(t7dpG%q&3QZj%5Pk4e&3M;!JLF|faXN;QvMkS^Ceck-Q_su$ig!B``gps'
    '*mZj4&meEHxwFh$T)C(hCw%2>Mh7qlc3dZ2X2XZ`+gv73K#AS_X69x#XWGL-{g@j&Mwu>^;FERS-@9@l+n>`v<E7255zFw&IX?(}'
    'Hb=@ZDscVtSN-$*c-@R&{wXKKv2wu^IJ)h7tXgf)%IjhM@BO1y%y@Eg|5e`Y%P~KQG&C3ZAk?t(nlH-p_qX-jcl%A%{&`<FFx}kj'
    '1ixSAP9e{^Ilwz1itYZb^Hz?OU19MlC!8KR__g`Xxw*L=?>)a9;t7A~=AbdJG&Y9RN$ziRr~2EP|7z#78<)LtjVUJ{vhruAUzL+4'
    'zJzX0p=W^!s9fy)*10+DeElUKxcSn}2k!SKF10_tZ;@Yp{*|BH`3NP>SFU!xz53q2>oIM;=BwT0?>0}(Zh!6Ps{j3-T;t}W&DHGW'
    'Cf~?}_x;xV`;347s(*f;t!}{VPdR|Mube<|&c6q=o;klB1KM4)=eGsLh!)dcVX$CwxHN|eqX9G*f?7qeGlb!upOxj}0#9W+mkYot'
    'NLOGiT(VDvV|*cSyv!kqMaCm_i>}19a8W_e{c|9d0kRu)JQC#9=J*h-)9`}+TO>@~n1Xs+y&SBcAsnHoDC}ILJ|3*N`U-M)>YktS'
    'Dm{t`Vi%$Z>VdHRuR_(&MWpGtj0)eHp0|h{Tqa+IQ$I!!F-yn>U8M;zYE6Vxpeis<e0A$KfPn?b>wF#TRxx*Cl)nn6zKBvS{iEiI'
    'Yhmj;9;k0NnW|eq1?5H%{0ysm8<6h<k{;OVFF}GttySHF{K<&|=THw0sS~-@6ARf1SL^wL9$f2h-TG-1PK9u91ZxukPw>77nFe5o'
    'h-W~EB=|G@A1bl0h0~Ypb~0i01Tu6?iai#HehcNv-3-oxukPs!hlSt^;?0fc$Mk0hz}f&pX#i}-{^$2+rxlQ!pcufH6&!)^enQ|+'
    'g|qr%uYqp?-y1<MfAI7TQz>Qjl~)sYHQ~`P&5lQZHm+i$jH<C}J7$uB1Ht&Y%NPfW1We$|y^KYOFYGepBCT9Th{cY7z3rdh#%jG>'
    '*rWt6$6LrLFmuHmkHL|#<Y44t=?uBnFFd!nZZJ<h20T_{-<bT=zYDv%W6l~U<lrv%IYm=DrNa1p@wXt!*yuQXzQ(F;V!iJ5jiC-@'
    'yid?R8whY4*>Gb(6A9e@A<*f(9O-%*JGr}l?wH+3znGglL};)jbo^Z-DoWQ(oZ!K3h~fF!Cy(xo`7FaUECTr=iu60=w8nW7#CAFN'
    '@nc<LNUMcUkjGANnSl%A$0*VRZ6;75bgl?r-6PX~-r51zFvfkF=xgLDOERNV#fbM)w|;i8|3qQ-_61J%b)sr^#l#%9Sw%}4CR9JG'
    'R`8r_3JxK^o>ErWd74lFYz-JSPqBV>3@q14WlnOHS@xm#BuLg!Rz<0|)`@U0L)JQ%tJB#2ExzY90eB28Z&-Nn^I>ID$Zkwr>zH9v'
    '_x$unk~+{B9=g`hWK`|Q6>JcF>g!{mhe=HQ*=GWc=?bT$CL^uGJ+E3=rdnotwts%>el{pH7F>E!%3gs8(IH>$#+zdQ=eO?pX)aQ@'
    'vuw1ASue)TZ^f_nYwLar&^rqR7^No)%~k;lH^Q!@gjMHL=%ZMgc$TDq{i?)%dw&XVywDxn%5VL1UXG{YI>~VcZA67qOnub^>|Rf?'
    'Dj1zQiPZ$?N5Y6Qen%lQUi53XUPIUeW|c`?+H10!#H~W`Pwu4gV$>#RR~s~uV&<A4!0go(%+E9<MkK?u{=MQb=7nAhH2sDQm1qSS'
    'bXfcH^>k@--mBP+zn>i=@lWSEI}}E>E_vsL5Yy@$9hMXEt9yPLizT7kr_v~RNjGgBTfvk?r|w~zB1tuA3t)Mw6bHB%;Tj@W0T~?k'
    'zyf}MDJ6*AsayrHAD_k!`_DUy!%Q-%R*c-exY726xC|5MB+CUHTjyuTYDks_c$&y~4bXjWfJ0COVWT0)N17=}+hsa#Me&`dbe7n7'
    '38GHB^_mM=3H4We1!KP_L5>mY`y!;>V%wFqcwZ#JB@x4E;WS7k7zkmF-<pL@po=$}!1%w4&>&r8U$*UPp%!+-Wo0J5r+-C4vJ3{%'
    'nZ^dg?}R`)KRc}uiGn!cNG4y`?BoY#)8Ie?=X;8~kUM07tU#V6j%6J}rLW!NMaU+*=3B0YfyN54=^&pI>BYu{OXt-691u9Zj+`G$'
    'i$9r`*|i%d8T~f*vG}I-h2xwSszI;@uK`xD*|+<|L>W>>oXBe+3E_R+5Ep-Tq%Enf=dJ_HjM}>LYO7wSh|D%UHui3z5~SlY9ECJg'
    '!&F`wBx#q{1=ms)pGuqzq2Vc#BD;=}U75eXo}KX{XSA-oeSjQOeyCyMM0$7C@hX=Te7UFY#~L+NA*L<Z?vb-ZKXM#Tk_ha6-dIl)'
    'i(>HIsax`jOy8qYmJwi%fA>1^%_~l?st`FmhPj*SXJ<l)T*6PugqXTA)&!}xAe{bx$$FP%U5;!^>s8X!p|jSq@agwv0frR~paDIs'
    '8vA6bl%jMbV=J@3&O`7X*`XdHUyK;vm>&PHqNDAK6@Oej{4y_Nj*)02x%9)Dnak785@%;h&&9kncxN2eE!gb_MXz4|iF&!y&Mo{5'
    'r%x_QS=!01W|K970X+doIr&QI4f$ZH+8q27SgATcUKcw>r@I`u4w;5*_Bh}mibhDUEEUSmD|N^?!Aghaf%fFZ&Ii#?O>l}LIR37a'
    '=qC282K00ELFO+}0u1wkRxp2TPNeM39a>R#0Hg%Kjh%4`l8_<~0Po2UqS0qedtMikX<uwxKyMzIi1WOsYqVyP$66{ubj1koIHpl9'
    'd@&d7vgnm7kxavah68<fi?IQ;qgE5@xYT&eTO2i$fqKPCg~Va*S8;jdn;ZDJk4Bx*{K&r<!MIc-I7~0Aghlq~x$a>()TE_W5!PKI'
    'IXb%1C)x_34LtDY+>=6~!Zp(%#VVO^;sP`ET9iU@GVq-LNCQB)ZOsD6QK?5#>S!i4wj=}84!7q{#n;zwpJy4QIGpOK>q$V$r_OZ8'
    'rD)=uPGM1ynEd;c3p%ZF!8)4>%X7_>g(b^miuZlgKesCF01|2w9*3{f5)2maQvwgqz}&yX%%>}kHY&AyfQO!29McJwQqvfg!*FX3'
    'YXW&2q0>&@7t;yEtJyZY`3U#x3aDEQb0W^<>D|iErQpc9+YvRZ<ycYBh}v}OW>z(a^h#RU=iFn*7cS+1XL;y<b-X5(s@fzAErh<F'
    'q+n3$Y-Nw<I!{u|J=~m_f-l-4@g&GBT2JllKxLs&{0$mc2`j{LIS%RW$V3)N!s&DJ`cIM&w&6@rl5Ded6J@>KBe#wIXr7R)?3p}-'
    'h54v4v_~(cTC2q>3h*pju;Se75iN^I;H@R0DB(mog0_>@&`qYM6197KA3Y}5C`#i=i*_ciWp}0u!?}KbZfP%#)X@M_WOwaSZhNF2'
    '$3;=*^@uW=T2kk;4epQEWuCX7Y`6Q4U^RQHL<tW%`Xf|zjrl0WT-KF<_W4R-lDy6toURFl<F-CrpIDR04{wgko~-ew$=+v`xp`#}'
    'V{1LNEwvE}dT&~XZ|&y_yk9jp`|HN=jWV~8JHp9&9`aeE%FO-P;$Oaw^=SN579+E{B#oG!52N3n$MqWJD%jUGpL?(mr8pTT_gE*D'
    'Z$49<Q<H7fIyYxz6>38LP0n>*AP^0M1ZMCz>)22Pv;2(gGN%n6X+jV3x93{gF{zNE-Lnkxp6eP?)3axRLNlO*l7wj$F;#fFNb|G}'
    'jUc!*eZO=4b3Y9G)K^*it<6sE*`R=g%TizGpWiQFaGn~|p(;N+^>jO*Mf<L`OV0Vv{VD47%-qagPoWJhtl<e@0?+Y0|D=Rnhs->v'
    '6vcQP)XlhI&#kfpZTa5|o!M_mXP@8xhyTFUe)#Xc{p*k4fBXBdKYWz<YRWyVkM`9UDaWc){`UnL&-eh=b=w45J7gcpW>kh`Q*T*6'
    'HsHVUQD9WRPb!y@{vR45*}#sx&nXnlw!0xqwCPAt%krTWTPDlhkf$?g<j+-8DhRkB5JKPG56#}T)j&!6ZfIw`d%uJwK<|e1UUfl5'
    'qlK?8$`Q(mS1RvA&P<W9qp4M7Y-oYKTXzp=65Or3e|hvG<FJUgvnYp9G&_ru1x53>C~r{&{4R~V+~c<>>R*H*^?9;IimH8zY^tIv'
    '9p&=RdAg{khjN*BOT8(dx@d|i1g4hWNO#jdQ8X{Jb?+&q(>c08r<W_|Rd;hyXuD{UE!wNK_1rD}uP|{%xf|j1Ic@VKOGla>z4?gV'
    'T25q<kFOyGPDH_OZmpqG;FRaFEC8+sh%Pk%-P{dmT68ij)x}S#UAHJZB<qK_*Z3+_^c5YE+o?~Z@Ro*PkrCpCU>*7{H0Y>KQd>uq'
    'g;rxi>8b{4GIx^FJ_Y8eYx*tZLGnD=7$5R9=@y|}ijYo<PZe1!!W|Wb+*J}+6gQSxS%gsHD+$M?L0S}x*Xqj2%5)ah11is1luIkz'
    'qUs3W@xCB$df!58HQzlLGIM03>lS*`d%`f*6m8V-FM^#$XaL-A$>7u8k&XJH(lvcf{=GX*7-(6YIzg~E!O>)<YKRK%CXZOox#od_'
    'eu{s}Z<mjzNh4QGVF<;{<hje#;qUk2s?nnHuCRqlJhhjpr?v1gi0*GH*o{^d<=SAAx<|%>iLiexcx^Rrp``s-Q0d;sNp;>c9Qb_s'
    '$W`iXUHg~c{_@k0KWOp!x>qxejrdZ{+C8Yfu*qxCdJ|fh<kw_VeoYa-;p+r0DsbEV{S;V|Q=%C(m`1@-nt4K=uP{Hu?ryRybWJt}'
    'zN+igyoXU^T29uN);L#F?cyklfv-Tbaa0e-2lf>rZ(DMyxr9t9jFO8bS#>9BY5;7GYH?%CUGh1Ld>erdb(A$+)^{l^4l>Zayj*f0'
    'vzM5WGTPPDLunfNsA5gJg8s&;&lLn`zMEWqbCd;^XOP2CF3##as-{IYS#q2Iru#5+Wzov2;DF9;^acxc<;I=wr_U~!mzsWo51ZUV'
    'Nvg?de3r`9H5IR1@aKnx)AVsZ3=7N`Qb?kxUMPPiz^%T1KmAb4^A@V0^-giIY6vK!qv~our1r|fH&*2Wo12>Tb(Aenk0)Vn`9vSq'
    'n$1BFd}BI;O(J;ocs7?y^iM1gYR1{ol;17x0hOK<_X`70@m{@~cCf|^u_0*WUX5kiUBBlU%H2P!^epG{#dIBZzDqt)QN&pN&9L`)'
    '&??7B)p9B;=ELaM)6|yM7zy3a;34<iSm-d;z`MyU8l&2KfM>w#XY6K`ImGjNI_aVOn#u~4MwckH?=AhA$7gl<Z-4&&uit+C`CeUi'
    '|B|-b-3?K`x@xQXv$`mD8gg1Ti>d;wTWP$I3kDZ|Xbav-)rI`0EvqjsR$aD@!g_q)TdOXL&_(Mh%=+2VqbvKemBuOpnjzy^Wc#6='
    'QFJ3@K#PpUt<Y&vbdtPCk#DehKd}yN)o~Q@%WSbA7NM*1ui6`AfENX=P&q9!vwS^WehZdW6k*t+Dk_TdAj4b~Ox;?zi$J+s3_nFd'
    'FL2RCAB~g>o95O76#;*@t%<5PZ)}cw%OZP1#@b@*<8F+xr7QR8LAv#VjG8S=xM=Eb!Ly2@jawptf{dyaUTU`zCq>h1E1+96H~2<H'
    '<v3f;ToKtcDy5iHd$B4E-HjRB^p>nv`UOv~+$H)7@}SgWD(T%=1cr{W!0eqZF%0g7WvpG-Eqx<d@kIv2mLcwEp-+8G%d_(GegfIP'
    '+ct0XymD|~4Aa)~%LGraS)!bzd$RJ~G{Pj><bKBPTi6&HSByFZT+m@Dm8X{>)VPY)GtIXxFRV01BHv?PIZpFfvpKUTmYbel{uCyl'
    '$CVsubis?lnW&!^QF(YI4o0@tMMYClbSZ<etJPK>!xl4A5#8xlM7GGw+zQhc(QxFkc#i37DXU);M?$;2$P^`3R<xpQr3i|=ldZ=n'
    'I(e&bE?ep~o(02Ig)bYsPo<Cu{G3+F_h38JVzxUbq54??yrOb_MC*!9*&Sn9{me2dG{<I{Rd{+gY{q*EA=Bfl&|*d@XI2#4-a;5*'
    'wIaMy6ujJ6#5`4F{@!~FIpd;-+^W@c-#4sT^)4&B00&#a^rBLb%{r-i5hRa_s86;EGs@(t8taDct94r#<IN7E;k_}uwG<#eQ{={?'
    '{a2KB+cF}HtV0CtilS9pL$3%`ycMl0@<g|Qlts~$tpQw=^4OL2PxDClx}t2$meE-hUD;R!f(!48C9csaleWl{&3oL#BCkyAUKv8p'
    ';a2T!2s43F;yI=(^L!c+%6@Nld=>d3S9gAaO|N@g+6}Z?RMz{-G3^2Ojdo!2s}B_E4tdi%*h%!Yq~@6CO+A&yX?S<JpFn<Q$Wq<n'
    'R}7hKJ23tA=0bCYT-Gg6O^EkttBf&Z9Bv8M-!4JTd&vFRO1DSw74&?;(--|!D-C@h&^Vcb&|~B=dadAu=)C(zr-eVFtK6eK1Uzx?'
    '%HHyM^1il^wSoVjFCBOkIwcyT^M1$>(*bf;A)-YD8ABf4mf;<;A-5nO(LGM@`>JZm=XHK@&wvo?65@!_8RJ{$X+)?<IaGLoEKz6@'
    'Z(TIlx8xm$>b3H(S{Zby7|h!HXCcO|jfbcV9jiqMJiE0)hV1vPqMn%S&KesEaiI1UlNChcKtYbJR8lCbff{&t|Gi%!lm*#x5Ww|='
    'KM3#VcSnSGagChiJ@me^uz`s6gfLxO8ST&o*lT05FAR=N??N$Hy)`z*+22BT8aqkIGuR67g|cs3dCchSTlbBkbCpZq1M6X{PovpF'
    'iO!1duT^bZM8lVSzNH{Ep#^1&|1AX3@h@a?qA3)O{?jOCKbp9{yX0Sm++gnsyiYONMPnDgyD1cGXCKVJu6B-T<Zu-QO9Rh$;VSp='
    'o*YVAZl!{wVkz8)O=voA#ezc<XDhH9s-U!mCmVe$ksD&R!J_bdld~OxHFAI2kSDg&QNM`NXxd_=V_Ozn7_AFor$VMh6wHl_Dc%TX'
    '>23ToB~Zc+o(zqU#b7<`DA>wA!9P==q2QRwuL?VDa=gMmj^g)WuTB<i2m<V+KwLO0#{kA?rps~n_QDatfFat3*@8FVkF@M&ZRPyR'
    'WfRZ8%`0j3!8YL&Ee|p!;qua0k3Q+#I~-WF0jU?XVN)2G^)aZlz1!T%U*zv)xD|$h`lj%lZQ94)3*&H)kLSGetA30!oJ3Er`rH+R'
    'Zl7OaLgz2?_ZXt%xfj2}ByH#SHtH&le>*MA%KYu=vF3QBdpB&RwPPr116KVH9a<Qa=NJAx?v;ocwvy&W{`_v3DVk~u^cNxIH*wG5'
    'g?1jpA%E*D*_dO(cZ@kL0=Ns>geI{w%sT$+#Y|Uk_2c=iuO#4LS!@?F0#rBQjohvLYo3dlpfRP=NxMnJT7+j|<0%SwJ?G`GFafl&'
    '5&%@F3N>UK5;iU>2FEcjY<kGGQHrl9$n{+1j0w#+#tJSgqnpujF%taOJlApE{c&0?&F5l6x;P^KoOgbuuUU5*;(Sd9#OrYm^RGTH'
    'e??dThm$d6)1LrB$$3hQ{G6BL0@D%`Pj8+kD525%@%%M=^P-o-W=|19cFR5p!A4FCQ^@B%&i3EG|NYm${QSd5?5};lxsodNt5!C%'
    'fv;W3A}Y>b(K`<XA0dyCEmL9Z6_U78u%IY=h2mGi{wjsWPuU~cx-a0XC^ACr;HN%o1;?TgWoW2;O3x)I3R4h2$m1#W5MX))2@C*u'
    '3K<8eJOvX2y+PGlg`j4iSYm*D6tf93fPy~<4EET404lNUpzqN<uUc6@!%tdttb9Cu@YZkt>E~a+{po-B_WN%?e0-L#Ix5M7FL4mw'
    'gU@jajb|<G8u?nCTe_p4EQ&&1)b7#w*>aeE4&r^F{PSc2f$-yKn=433hl6<A?@{3DA+Le#eK^Y1)M8yy18+}_i$F0ak-!Z`mR^5j'
    '2kG0%rN8uXHc{$Nc_;wK$Vn*o)ksU}PXpWPZfKK0?VwzF`%VbWN+v`HEwN&UUaAw?p^$;t4tRdzf>{bcPE0|lB3Rc0i4D!hby|zN'
    'u|wQf!ST|r@rhFeQwO0ZN+nn;siq{7MyUniDk{$!4!E9=^$4aP!jIIi3DR8vb64DWB}j=U!jutCTB$|aMz}@-gzB}=AOfb3El%^&'
    'fF=;zBdt6yXJM4!m3s~?mQo`Wk*GocI1}~B4#7~!Je0Q5Mta1l8**UxGddus3N}5O%0--EBMvx-eCjRPX7%C<IO{D4pZNg)dhDs8'
    '^@gwp(J3{wmJEl`g9T3w0_C!4i1{?xE;^@TF?xFse?_(f83GD{k=?^bRwM5viN?arXry2?9xyBv!6EFehg&=;|9Z;2X6Y|KefzFZ'
    'V)U=uE>O8qeBJ%16k213hF1Ef&k@3bH*pt+4B{qqR)}P!DI^>USTwzEpzfrR7bw)pwg{@(6f6qFp%hQXes4u94`N0dzuA(-ibjeo'
    'z);Gokvc+UNTDe}`1i3PTfOHm$gKI8#6;9vSnZ8O6e@8AOP)8<x`}=czg3p0X)XmaYwF4fKS7g4t?S7%5O%!DS_{=2xi>U<nkvCV'
    '6RK%F3@wihpI2qhO+l~Fe;PioOsJ*{1d=)U+FTj$9GawxU9LQpjQtQbSF7Atg^U`o5lFw`^Ge98@e5r~8B;vn{OhU6%+?zDZtM;y'
    '7AaMMH`8QJL_?&_{^(^9w?ITnpu|9<kY9-Yps729p`$q*h-S)tj)|xnUZX~i9HS7bpvic_YxMOLbVb`O$fF649I`nZ-nsVtrmBWm'
    '2){LXjXVLqtT?B%hKB;mjR_`;uBnS-(!26Y&M1CH{=KdLcyH_X#QBo&jvo5!UI5ki-qwrrvYY7yTb0*X3E+|3MBWaIU1fZ%%35i$'
    'Oj(^s063THtN40O&%GT-(`{n$t4@)tqeWTmEok&lQa8$CHYWT1JY)x{C@0y9QkGkM6&4N@XSoW3q2>Hy@CL$CV9|B-XX5p=uWtJ1'
    'KYsuDmmjou-Tedt>E6RHxu$pj(ovsKOcs;63v)J;TXmYoSvF>J(VBkt9?x)`ci8)=zMI)l?v>KK!v3FShsAWb04#bEsg-7xLTwHs'
    'wK+?#$p%)EIb92|>9Z@z=}&<1Fkiv$8@`HcqlZd8tVJ10Js$zk>dHT<X<g^3Oe)MLqcB;Yuvth{3xWmz3d>ZoxK}^R{TW?At?d&M'
    'S6Fj>Oc+z&O{7lQy62>Y6q(-1M6Ll%X_|-xD6oSxF0NDfY*|T723XRB$7t<MUy?#Slm$=iH6P(ZuGB3R)UMPkwyLfW9F_sRPzmC^'
    'pj4kaaJ3qm!~6X9q9~QWB3^i8h|x}ysR;#am-R%ScBSj-tFF2|IA$p4p*Q#~1!7%JH&ylAEr$Q9)f@q2tz@XoZCoU-B7*_RFj`1R'
    '3BXi*51BnYi;!8`Nu@PK1{;McjdK1fH}2J<AXhApNRc3mMWtw034yILfxIX$?cjwh`b0(zp(jEWCDormJyX6{*3lr$`eGRqs6=UV'
    'MCzpj@u0`Gbw-RlFD&<`eKxiFH11o9{Zaml5MQ|bSUZrF$FxVg#8{Cvi|}do#HlrgR1uLmly<uO69jyECFayq&kpOdE9DhosX2vo'
    'R*J)g1>x@Vs*l}c8Pc?_J=Nb_xoC<j;BjauCo;0R8;YuKX~9G1O*RfQrOQ~v9_?}wU|Z`-dHFp=xDwdVLRP`z(lwrKFDocsuzrCg'
    '$(BX#5r)mSS+w;ms!L-q5v>aDrE!DFE%TUX6H=MS)e%~zl%)k;FL-M+?a+i|#YX|!A+85y1PXPC2NgQYWEs}kgR_Ggb~+r+K37S4'
    'I2Mzb@SI@idGuvKF%Qop#B-F%h_4EvDE*h0RbkgBMa|u=6)#~*D)nJ9mTa0l*rbV@*y%=HC)~GM!C8PU%1mEasm&LI{g-F3z*{zP'
    '4+L~=27q#1%A=Tx&5+OPb657MMdo<7{28wGGGkMtDZs2Aj2ZwrhZbMPy+fP?Zw_9EiCbl^1+}Od({0Qu!Cxt;bc=mYEj9{TaKxzv'
    'L|-w2hJFDK+b|WG9`<le(FAF(2Qy<8%_Lva>zWTM@Is-0sx;{^jT`I6BK6cd{2@GZhl)hLRx=P>9ccOpp8$iuKP9~W6tK4j)RT8W'
    'JeV<OC1N;dTf-d;|BU8*j#irTKl23jk-%Z{QP=APVh-ii#9_t1@AEtv8+Kk{<J;=1%Gr{ei_^)u1^;kA&2c>^_$^9CLT>+;um|q3'
    '4Cg!-7Iu&0`Wn2|&#QB}z3&!0B%_1H(S9{F5xxb>yl>3pg5$9<gyCkx8_5?cellIbA*RKuqLBy*WCb-(^<Sla5&yd~{Go8H;CdS&'
    'At254VMQGV`GtzM3xwwt6&K}qbs4X(M5b6wbzO=zlzdVsU^^snH?8<e^u~sXZqd*d;K|@75!&Sw%<m^ODz#-{Jm<P{iYriz!Y$PI'
    ';EgL~M}(;=TyBoaP|ZPQa30nF*I{%wT`*jqpm|w`$>bJY0pS+R%hi;EMPY^0F76}ObP)EbUMu5uoL6UtgWprTnAnUqbCZw38d2DE'
    '>N4?`mfN~aJVXbq?)IQft@#}1n-*$e>!S9Afm4_3ScoeoaSvb6D0{RV>M{wLmb<!5<fXY)x5>sB2YI}6FBV`Z!>a2yVE(Kwx{xQ5'
    '!>u&G3+oSjKXuW!$ZAw#jd(Agz`^eo#zU1|iXN`8RTOHbDzwK1O4f?PW&y8WjVNFAGoiL3;ECE7dj{8GRiYT9c{g%U4pzjnSyK@q'
    'Bcpw{tJ|JXwEf#bo7I>dFi?zc7eOqAAQWY#&2s6OBa^zZg@(gyXzT@Bb7M0PuO&!XM(}JHcX5^l+RQt?elQ*y224N%+bBd26et=B'
    '8RWbb6YjJ%sVVpmFDlgb2P*|m^JjQ1fgd(7pc}R2QQ1c*7!TE_-tEDXcCi!^3uat3Na^xAjr;l>s<HC7&A%-^&NagexZCq}+<{2?'
    'mPOP`<M31E1{LV1SZf^GhTP5qsrc4&FBxYa%;OMZ1i}&x+E5}w+OZd_Hme@rg32lrG{I^1X<^zxJ*h3m40mtL*+5FEQMerly<5sT'
    'I`6f-E9CL|I-gY?uBFE4J#B9&>~+lX=(ce>&tuER*QXzr4XMR(`Ve*ZMuknlE7Fu1eA;ZVRA@HT7y=qg6M@C>GZU&s6f3Wztoymw'
    'WN-B|W836)plQr!UVdM~>C5lsHs6s#;DBPxq;vu6fH^jTAbe*&I%i^gv|802BdyxO!sw`W=$J1R_UVoyX0-{=Oso)8o3g1w#&dql'
    'y$F+6*x)<iglg+q*p%m9<RB2Y>=^yk1~S2kTI^PAu7d*tVSh!<u@ktzCvG|Ra<$qo<N)U#)iV`!fDTzwg-EGWjaHo^AlXr2?ddoR'
    'Pn&yX|8O0go8msa9`_@!<LAHZWU={P-qd~0^j6L}cA>~V2^ei>>l6OW(`NNE*&e)@ay8snxO1%hmJ`NdtWYX-=*6o^v~a*Gre4Qc'
    'tadyhNUu=bnFPAl6iL_ht68)<c{8i4yKSBH=41qIUw<Aq^ln>%op%h2wCf3L`1<T_j3Se9Rt(v0W66#;x-`{+16xidU1`Quih?)B'
    '?Mi`g9mfUtSn_dpF?(3u*R*sv&*MI)?zEP4^RH*Txs8iv>Lq6u%@yrDB7)I(yOazss7<JWtRQlk=2Ob?1J6&D<16Lr@{AwY^<w+_'
    '>{Dh`gYqZX!8<lxSZ*8fyjW&w+s`^+T5Wg{txB!-dTYa(#+tAk!eBjQ)$LRo!|kYkWBoDnT5;NLo<5J6sj@=bTJ1LKdvd1T?Pi&H'
    'vVhDr`v7=kCTax(rSZTzNf4%bo#eX{V1O4?=LhR5V}^HxAOLZCt=sU-u|Kkgcl##L>UPwO&N*JIo%HiL#pn04AHflK?_Z3iR_zaQ'
    'K7GQDY;xOxG=}ibC#W@R%YTAuGkKkk_muk1{S^MZ+;dR_SGL#0d+hZlzT=q*k)k6md&X+=w5!K$KQ>QC`Ys%nf~wqhQKv{f9C)Gf'
    '94y}%V-DL*+bxU%L@&ePBgljEx8BFYW83zU$Gq(;`h-f#$e^3;H|w@z8#s<a#eHDKq33?vdyFgpmerdep4#ZcWAEXDbz%dyA^T!n'
    'p?Hg0ylhT_f)pp40T3prlIrsglQ7545&+XipZ2a<Z+Z8!a?TQAiISnjyUjQ;W5DdsyJy=z#k9OVF%b6j?kEYY0jHe-P1(s1kPH-k'
    'iiHl@L5h_FT?K8=&Lps`awe0b22R0Yd;wU}u&ay1){w}RIuG;bCnfI+SaY0EG@-Y`sy@QL2|R=BBVfg)zG)I>&PjseE9FQaf2Ey-'
    'xG0O*KjL_j{8}u<B)zn!+ZOpqGTbCgguFNTJwL&<oBi*T+D|e2;XgXlVPBGvx9jZLFNUKl$uemF(eWTEubtY+;}*lrs%8`dO~77!'
    '4z%t^l=jn#F;J*(6AHqcVbV{C65tk`CHct_xpOQDc|8}ka~luG%fi2x3z!k+b9Br)&E^ERBr`)6lgQjOhG8I{{0Y+N)ACCPu~FM`'
    't>@K!&e6cm-@1(;jzzC=?_7qD?>{dJ<*k_)kljxt@scJ#<?%kPdBxI0NWk*zrYRywL%IB9tn6kLwWo>KWd<FZ_`;^aHy0E}#IllG'
    'N6@9G9NMRe+^SjmT9TxGnhMTBeGm-cC}T-O!1`$}J~z>?@Xt^W!I_ztkJ70eC1+(_b!Lb!=FHQ4novji1SI5;2!ZHkdQtalrfr&9'
    'aGz}+`N2;s@LhI9`CRx`h_x`e+2JfFh5)B!$4Nk0;ESzy>~v)j2EQ^^X<7qT-lRaV$YODf+_mHI_L`KH;e~a^2w{-6ybs)uLe&kc'
    'lZ*0<(I{#wdQIFkNsGD6nh8vZpHaDWTFfb@hGr^e7Ym20@WCx*TP>{(<Q}zFAor+gEBV751pmegKuMI336fv|&2~AZf+8!s6mqpC'
    ')xt@wy87O<v_wf6nguV6@FLOdYT%TPT^WyFu&&wzY$uitdk@|vk&|Mltvyk3_NVG*eIn>&B?!AZ7)0Cy8S;Y5=sAyf3b$J}SulMr'
    '`;~!EP>D=DCcdBtA1f3#miGytK*T>sr)kt_nlR-#q4eWU*&DgY8Q4LyIs|^0-ffrTosp6=z3@6-7q-Xc+yY4+c8iav_w!p|dj&JB'
    '5&w#sLF;%$I>4Ic@o;3s0qFc~Ud!J3aD*VO%YD9C3`Zb3IK7CExAXJ&Bwqvj^R)5wyYYEsVc1PTqAQ}p^AagS`!bMddtK+eBlmQW'
    '$Bs#R;x~lGnsIWZ2J&40G=r1XfkFBo6j?P{3<08QI*<%_n#Et{Ty>6zz|`wH=ekg2KzNx!M!SE@nX>L+C58eneI9QPN|v{FUgx~('
    'dJ;qgpFQZiuf#XwLJnG)NYs!y|Ljaqp;#Q>xX1{?#LtMwkR-y<J{Z74j4*sdnN)^<{qW>o!iavH%O(4>%q!=C$y9m}E8~izCicY4'
    'Xez&3+Q;*Y^LrC=JOPDzLI~j&lj!9B9D~pd`FMVL-q9g$9<O-uwZPrZL%jAq8dxgCg2Re1KX@bm_&k!JYkngg=2r31Pg>Sl+5S=e'
    'f+CXTt;G3BLm;w`bD49^BJcqHD%1oGsX^pa_Ail*rP1<F{3<U&2&v$U700pr2^|><P|RbqPjj8jv=<ce#GMd|9n2GaaabkwVE8cu'
    'n7`*e&x+9Y@n=u`ja-pe1Uv#KGGltxAoZlO*|SpM|GuvCAbNiL+wcF^w_m^g@Dc5+h|v$H>Fa(V843ydV<9LL1Q?dT(lQPo7R|co'
    'q?P5vczG(?Dg#>tsCy9Br-+z(Bl$-vIKqWhDkk8~$>wz6T_>>m6$+Mc#TEVoy`@My5M~@TW0>&|1vT4t74<m86hm>QfDtN`@elG8'
    'RJ;T`UC88j3<3`}aH)dovq_3lxx|<bXN%DS&kK^^z$U{v2KQ#4IR-ALkP89B8c}C2>LK@@Ixgh|1vYXxmZSFeDJPmIs2h$|bc&H2'
    'VjfhR?Ti2iDLx9C!f}5$0eE-Ni;i#$=19&mM3fLgJZDf6#G}hW2sYXnfFcOU=mqTc-c3PGoUET5t?WH&E(ek^Wq2IJQXDo{B~Eqp'
    'LN*c3Gu8WU1|^Av>R-*G+&lfP)H1J$zk5;sL{l%sURqpGMe<Fqc>pac%5TEU{8_TTD~IxH2K@f(&p&?t`yapm+du#D({KOz=U+wC'
    'vA!-SrCR$E|B{3`3<DI^7Rx-fp$!PN?8v6u71?2I^LdOFSMvx6wRn=6>q>#GP#YiJ_|2l+R)f?BUXltCOVeNbaB)Y2A3-Hr4XCwf'
    '(A?T+Z*X71t6*1<1J<B{&}K>ssK{AskSZ+DryKl}(5Lzm3z(Sj%KoeFt9U}wJfN<=DdkiYX&|Ltq%+Vcl9-?#D=pnk0A$&%;8n}e'
    ')2oQBSiFKt-lAjrtHyN`T(_uRuVpQ<I9Gz&->PG(Al5xP^~ypVcdo)OI1iQ<6-^04(G*@SnxzdQBA!&q<D!)lJOe|j35Ho<bn!8v'
    'FBzg*7|xegYl9y2*MAM3Uw;120!-g7$ydL>G~kKxH3sOmnuthSz0=N0H)aA~$4CiWeR$a@0RK$FH5QZMwCJ$gf&F0#Qm1CE@oH;*'
    'SUiET4ykN%;8|^zvX3$>1AxEZeI3?omUUCFmX-nnq-lvDI0egkl)gOnUvlz+%SakOBOa|AgGXqg?NS{>NUQ96+AK>WQ`F!MPn&qc'
    'NPQvtZJqeQA7TrCh&jB+BHm*ZZ?TK_n8qJs9q%!ax7f&A$mA`M@*ZA!3%b08V%{S(e~8??MRflEkse2Y-XlZr5u>+A(tCvI50R($'
    'h}2u8>OF$>hsf4@#Opm0_8uX7kDR?l)ZQa)e~7@nN9NulcJGnAKScQ6B7g4@!9PR_-y?|ck;V6j;~yfCZxPB@<nleD`Nv4-dj#}7'
    'GWr%VeUGI6A;S6|dHq8~_AOHT9>M)XWcMxN`xXiQV}$rEa?F-~U5+|~M8MU3k2Ehb+r+*)T)u_)SvRr+()EELI=jlJ)kt7zv0PGK'
    'e}v7m{%RL1J39jq7R6`>=>UZKGjbe~-Hnu9#_+Q0^(fD8o1^kfSICha#UI3_f>_gF@d_)Wp!A&Q4<jiIGk3ND7&lTKVQCw*I)bOY'
    'Pv>3;Yft|cP;+JseBABkD2=lgRtItRz;PV<!OIGnBF{}{{tYxYI)J?EAm0_aY<0d7Ye}{pIcm-%)U*7-&y&kvq<)T%GtQUMD9Egs'
    '(~?lyAPxY-gU&bF#n;Cqr+<vfNx2wMHp?P0jE&{kml2w~J=?(a_QTl5Qh#k}&sO-?-~Rr$Z@>QVv9B*FOv){oU%d_yo7i~TMOLfq'
    '=`9E<xPeU_4cr2*C2HcBkmW9mB9EyQRN=9J(I;R$P+_gBg;cEUY9STS%2CU|$^@2uurG54+{Z4M0`v>&)8S>IXkAaaDH2y!Z-Exq'
    '3a}J<yjyX%I@eow3s-aLyuwxVS8-A3zKi;M(IYHs_oX4RU@ednw5Zb;zHTj;7Ml3w@QXT(bHjBiYvYkz`s>CwDf!I8n`Da;E+MqF'
    '!R<no&F0N#{#7m|x^_?BKi})0|M>Oi`*X!t?J$6p>#GJ5W@+jUB*|r{eLyfXQ56?)&4)>7E>V_$m=SYPZXu_qK&6Y}rv6qrol1~Z'
    'Blkf-ekR`|g_t6@$->Q3mG+dWy$aW!QnshK?J0MA3gDiSxKkt-ltJPs=}+cS5PDyZ1Xbqhxnd&BAhLQNrgTwAWlUHFbg{%0aik(!'
    '06sTV;!B;5Q06mBGht;^xT2beDL&w5gTfS6gS!WPL<o0HfO|`*RL<+)YH~wYt$$QVXQ-_Gz`6DirnW)p%0L{Jul_M{ozYF;Jlr>='
    '3Fb$B%IDMn_{-1#CZlz&uPIca?BlQ7bCy-)a;Hy5BDS)c#I6Yw>w}W*Tnkq~8`=^Tbe;fz06^*ml>uwdS~Us0?xrrrZo8mqV%uC4'
    'b%Q!sa3LT{v|v!M@-xwO<lz=Y;pQiY*oxrjf@_NWGE=OWa~@VN&<1N$0_tYITe7EXtuLf@7iH>xDdl|&6!1$M@M-3`=ci_J3jU&A'
    'LbgH1K>=m3AUi`r={fi9GyL@P|M~a(L$ojH)ienPeMt~RelwpzZe!O363_%RPD5Xg@;`(w<>_2jzf`iHV3R=AT}f>fLsDrV&~8^;'
    '9_1pPEGwxC2QJoAf>5%R00N1|sVkN)1V%?N*OfmlkLdE$^GLEt#XfL-Blg)*82KPhH!K|*`6o#$+~2xuk?n{hNv#KYQt580Tngo>'
    'g&^L24S9fal*Ls|;(%&9)c}c*rJ*6Y__;(DRW&Z*!lj$4v~F=P)zTF!n+jk~O@f8|HU*$+s7_R`qoPGsEb;S#j%kN)l^|-Jry((P'
    '?aCG1K~=9yYDxhsIoVy10KptdG{`=3eTjNaJr?|Gt!!j*4kaR52f$gWK#gdj)(k$1BK2L&jXanTL}zTghyefjk6(WK{`cR1r-FBV'
    'H34X)DEf7qqnZ<r=2e+{NEO%o{`Z>*y4<y~0xIg-fGvqETwT?xb*;3(i5Q0#_breXRE~+3KtiRgi(1{f(nGP&JH=+mX+2(yODKY7'
    '^(YSIQNfah%$xkH29EY0LVlgXUCqUA^I=|j4?(EA$`{yJdD55&z7k!V`YSX)5*E|iPMoL<0~i&-SY1!#7{p*h1>e4}TQV*x?>nB>'
    '<ga@AMN2(-HOHms-KZs{j1f>p;5Y;*qDm(;$$L;Xr8m-m@oy{fJWWRVY@}tnSbvE8bqaSIqOXdpHcLn+HBqM!ShE9^5rk7N(o*0L'
    'fyofxB^b8QyaM?YOzXFnf~>;wgOLhNO>{7`C4)nd?cO#e+DjzI3Ug}hP&ftZscsS~$;t@>8)8y%v6*TIdbjB&f@JliN((@^K~dGU'
    '1ckhFkVM2?y9un2wbNFpeWNR!5pu9l8l+&ji#0BkS5kqhBgB0W71oOBV_HPD(0FvJJSJpsb(N=D4qK?Dsx>QHJpJ4`1N8Kk4_93S'
    '08f-Zfp_)}1;!rkjj9+#*Qcmbrv#5ym3Zt8@@9!2Yw50XuDbBZRq#!)UR9F`2~dEnT?l%&G&TSpcA>kfU{PHdgBlk3-fC4#e9Nqp'
    '2)Ih%kQSF!q69x8d-+$*Wd9;`J<T4W$!oC)xYmfld3qKlU)+*hvtj~(WfA4Bnl4BpSDA!8csGqxd1Y1RBE@OXI`k}H-@gCLx4+0P'
    '>HF*c5+ZXi_3K_S6bZ9YdjF|y6PBxJGpRGxgpPt{hmQe=7d8j{<MXrE&3^sqm-}wN|MiF4!v4B_B;|9XeF+UF9%=K<V{HXo(6mCf'
    'Jj_`4*-PtK$T(X{hr_wGBv=rNp4jFWv(H*EferisF??NDp4YgZQvRy$#@-3p0ZYaP*!1PFHg_u-%Ekf;g(esA@6LURZsTFV7Z>9k'
    'tK1h}Ii06^dSUyP>@j%dmRyC9Y2<ro1}!<RG59Sl-UFXWxx6L&TI`ZbF}ILEu;3@tBCy~m8}z}RPfO=jc~2L=kdITg86cJ-k0F-F'
    'r5*pYlHYA@7K_j^-sm3tNYg~;daEXvoAQRCAg_Hyg@Ctg`cUF_%drn3b2t18GLE)<1bT9_WQ$R17ca5$O{s$w)esC7Dk|T}86n?P'
    'S9VKRfjPXuFDS-3I3pl2;yhT`+~RtA0=N3%A!KtA8fUX@iJ~W4_Js8dkvOq|X@+mrdPg<0;!eIoh5oWd)vZ_sVH1Hyi9QH^`(Lpw'
    'iUshEl@TI0^#NB_o^uN@MP_F!6cUBry+{)2ne}C$@G3A$$fY4K2a`L!rp0HoVawav#mw+>ZD6CoUtBdg8?SMg8ljI<>cif<R<G?W'
    'xyZRLb2j1_smFESi|t$#JpQft>=ZzBpNgD_+L?Wk4%4$Rav=YKNe=!<t|Gt~>W835H)%^Pz=$h_UV00%5`{>@MItRSrl4dh8cQ5o'
    'X4I230Vz?ffs0CYEx>ET`k_@oRxlB!wfQo0W;8%CxTPRWDBZs{40}W-cq&?Y0@*ds%4NbUX!T&qEauGAmw*kVGivkd&w2T)&X?42'
    '^<K8%GoeK?QNPdg{42~}t(pEJ3fyehT35cN?jZU(z9zqy2oDS@{$46w+=3f~+;c~7JWdb@(7ioQG}F)B@`=~X?}NXu>9e(ig(l7O'
    '*F1N3^A>u<YKUfwak{flpY!tVMoT6|Mkx9Oa1ugJrVw=4UC;Ag8_r+9|M<`U^6L+B@&sbn_!(dKU{sLKrnzwrGEBO#jv<4nGjLVJ'
    '<krB&U}?nlBA}(+pWoWU%B#HH`rE&L`~3$&4ZrTM28rkRYPSO>z7%h)$iYTK8gRn;Vr5tePK$L6*|8$k4Wn7#&t@%}eQa}}8R7Z_'
    'B|qi%2IYa19By8LjnI;`dE+A^sKHFJzyu68ETX7oZTOLh*IA2}oH{s1h2n|HS<N<vui#i*OXb^h6oYj53OWJd$4P_{?axUt4?#rL'
    'ztYvOCp(h9N<)Lon<n1fKraVPK$hyeb5wbC@Rktx<j?f@K-IuzvdrqbNp`6w%dn67Zl{;P0$AsAVk?Z)xy)XDEC@^bVpd4~ve}xD'
    'ydzCZUc!B{Q<*d<&O`{)AcQGLEE=MZj9Y+E6XH@5RC$!gC;3jj_%uueVC#9_Bh%T4rTRIhL#^k-c>rN9@QKX$YQ;{7J$w`%DKy>J'
    'fbP~xmY3tZuUniBt2jh-7@&birV~Vhj`rudgXgw<rGNIw&<^iUYUY;qiRiAYAf_z3a}?&MWpJ;5cCON@*<C+Tu?wSo;RB{V1KnUK'
    'VRdddpX}>r=PG7x<)m0%-h~*3b)(W;UO3h4=ea9H%5;5Y&oN>c`l;@7r+2{DEgmh|ZY%O`YV=TwIYE-38aD<0$<Z?yQ>JHw;MPsX'
    '3CqdAFk_exY|?58P#JirR?ByMni^@$^YaY(NF6hwamV9+t66MtzfO1*fc2isQod=IN#GY_jb=&9OwDRDLK4C8To6y4H-}IT|6(?~'
    '0*HF&EzT2Stowq@D9-Pu`bB=Y6?+1Ab3U`rTrCXs4nYc^NrgD@alNkltVVE(0pr$!K<a7+yCVxi*ytE%xeBAofdR32oGau7wX?aL'
    'dBPT1@+7uOBjIIh7(k!ro*n1k{o9YXFZa*?@|XWZ8KE|Buj-Op-L8ArVYje5CyK2PD!IXbP`eHKqHc7NN}OXwe?@19nB0J<NNPw5'
    'bE<5t%4>t^qJpmJT(64SNF1ABI0RFxl3r+XI;cKx3VTww4icn-$2m{M%QT@GDu@_?t1GY|`9Bq=GWlXvwY5#V8<qD>X*A%dVi5ZP'
    'fZ7z{t_p!cBT@ysXbOP)$rkrkR8MPhRe-|RBiuQ87xmKQX@zQ0swLlL@mBX=N9Vr!tHKEA7ts<3qX~OfK@Vx!tOA5IO+w#oaAhHn'
    '+){52o<*1;Sr}CgS?l~(oOHsVwTqsU86joHrqVZkVVx&-EJdRQe`pto9eoZNc1?8LO?8cwa5qRlI|P8j40U+2pofBL58O|{qQ#d@'
    'BEF_M(&V``(*yEu>R4^0O-`w=2V6)1&8D<Llatn@HS8;k?YC*5H^s7ALER~m;;#W+Qm7)Yx9C3wh!upE3k;=UQ`)d87D!<^^o4Cg'
    'a~OQVolhl;q!GwnP!Ni?NKpNzs3l<hCeyV6Nh*Z3FDMNa@($iZoNW4;f=Erpm`!G9lP^K5E1lP5R1>CdmjnO$9z@X9SBlFw3t_3('
    'EQO^W+WzJWxhu>I;%PH>)mNK&$P+L(^?9UVQk9-+nnuU~HcbN<CZ5@IPo>tm-#|@EV+Yg~cLts1mc~<_=lLgN!*AUP)w?y$Sn*$9'
    't@<oNfB)O}pa1+<!TH+PZA5jF0bgCnxMTK8asl=oMSt3aJYOFU<qn<R7i8XTn%`W)UTFGgtRj7N@lg>YQm;>|+3{BrfK9*cR_f!r'
    'V3OKI=}NFYt|A)`W*{U~xGiEYZ6cDQ;Uc66K%Iv4h@AK)d~8t=Igc^?kWbRIOzkec2B9>V*Wi5B65;C!LMSf)LXl=hZe0Zqbh*`S'
    'k-K#^X<5Do+%X^>mc&x|#Z37Xs{awZfu5;o6U!FAkkpTCN&f=s&JA1YV&DwR()ql>6HNHG8B9v@Eiynr{?0v=cxM`cnrrZh&%#oi'
    'Vk%7l;xhSeV!;MDDDS2y9!=gGAt3juJc5P{+F${O0bB!@uRr*z?J_?&<0pR`asFC3Pge%^$a!e*X@eOXMsh7Lt63za4NIf2G@;0+'
    '<ptpxT~7;FX)HdYSiOw}-6LX*<k@cMs=)z>zLaVJno`0g7^iT{YK3%7k=Bl3wYiq(75@Sn+nP_M3W0_g&Ba1U=3ZYF(HiCY+JJk%'
    '!5U~Ovb^5K^%lBHzOV|anle-1Vf9EXHv-+A>G+z$zyIy~w_lXdx`*J`Wgam>)$f5DCresdHDYc<e~W5qSd^=n+zaG@5pB(aK%6sB'
    ';IbNNa~l*JI$%`RSvm~b(oZX7wiX6GqIPIhnpM95b0!&Om?aH;NbU=TWghP+?R6v9nq@YrFM{vb8qLFsUuaW<u4vuBPGpR%O|8Lv'
    '&n}Q+%>On3g_2B9f-qAO;>v>WEo^9?pj*{8D$UZAE74~P5r7mE=(br=b_X(;=FO=b2Na4xl{wAIhD_NC<(s(;Yfn+8R&$?okXHgl'
    'g+9&Q2Gym#r?f;iAy%4*Cq&TT38FfnBsYp;AtiZ9P||D5ygYonZ3LE}1`Q7MDEyY<Ld#3E?V58Wpv-U{i~+E6MB)m$VWY`Jj|yrG'
    '`L*r3^a{K5<@$5Dr_IY#Afz_uj%81&(+KN)n%pegTwnc0t;=)1Y0)~zr+IT|gSJ!LvzP-Vl=HdUR3kue+6~zUX6`Z)!Z*eI8gL;C'
    '?K>b%qwU|KwrXS_aOxh$@FVXU`|uGIb8v)A?t36#rbE*?;633!&~Yl#kc{$i)`CvaS4@P{T(U5MedEBv27I^aA4I?IHXJd6xRc2e'
    's~P7rP8Q0ppjJcHNg<0R?wIG<>Sf_Jn7q(H>=dxo;&*`|3wrJz-FzQN(!zxiJOemcxJd3cJRLB3cd&;wSgR?pdfI05cb{`j2+9lm'
    '9RTRbT7Oo+`DDMv`5!30ZrXg)3fIZBo-OAKVbUphtP~G)V9zzbJDesBl2J;nD{EATxu6Oql4Ge_n2XxS3Y@-QrcPtbn2!d1-sGt<'
    'ApbIhn+0;iRBN0=@!nu<p}4!UD`1X_WNy&bvrVKHDfX<>=bMO*wF0J(3UtPNOf`ktT%ZSWJ1W){xj3(HRja|P&DsTb48R{ucD4Df'
    'iv+#yao_Fr(@dPDKfgtYuMupl%R;`_O!hRby)6p{wLb4GR+Ktpd7iENnR`2HdzHheU7O#swt!8B@h$g-(`NjjsDhbm_P3IhhS142'
    '^YOXe{cK@gd~^D3R4D2H?R|>pDf||mnq?8=;om}A>*7rII2c1JHdd>CX8O>)IoM+@{9e3^$Gk~zK@eBeYoK^nTm4gfU(Tl_u3}l@'
    '3+Z_{2f9M6&}Ib;>ZaROU+)5|QDwg~zgwL^1i?yIESw4kwhpY`t^J|`gR7e{faIMy>?-S+x$0`izNKIRa>0Ust>f`wNpl_Nw-o|H'
    'iv=(@(QClFR$I@*o>e=R@-hjb+Z3gCo`3oJTyv#_0;I$6D8a#9soAUOZ}VmEHp&C+r`4+6({o&N!F+<XTWXST#9XfI-H<V=8e=_e'
    ')Z?oRa>*^4nh{Aw-<N*yb>KHY$MFxQxtSJimyy~>fUOS^x@xvQlxHi*x{BOJ1rZPLlG?19i)U!8Z-w){*2(s*1nFzFAVYiBI#WAV'
    'ILxcLkg&UIPwGfr-}AKbobcD2b68^=-=cf{<as=g;7YINvqTBK!b4ZF@%ZAWf2i;{zvUc6{;KDopTD_J8glN;i;AWVi1>V8*bv~Z'
    'Dtu+_7mGH$R1>8Qb64@LvyFBxb=?(+Yb8=y(Uz#xp4YJ8D4bTPQ1DmncVH-Z9<(*gT2%|zEQ}dcyv!;EhnKud_`L8^1vkjwf<x9d'
    '@k;4iM1g?;-EAvEd9UPnH*<sbsW3xmK&Fjt?0q(xdAPvmHjjr3Pq>)p3*{2VKS156IVP<#WcpKd@68vs0sKZed#EG89#=sbd#(7x'
    '0mq!rF_#<)&H=qa1?L;Ex#e-|^zhxyCRq`VEM^_0#IL||%-1Z`_GwOW9J|6?|CFLvGxmechbmRR8gm(*e+rkXY4PDs%5(bmIJcUZ'
    '$1OSq^lfWLvmkAQl4H9f?CN#EngO0W))eskMHj4F+y#Nx4sTkvc?Qc*cX;#|&jo&{iUf4&(8&sHaGnicjOssgz^Q#DH{7|Gdbn82'
    '#kO2N+hyegIG(mq;qdVyv0Kc0#ok>^Jj`C4vNG1FGE2ksaqQTMYD1+u1RDl<7ge_cKc|FQA;MyK5m~{4^ow|aIc_Efp3|hj1Ms+c'
    '#L$7(;*M=xIiH$ZzGF3?qdb0-6;Rn*Tq8xn?|C-tE|=8BxctnKCB16TEW>^7g*q41=_s%0eRtOVS|7p;6KebVT$9>s*3V))yzyON'
    '0Sg3LPuusUww$?OW>}CKdWI``LeDefcNcu>DBJz-Dj$Crx9jtqtX;(%ZQ<uHr)b++7N2B3ic1mET`6eYe}d4E+TBQ=uY#7WDw!w_'
    ';G7Eub@PE<DPCN%I!6-p>pAYqo(r81b7fx4mtC%`xKa>eXLU}&3znbDnt)#q0fd>w%4WTy;J?5q6E6EljSa0+RY0@;R9EVtvQ9-#'
    'D&{@a__<zzIxl#jpCIRp+17Zw;`~q<KA*}nVmY7~J`uJnW|$?2Rm^8ni0zn(l{#Oot6a0QuA0Z?VjUiqXyUjl<Vu|NnF`$k-1WI8'
    'RdIv$lkr2ytN^V2B&1e=;JS!4*LIYIi}^|lg2(5Ys1O1*pa52KN)*r1hi&@3h?zkzB5Pmek^Qah^Es!vU{>!1A$b|}zvDdkyic3s'
    'R7Z9CmT`z$a$%+@&cicw^Yi-|919w+cJSNnxDb0H)t23gi|{6Qz6&<L+-c4)ul6z(&v^az@^3kg1|REop(q;soQ~6GEMXUgD^N9q'
    'beuMOJwyc-xMxLD?2CI%7yoXK6D#Y;vk(hve&&T@f!_tn6ciehJAy)RV?7*0J8dd;Mlr9nbtH+=5z`#5G-ZAbKQr%0`7<j<G35d)'
    'OaZa@95)TkersE%%TzG(aj4ERZ*?&jMNuNRpfhxylQn_j=wkL;s!`Jxa;z_?&MfG%#d185G3M$_{O+(20g8oY$Sst_NAhK3|0KT>'
    'Hi-+m{jK5BRR~ONKeK%2fTIYdA+`AyTH6DgB=5B?p)fe^?}%<0oLi^OI6;BvTp!&sQ@;3HZ1ahCq*6*t%AI0H(PBk7>_!l|YR}{}'
    'S7suz7+eZ!R~Cbd7l`A4d$Wztb(!njZPpgzX=d6j#OTg({J!F<WzF%^#x{nrz1TJ;_ENB2&G389X>M~4YfiRJ`-4af_F*0m$C<I`'
    'qdoCwv)Sgo!d2S%Wq?JBaDl~)wAw0YAC7qamiRV)2bH+U#@=oE)Al*+#;%g9#F|P$DHHy&QhZ;=Aws&<`-0`{01LohuV?c)TptD('
    'w0$dr0&v;iZC}&AMpmfPHupWg33c9u{TJ){6Zt`z4D<tjM^>m)C5W9?P`|^Ib#=Sg0_xMB-QSDn>?x6AAF!VIzx}~dP~-Eu6pXiQ'
    '^G&@j5#Ot(vGF<B26?Z+T$Xa1Nbkop-W*HEm#HJa^`t$YZ#v9tl!}LL<DZV#cONsJ@#8he&x6VW0;%V>kaN^w-hbYAIA?L$7iNC>'
    'I*hmK>oA^c@jKRM4rY6}4>@PiC)?|)`;+<Ze)GCS{J~5HjulvSgncUS)tURlRrrhtU|q#AmTdd^r+NSJ@h=}I4&eRza(sX`?=_wK'
    'AfM^=Q@xjQ65B9!*=CrI;W$^~7!H-nj`{H0NsJ{)Hi)xPOO4(@#+OdpRnF?GcG;dG-(hwduGzeX2$m$Xw4CQhNKp0wF)<6$khv>f'
    'ldWyppFPp>G55LEKKw0Ov24xaI;P$trdm9f$T1ZQyb%AibWr$wzI70<E7pV{_hH7WAXin)0AT^lHP5_jUN@MV&9Xs-cOlm8wWYzV'
    '2`NWuUPCSmBQYjb_9g=XG7Yl6#MMcKizy_M#G^nzV-k;o5Mr)TE&M#h9wQkqh(n%ypVjj+-=J(+#`;l-R2zS0I&Z-_-RGE(pZvsS'
    'er4wm$Y&4}w@w0My;o}mEFUL#!ZoQlIp^R9Hs?8#>1aH^Rk79EbNacftLR5}Y?ATom?i79kdRM%)Dwd&qgg+7AA2@6F9z%MbIrWR'
    '@E*OGr{;28jsZH?$vVTDiTzWDY4LKNa@_0Kw9dzTjyneE^{TPI_&V5ul~}s%1jp-2%e4`kZFxN$Ne*+?httky&UY~XVj0FWtYkrv'
    'O;Daa=IY*K+G9@?;*8|va-Z-#C*M=eG?~}LhAT!%;71i>C+P1|ea-f2pXWKxXW|*+qcG>6Hht;+)^<lp1>KoCGBSo*GQViCp3i<;'
    'd=Fxfu;>@!ZE2@?zrAu^@@{<h!n#f@Dpg>wx}JO42emEmJaZ~gW_-i(1n2B7kOBDV%%_rhKw>yy0bgQ1XKua4Nv<=-CsdK0?*)k)'
    'YCR`j%-pmJQ68rt9PJ(*Cl2m-w110sFt&JQZLP4c)Ils5#=DyPN0d}2QrN`D8TZ5!eg3d|5c`nh%d8=FI=)ZeXROVFz&>X<_MPwL'
    '_iW%|E0yl8I|t>Q5An9l`4Hblx7>ODN-_KEx93_2D=%XSs&hc(m5A4o549#r=4+nw^Y`Y$%iQ^Pn;f}0%h_X|bMiNNo^$WZ%rcGN'
    '%oEReos;Ws=Ahd?dC4FK=v*X`fnxm$s+68{oU^{_>-=l{*Lm&pF4UA~+{ZkuC+|+YDV&e&{dN4DfBFL57Hqn<*G|?Lwaq5uTxCgq'
    '__y=V-eQhL^1l@_JMcr@@w(}Jbn`hJKj)v{`)}2Igb7l)7PtrEGUvGS-@XTVT!>9$UhU)UI{99(;zMec`8U8hXMeuPi`qXRcGau`'
    'cv*4#{PtXnwGE-e7D-6iU%w%@7#5r>-s{b7#dXhb&$Vn5_&h&L!6qpF4SRVp$#Z`_`?Wvbew){Vo9SNO>&eX)=NJ>i0;U7DAatPK'
    '&Odvfxqz$f=lZO(wx#XMc&{nm%j|cMce9O(eNw4O$Tjr2=DCOWg<+xa_3SD32fSvF_VU@D^UwMS^A784otNWD+#kQhb~HJ<lIzhq'
    '|J)xtpgTuqdgW{rA0TgKza!h{{9}VyR=r?5uy4#_`?w?tT}j<7kYb>)U&r&e=UMRWs-NGn8Mhfb)j7IYFI(fBpT9>vB7InRnDV=%'
    '|7*WEiQwkrFI$QI`K_<zYLpYVDkiOS><k;F6RRo@uIrrd_|*ArUJJUrjL9|kE3rw6YBV)p#~s6Se`jnje|wakiS4V7QaO~J+7fER'
    'yhZGt^*l27!Fq5xhjM;<_6Zo6xt8~tc7dg^5)|VY5cemT=K@?=Z7>ki=P-vBa)adH!C16{e*%dq=<EWRS#5A&Fv}{6T|lVb$H7&Q'
    'hi<m-r`6T)NmlqA_7<zF?N{6O{X+H2Yx9C`oE%}E6XEkwQ}n)sVioZMYj8DVx!_(yqFy+&cwGlYGGn~(CtHBkK*yt1`iDF;N@#j7'
    'nCdULd4_F#n`!HLcNo;eI2HOfc*lvxB=M|l=lRXoG_HkbSngY{W1L*=uWQCgg*UbP>)GG{w;zcAoX_v?mFJ+uRm0zx^YK5e;#G~z'
    '?Of9sLB>)Am(zV6d}EYtOfG7r(N=B0_?p-lU|;ebtyJdcxB6Ka2ZLf5@s4I?voL<evY57YP&SLSp~l<!j0g4<^7QOXYQMgFvp6Pu'
    '9kG5q@Wf@di*HBXoeG4vxs!9ryy)Ja%!{6F>`r49>sqNWsB<{Z7(zX#ZhUJi9-g+93VEjD>vY<v2VTBc-uKvH{ylrS@gW-@A62Bf'
    'nTy0g+Et97<XX(#<?DJ}SQZrj+IXrBc9ij3cg|518L<_f`{r|?V+m4LZN4u!By8!Y%~*nPdzSHP$7}38*E#Q3Jk|^MtZ!|;J$*&&'
    '9H~^DPn-D4^V{=`t)`;knK#72az5pp_Hov^=luMAv!bJAyHQ89eMIiCUCnLI;k};o&$}=#FKl?2uJSyvJo$ULuBgT+))GGF6ra<d'
    'wTZmq>zd~tjPVQHDJ+BejQC0C_&FzSPOwLv{7r`}wUeLdU<-O97O_yNV!wNxe}LEj+mGMB|NOV#e*XUB?|=I7|Nh%Q|MC6jAOHUS'
    '_dk6*fb3s?|I=T7{L?T0&HwV}fBxg!uYdX<fBnmS><5WweBB!hTk-YPRGodOJMKSWUC}F-qp@W4ul~@m>VVZ*Id07c_15dRuEHMc'
    '1k?9$od52bJxqkEa{uRq^G5&te}B2(<=_AK?fYLofBf6$|Mcz8|NU>je*5JwKYYykC8q(&@~_(~fm4S^p5)t!;7m;MF6VZFBqyFk'
    '+2kkmD6ey$JN`ev{P?H;`M2-?@wZ?8(+?^*_}bl`TI{c@;-nv{y=gJE%Z$brP;0aeYPe|gQ*C&!^B<)A_kR_kpUA{u6BE8n)poa!'
    'EHb#PwtH&ql-8;8hMIyk#6O#0q^{>u>c0Wp*GsXswCZ}Ub^faVr4~e4@>Zz@Q<;F=lqL#ZIImsl1Wi}A%Bk3TRd&5qdp#9kud1+D'
    'rP!-_>{U_rsxo_3p1rEkUKMIj)!M6)?X9})Rq^(!f_qiQy{hG26?AV^bx)<;tETX&GkmK(eAOea>(njIe(DxqwT!R&Mnnku`a>P$'
    'QyY1{C3RI#h19DW>Qy22s+xLLNKseys>XU%UcIWZUKLtTHP#twKmGXY-{46X?X|+`gda4Gh*w2Y_549a5QLT$CSeu47vyo&)u2ey'
    'agb+dDmWsP%4{Hz;8yL+YKqfJYaEA1cNuA2)=^+{9<DE=w7Yq6ji{py;>tWi@cySt<T8p2=?jz=QdQDwI*{-8)oUWCbpA}_mab_;'
    ';7d)B9DHXts@?s-GjyxW@yy*E!qpKlpT9D?CDl8kRaUeE1%>ZxWxTtscO)|Vw1c))89}?kAv}zN<=ych9V_cbS5NZ{gO;1AGS?GH'
    'oE*Z{%8eEjN1ml4^I;w7VFgrlu?Crj%KkFK=wwu!5+!Wz=<45^J9-|?>Jg5Rah*ss_Urhz@*86VvhoqOkf0z(OVB>)u$m}OIH-TF'
    '=I{>h%;C0oS}Vv|^VwX-J*~vNZ@c`={w=)0jG|_$aVis<B|WMwPo5bRNJF=Ffd0(uK&sxT@fgh>lhxs_`J98t+AiqPd+E=;7*VaD'
    ';{;-_9d7WEUQhBWJKi#2$;JZ`H+Va$^X3V69B4FqE^rQ$k!|$MXtuT**SW^){OGUiq&2SJT4On~Pkh{}dcMx9uTU2=_jogxc(WzQ'
    '`!>sk!DEbmQVZVKTFvCnhpqZl<XiUU%qeYa(fwoVlI!?+c6nXXePxUA^0#P56MJ-fTm9j3Z{{So$AP-0>E0W-v|&#NP0nG@h1kFV'
    'ZQL3%W7s^+l={|fZvMW*s%r3BW8s;CpGXaI2zT2GPJBmug;-64D0Bl8Gl+yYD>O2f8+qtKB#Vl>yKRkYc%)J0wB0_)h~Ks{<Iv~('
    '-SHyG#SVLdFrN>rP=QGuWR&wbTJEm<8YtE_^GH*@W?y>@D~NF~__`ap+cB}A!4nTqca41Qm}49ju>oqlnd91C{5BW2rGJV!xA*J#'
    'c=V;m@^yAQ@&mAeCZD$Jy9Vojhev%Z%PXb4!9Idw=FRTcdSntD%eGWQzgWAdB5>^6QVrIW9-w(r9EfU!Uivz`+?Ed->@{%YC4YDa'
    '^EXfthfz}ZLgU)p-EF_xtngq5XYn;57qND+5S9`_^u1t(XlOV?{J~o1XSO6_H)D;pTY1OHjPB&MChK!YN?|f9VIC6G#5^)lv!>*l'
    'X(z0&2FVE5yi7;YuhhFz8_6pR8a97BsB)|s4&RIN1R4+JXd1R{iX7I-ZE79H6}#1h?4;rQOtR6g6+CT|S#;~M$0K-1UE&NYuLbXJ'
    '5j<#Fcwfi3W)QWQ?pvZ*%$*~g2ww`TvFFzeYs<qagl>65_G9v;s1jDYp;xc_qITB?{RvmUV|tUX={lv!$Z-s3f-t}c)%kh`)ZkB>'
    'KIH3T<lSxF1v_M7akXwaO`hK*FWIIaEais+TE0tj-?B##dzt98H?aGYoe`HqZPr8jq*xWt_ab7+jyXXuIYyH2MT}$?N9U&Bi?tfs'
    'Aiozl)a=jNUVTe0KKrf0gp%jG|EHsWd@pNLddwF`;r#8`Ur_91mNjC~c|P@l=N#y>;yTePy~}vD=Q#UvJ_A>nh_@NrF+pH7v#OMg'
    'rtW7vTvp+9#AsNN(@_M!%ZVbKF+<Ke`xX_9Sy%%vk1?{D+t<f#1S2?k65|UdD+B(B#s=rO4^uXe^Dj&D4P-(4k?EUPR_j@;v}Z+J'
    'N{3cR!Y8XrT^_gC#xutZHY!Y2=sMb7op=Yv2~Af&PLC7LA0Fq}oF?@uS)GVFDAh;c59XCNGn%BfnYFS_)Gq3<f@yQR9%0j}3T?^O'
    'VxLb7o<>%PsXeoTE9Gpf>5Y6%EQ;w$ww?;9JFB)*7qzAuDt*-~+v5r%3zt%{N#1c#rCcla29|BDzW(LoJRA0~xr|VbsH}>44z||3'
    '>?<vv6I&WPYF_r07~7YtjM^jFYd>qUdRJC}e3$U`t!B@@bCHop%fq6*729=^v8~0btyGVg2hIJ%mz^kI7&kG|4l}+ppUqM|t~j*~'
    '4+6FF7Q+>QyxlqHa$i*x?A<@KtNXcTphZ$l1Eu<M_EuwjEz1<kGPP8aP=jYVq986h78M24b_kUK)D;Y(fMj*mrsBey9raO_&-1BM'
    '9jWPAxhB;JmTq!|JTdlR^*>8DS-Z|UIQXO?1n|)r!Ll8o<;?yz@PA>Qz3nI?mv}a;L=##4Q8VRTOXW>t<z0K_51K6RS}pGyE^pc{'
    '@0u^~S}=dmh<VqJdDE16*P40PpovtE-^Z^E>mNl5`V};8KptT8v-ep(Z?GS^wqYbX0pH>BIoRZg(o$(J|JRlFvc9l7jPLJQ6z=nG'
    'c7wKGx;)LX4x?(UNRONz>xQ+20?cKq+_OjbX)!>n)^lI5s@)ni!*LI)Q{Z%PY$Q-g=x2=a42<F7n8|Q8KIbs5i1Ry65)&(t)itop'
    'J0lh(M8+8c+oliy*5??d(1}GD;eg;JR*Gu^JC;d+v4K-tzvZ~+4)0ENK`=5q#BZKoy(c<?bBf<Z*I~4qMzEuY71&F?xk`%n!WVW8'
    'LvPkuB&O7?n27S{(NrYZ3ll`=+trs2Z_f%b;w@MsXvV1h%;oNiuUUNk^7N;xn5Kn58QXq$KePLdiViDwcCu7f{_NcqYgpIJdg##A'
    'Com>34vCX{z3Q-sggv7UD+D>Dr>*D2zY9ibC*l<LIPmpiId)VXkS6ZLShc4!9E5gEk2y{~cIOLLR@YYy(5!M}`Kwqbo6+et1#hs2'
    'wULl?#&Z~CLtq%!y5=^%kiQmdJ^r3?nS(fBulAi7C7#cdFK53U)Imoda?nS;S!<16@cM|??a?!O^)u6El=FoK;}{lv`Re{!tf9pt'
    '((mH0YdxoY_w0-2uRR?*;XL+vDyKSj^neetxAnv=y5k2Bcji5?vOR)c*UWnqXa11l!=9`N_*v&G+w9<ZPB>$EIgXdVeImcV{_W?V'
    '{{Bt*6Opz5{V(4X0xn<oFdT?1N=rzY+!T=+vMS5GEdGWfBzj}QemWdM*{63)Pyo{pI$lTf3g>mW3Xus}(-v77fIf+Y@Q%gd3ZF~l'
    '<^1gRB0v51%l+{86W>OIktXpa;vCWCJ11jp?m;;Y@ePAK@L++6;5UC*p0HE?Y}Gd?mIm|Y90OFsh@#Aos4#^}<iscCZ1|JYYDAwd'
    'jl#C4C=XB1a85rJ?_su3@YKQ3rX(YeTFGMXEMf?9*Id(ESuua3B^j^lG_~ACJT-J#Ryg2gHcQchW_u;+P~a6<p0yZizfpkgN0Svp'
    '4XWWnOc*86vVeaBvuFMm5OyiI*9^ozczcT!&Ttt$=%bd*|5?n~Eu@LO6VIS@I6flsn#IL2kYic6xx1#ipOOT8#nR{bJO#kEv2ex>'
    'xdiHJ`NgSm%^OqHAFdmtR%}0T{OSr9Dw3>Y9wtHnlML^JwJf`-)r8ig0t4V2#V6euJowm1f>QROakX8rFDc4uNb}v|N*WA{GK~wt'
    'qbrZ?%k%&esYWJ2!bFtVEqFX#poYF%j`0Cd3O~c8oTw{87-t>99FrYDx9t$9MaYgw4vbZ6L6KI;AbeO`krHLH6Jl1QI#lsAweYT)'
    '-g>_BXfkQ$v8q!Fi*2j$A|lkEXg3ihq#&030K|8&%*Z?`+wl&GgbjNqk_)q6a1Ye($eLRAo275;h?cm(&6oON>s8x!9y6=><);cp'
    '66DJG_3Gy!qM>kK5D(a9mgl=dx@tHk)6KG_`Vk98xu9*0BXXK2$PJFH)PjW8AopvV72o){EWs_1;W9dvXL~IguhsA`zkUBMj<ql8'
    'iikijzPcy{Yp&#qEVKZ^Ac<fwR2%_xDS=>Qd{aG`lLxw2sUn;|e{T58Paq1UKDVwV?;-BJW5bi2${R|$%D|9eS``PAT>^Mp<F(4N'
    '719gWUd|N^FUyii*9R=BioLthbYy=>>Fz^WI%q*SSv#)sg;l1zAm&vX5VE$b(i{q--wyZskgg}B4-)Uhzt+dDGDdg|;hkiwR8!ka'
    'R^>98Bq}l+NtpXSim*_M3Cq87#PV80;a{{Z7Lst1bd%76h<s%yBX*(DigLhV)5E;qFr42XV#9Z>lhNiV-D-RI6-CAv-Ua~NXfXIe'
    '3$E#Xp+M4-)h}d2P=pYouvQhj^cTVsbJ}baPGnYr4pOoXC>r#inwScZ`-o)kgw#S(g@!bAgfHFD!r%9};KtUZLu3%0a2A0N7ut)='
    '-pmiv8slgb5SebUAv5zvDn`16a^^w>k+nn@+r$AIEl03EkiPpId=h6nBoj283%XmZgzCd>i%f#^cmqZ9SjEw4i31r4C&?^hv@8y!'
    '+^dCvK9S*ZTxv-U$5AZ0!svcW<S6>}0sn<^${(<*d|F<|@d*tuJc2bm!sK&)WF2VeuVy^d_%kkhTHr^k$4B*c#V@k*UYdh!jT+TP'
    '_G(!3SG7>wRp1q-64lFThM*y;DFV(_cm|8ri`lrDGf&fw4O((X#@wnmTv;yJA=jZxVhE4yUTZ4JBW7^@&_8gceGqj+ZcTa*n)__G'
    '!*gF)BQV*)pTz&HNvF63dPoDV)8U3Pc~y>~+$7H7+H#Fm%E^MGDFhH3WQQ_k06=8AYImyIAQZc=vOE;H=Nuu+AfM7k#Rbm__FDA*'
    '<r@H<xW3E?(yp%|cc7nD7O8kFSf!O3waB8E0vBHZfPjNT`YxD_VoKFl4GA8Ds0>&*wWWjtYy1|9L@EPA!HkA_8Gl7J4F*qC&IHbC'
    '$cJ7ae<XqdtVfYINHai;AZQD|4UjS@wg4=`^1>4+qKSAdxEtnjWgYV|T3R{9YYlZqRW&(e8Ad0<6H48Q*@*)I_AsAuQ83PkIddU%'
    'O#OFx%O8Gv|5g*izmT<B0M&pLbxW(EcISp?a=yaz;r`=Kf4#rl@2K03@zpk2&f31jQ1LhnL!+?@$1${BRq_80QFKyGG(1x7=NJlt'
    'P@Oa;zX+W-`K!)bIgr49G04y_r_PQvC)EHX3E}i2#R<>e5J{V_CqR<#tJGLk9AdB-bKi}0Y14pajB5mLetXa7sV?W(6m*CBCqlYa'
    '#?<7SDeg(=8?LsgAyKvaO(plJdzykSP@OajZ0I`1q!nIuA{VGQ(-gI#uUD96TYqIT-HXYTC<~Q}lhs3ZBPow6hSI6DXqB}%1?8@a'
    'OHL%nWd2VX<n+drnG1*6l!;wsx@AkxET{tgH$^YPH<}^}@?RT0u3v}>He1}SBD%W$-KGqQsvT%*p{&ZPPXR8f@_AE!16Isrg;$;A'
    '6d7g}dT}y{_-n8VO}#niHMRKbJ$NT8yv%UY9n`ath@oD4YARQOwMfuWw>_0;u3{JTRkZp}S!}G*6JbdqP)NzZ6*bYJ)GLtIsg1ko'
    '3(CQqO6pdjou@Ww*sg{k$^4aM1#FZ<6PT+1skCmbXaWbjjqavCRW5GG)B$ZZw7S6(hs-n_UmmgIAQ+n+P4=kP!>Vl95LInOjTnw`'
    'ZDC^}sW&uJ2BLZQ7+)==@(2#)G4@nlS;M=^*ck#r!Tc~L(tFHC^^hvOhnD@SN>0yi4dpSohn^H<p!WQsupsh&L%~8SRL8*jlM<>z'
    'PgOBwj)nMThl01TXAEtaAya86Fn_nCw9K*le}}f`Q0EMmes#`J*+s|%9CAev7#wmo{451IKJ{}!lk}bjLlzW_ZsWrKSR9k(a1Xs9'
    'LkpueTUuW(zL9$9vR1K<F%CNC>DbV*#@FBatL*5?^I<uJ?uMkYS{uvZz8lKPji=W>FDqd%{)DC(jB#bq25W>I^1%wCsO_)~p>OVy'
    't+Am9Mpf>ba6jXIAT=oYh@tQWS-2r`kt!_B(5u2~Fyuj1!AXab!Bs3~Lxw7qs6&Gqc8H-Dtb%$1p`w~=pckR_YUmZb%4;IRObNWF'
    '%KaqaQ5C%!U_}+%cy^7m+x+y)w?F^<<9DH}|JR?t{rRs#RZF(%{_<K2``6vYGFlR$EJLMAl4OdXo6=O2Y^s97Q<0V`x}oR1w@}`<'
    '?|=TA^@s1PuGI-0&i%DiKyoChK9O5c4kS+gpU?Y=B*!P8_0xQ(42r}Zj0=na$e=+k^mQGF@(|5Z?u?klh_^QRDV=@J(LG-0`}ZB3'
    '#Gs0@lLHvA{dLezOqRP%OcX3yCl)^{aZ~5&e2(Lcm3UqrZ;oP*=bFUO2iYK_Kb?Q-#y%Pq3fAOlKiIf#j3c5<ec@3wV%;mTXrNRb'
    '-zD-6Hf2yv)~UlaZtq-^2&Ch-KHqppUuSpY4o9kEKIa%eM<!_x<Nr|5oQ@fD{+zGtlj{N-H^}%pt~`oyav-a;g(n_y;HNl;H%ej|'
    'i86QN7ddVW9N9|HNO_z9;~@KGWaogbZ8%>@R)&R4ZQ{P%rFUoeh{LaTP@bbF(sY7#^EHjDBiu}$csTP1POkf1E@RY)Xk~1l(K_Pb'
    '+s4&j=Nqr-YZ?azL+~hX_aawqJfE|PgUZhVnt{#Xa~LN|Rn>q8dVusDrPqVE8^(meG?V>69Bnv<3cHmf^#f$^2*$au8TniK2MipU'
    'u;HA`F!D8hV9puF9dexcHu!6{XE_<xY&+*NJUjRzzE}0V68-6WtuuoLEH$H%-yusk;2R!P=AYj#MkC+Fvm1`I`#8U4jyQwvyo^@^'
    '=aoF8eU4{+=4Ha@Tjzrq)y{2u6d7x11Z3^6<DFbM&!25)j=;-0zu=KMjSwXdF<;pt=b?=99E-0{?fxJioX@`!fm&bZv?nh%x^arF'
    'JTqwJ%p&K-H*$Y2&*D7J>QNp%fmf}uc#G<1;4Q*f0gr+G<kUtr{=nyRWg|8Hjh0co)$4LOE9WqRlQ`snWnhFck)Gp@vT|k^6WQG^'
    '2Zu2tF)Pkgje}Eh=6)u68*Jaj7z1-)oQjd<=XiZ^Dg%7Lge7Famp@?;7$Dx}l{qDjGiz99Ci~k+j0-DLKF_{h=QE<L*az#_C}C`q'
    'W98tPyzcP~pkDGiUKJDX<@=)ORuY8(W~=YlZ#n;ZVf(yx8RIp|esFIC%=Zkh&pcb?y3`MLZGe25#F54|oB<eTSD!|K!#zKVA&p`s'
    'vElau%g7kW>oAwu?WYmg*<Rj9Rw^3a?Jzc!H5S|tto-9LUK89cdysLQ!+13#vl$7~$S06hlU#_`+gaVW^c(3CudDf6j~Q^HqZn7}'
    'b=4wy9vkh!o}TNl`hmrtuu%@<@W26_`{sA2&F%?8ZdQ*IuzH?-?2of$CbXrJpuUk+1-(a%R1G}ew|V@7zii$wz)>~tFPV|pXJ4LR'
    '3kICc6UMbcY!Xqz+^41y>ZKkXLX8=&lZcu694{8kn<cZiYOiNLdPM~pIgL6BUz2qp{G1+>7Di<EBv&+@VXxLVWc4;)%^CJ;6~gtb'
    'UDGQeYtHnleq{T+AEAtAD~{*vUh^5lu<Q>wo2FNI_OEH{)%WtMJDfXWtj=!XLRKe!YgE?ec@kL&rqc*r_`@c(H(n*C6)mx6CP<NK'
    'B}NNbl9!zpz@&;`TGbNt)dayaiEQUNq{yd{!NYQ`DN%@^W%JkM3go@QnyB3xfE*vs$e+_+*7B?toCFI<&5wHqaZf9FTB7rI62N@J'
    '$g{BY5g}Fn>w1nCKZEbiw*Uh-sllCKIi@k4D+OF>vkHN6nK10G%W|e#-xm0-$v(J<QKL3`x5tq@z1v~&(m~tzA|uz^>o^!w@bt@y'
    'ky7x}`5f0yw^(T>h`<HM+vMvn&&|y>jm?!$ez#fcVBcC7JrASTVwahq442osonQIlxWA*Xzt2dU)CNo}*a?g0w9M^J0b9<ugz4n&'
    '%xH2Ki>d~H!F74G&|_dgCn&4s)fgP%!pfc46^lJ?!neIH&-7KEH?7XIBIU`-yr^!PU^MeO;F8y66lv#qa$dA^&4I76*ck^J3(I>Y'
    '2*`CMGP1sohtHYEYFf6UvAb5W2jmfmzX0;yBhbFUVuc@YRxR(djeaXe{^-dU?4E-fn^o+EX%0SZ?z`sk4pvDEpXank<B@L<Ws-xk'
    'M?)r<gzj+vJEM*bnQd@Hi<Q&+=8Cj)xDNY~nEdWG_9sUcD~YS!R-_HUb?n!0FYme70L%L<<9XRCW6xUEXdC+979IQM0&U#UN?KTS'
    'eez9_Q0?UQ7al?H<l$gy>8!L1yhsOv;&<s>d9hP;?7hX4yH;cYKxv+##dSxtvE175cTqn#xMh}W52$@wAZ{q`)xE&TG>%h`!79D5'
    'HW7JLA5!z+Z$!-l2*3Po^*LrFxIAX`4DOTSn!s%w!aVqUx2^TDE-Tms=NN<Td1R%LJ<psnUfgrUoUF^ZbAPYIPkVm3cAT*R%Gr7d'
    '>oR*5`z%(|{nD%kFyc3_OE9u;Y(LQa)%z6qW@DzkzL`63!a7)~WttoOy=x(eJS5(I!5(34Y%82ldp7Y$%!M>vAM4r@*Y1)F^d3>Y'
    '*UV+L4NH$xTWeu(D;rpn8MZX9|5h@g+GM(h)@!Vp-LCN73Wgi|Byp&vrU0?9v+VWs)luFxR`^H;i!B~o!tC$U;=8c*a4%NQvfT!('
    'vusy1E3L)Zk<AnD)%6?)ss%D|n(Z4L`D8@`2PHd=ylcIVEob!11Or&QwWeWcVy>Y8t<3H`t8^hZI}I-a<%kHhnbgx3Ev|V;cJ|A5'
    'EZ4VwISP2TaEP113NPCj2Gil1D{xcSkqH({v~BfBhRAbH^YvkSZLYV0qoKLpY_7L947K6fEdXq`n#N%}&bqadzLnX9UGp|avMgr3'
    'P|4aGv!%KgsP1G<{G7?A$X;IzucD?4|EpZCzcOnn{ZPz&c0F||#r_C$yj;F705aFMQQ3YUur+8lG<-(gJxgzNvF*w#_9{PsL0W!x'
    'X7*FE12Wi81reehN?UlgRM?1JM6sSb+Z3}KGu^b@H*T&A2hLFsfC~un*r#5G2{5y8znn^*VJFp2Rh)ioF4v*V+s|9AOL3mp{wc4S'
    '<%$few=cx26w}eUrfez>e5%o0>w{G{PSsmKbx{83_vYmzV&$TyQ1%4A!j!)YYL~hc>fnh8P<cwsU_DhmSfA9UY`mNc;t|Imq>AbA'
    '0|X{~ybGla2M54(UOozEm`jMDYOm=pB>e6&G5wYCj*_J~&{_Hlfcuyv7ih?QE^C6(L`an<6V%iOB*cEhu@H1gG0ma%8~%>qcwBTN'
    'vL@F%!YyM%9V*GSo}cXuz9{MLSdC8DzS{t{aQq)YKxzsY;94rw!u>+ZU%|FcWG1rDgiOjfizL(SPRgi`@OSDXL2*;bhzQqiO`Qwi'
    'N2r9{1r?jHFvS!KBB)ZIhCI2n$olIotCd_7LqD9RuIo}p(0JVwRLDIW@tW|#UQdl#%;(DoQGVZe`^zh*X$xKNW@=ZyW)0iVf<m|`'
    'US+T~V`oZo86GdPgX)U-4ZO#JMdd(QMV$m?0VV`k9m@bF;BbKgcUfPiY(sTuPwX?*k=RWVcwAZdY;+GbPnN?nZopAYf!Pe^8FCBF'
    'FX*GwPP6inSdC+obZ4zm>Y+LU3iMz(B&U2FSr4kbYIa{iS)J}fVtTqX3oev#p3FtzT0PqpC!Yjmjb2m85To!dA3m}mDtQUyn>HKJ'
    'ng<Pjhau=`1Na0Qqp?E|Rb4bEJ0>#q*y*xd+6e?V)6h6RY>B}!O*PXpg6tyRF-<n(CTuefPm5Z{k!>}dn36ecKC5i{x<cE`I2H8G'
    'A#bG_U3=^&qu|Jo14AB=6BAMqxIRsUne&a&cjj(_04*1Pajxt21t4--YFZ|oD~N-KGPof()tatjqIb_g6ppbivk2VVOcwd8nKk$V'
    '-kQ>t5;Rwp4|PGO6f4ig(-RsUm2;&JbV4|5f%bXL#6~x=SNV?4Ryfq*J+mhFuy=V6upav^)ONTaX=s1z&XvfL?86Bku{RN;8DiJ$'
    'Q11<eX!>9nX=UwB^YA#X-%~uN4+oj1uFLS$C0%AiQp3I!kfHpU;3?ht5kZRj9H&^XQ=>z#_oPd>FWF(YWke0n0PXYk0f#EGGDRlz'
    '1L<%L4Q0XO49`(L)|GX4KS(fqxIjb0-VXXu0BKNLKNRH|+i4NOGPZ~E)KS9LK7sKdUtlCfLjy!ov!Zd};4ou{^O?&(qq*xjVS^AE'
    'CSIuL+zqOP@vG3vaF$gE#xl^m0gqzU;BH=m)DPR0o$80}V$r2@XUk|`CQB5887r$yn-L@~upN%XT_C!=;{g4#G1YX1e)a&KKTp8f'
    '%BHJ$LAsQ)`LMWq$8oY}io$zj)wDjWgKBb=hwGuWDGcb?r-B~C8<pa$W7tUmfM#30>St_f0B~U9o6>~JqXBrv+z3K&(A87BvTJLn'
    'ym*%YLjvGz3U%t!K_LaGXJ#{IGwSOTOcHP;n~z|goMy32j{!Qv1>5(q0o>)idxlHQ1=fmnVz3$h7jkf-HAX(k4$+W6=xp9Mg_m-e'
    'Si8n_E>Gt8befJUbXP(MjpfB$qvS+_u@t@Qsp*wKtQ?bqN46w;w(Qb#xr9x(aBfX&s6(G%`6eg_B3;u-GYv7vAci8g4ozg!E8_j%'
    'g*6Bb%&K1P?*(r)H}6MXIeG6PwuX0xd&OSFp;?O8ybi(4N`97miC^24FZT;<FU3NoZCo4Hv+b;rEa>f&o|eH~^XUAbkU&RoquKSx'
    '8iTuMT3H-@?YWp!wwy4%4BlP6=Zte*BQR7Oy_b)6K=2&B7w_#kj||6UiY9#>$<Nm;J61b#!B?9A+{Sz_GnagU;=L2-I8HB=Id|Iu'
    'z70s;k0{9L<pmK*h6`_uN59)nQ45TE5kMcpnaiVe7wddOH;{C{3^pIsA{lltpwmBFmw@<&CRl3FbTY_oyAo=2WFMWB5@ewiKQprO'
    '1<;B61mUfGW=ks$WY<hPJS<=6dBAfbI;Q93xJThtkE9@Jb+s90$XRA7d9y@gO>;%pUzvI@CC#I_jtn2;vOqrVkt_X%mgSTwg*oPB'
    '>AjLBMM~@qsHx-f7(lWRS>C+9>Bpd8FWc9zmh$8oF3WwMjac%ak@iG!wG3lWWpq~#*rau4lb7sT<VY8#^mV><II(0IJLq!V|1lC~'
    'fmE4~Q%vTSgMJ8BUiWk0gAGgRQP2RyL9$;hC$Xfs+WDqQdw`^)<uZVS6c-Ty(U4wHB#o1#op=UhA)Fkg!V9t59B&A!>Kvq++*jfH'
    'HqQmx`M$LMU63(q?n9(6oOm0S0<rmIxezZhrf1Lji)2KOTk7Ks4WT4(#XMs*Y2i{v#X<RrWKXt{u<>&u;q4>=vO!6irr>N!*o<L0'
    '8yiK?=T_Tm&^eomBimknpy$o;Z|Ut^XKCA9H$`zji>_#iIVUB#Z8$?2Y>F?a?VD}C(GJ}(*0<U#ttp+e(<Mon?>r}AG4z9dHrb4#'
    'SRP3gR+_;~mNzm~Zf?^^U+D+wxDznY1)E-?LN(k*QcXxMVr6;uy_^^hf?R@aZDhWCCT}Xz52nPJ<4E3vT)*4>z*g=rWYf<wJBGF_'
    '<aC((c|0L0$RpVGtw<>BlGGaFX}=QYA`M30OXnL{2IFcUZD%4%3bg&z#&X`sN<fvI$0>6kN{UELw5E~pv~~SKczp=qTSg|pE~=Rf'
    'B_TLdY}sf`jxtVYq&S5`D501%5z|kR+m+FVVz?UtxN_hrR8dC8G<^OsF;&YuPkd-34NRv^z|FsPh!(ks(K66d2p9Prg^rw%rpRKL'
    '<_ennF&$RfX0V+}3seSV64<N708a<w{=c@)HP==h2f`D&SIXPTV0!-3=#~1)VXM{W#}TXzAwU4fu7`46B<(7KF~YA`fIA_ZhOn&B'
    '5CwdvBvJ*SBYCsCeY3n-1uzt{(p!`|WHGFK67g!yDD^&HF4Z@VTc)8c74WGf&?=$>xx5L1r#PzJsp^XGj8fIrvGGlXSA^d}yf>y1'
    'imxkTEg@O=wOa<}RY4l?l&UujaGBroNQg|;8=f?JU4Q^2_rA#a-U8ciSxLPsYEl8Kg{&AC<pi;nSCEdMYLnw$D;5O&Le?z+8ADc>'
    'OTrQ=6;$AH0xncwb0JF?plKlsNCYg!yr>8?hRA}HY2!l2@9zL0G~SfAljYeG_(KqxyZtsq=GP`rn9D3&Koev|x+IqpvZ_R^F+^(w'
    '`WCVlT^-nswvi-6T&c_J`&wLXxlCosD)(5xN@VZ8NXg!>{BL;zG0}6U<KDBblxbtCZz?dy)O|rJPb5rWpaNCgXS}>+o^h!J7=kl-'
    'mNnjFg#=_XWY4rDX%qSuFOkth-}j}$NyswK`(AUveD?KyTuh<#O<aw#+2&l-{?YZJM5k-sDY<DZNxU=ee76Dz6!_r2Hz?hS<E?uX'
    '>a)Z|BQ#FPPsic+?sDMu=2e+c*NXDb*7cMON9Y~CR4@zKz)&*xzAi&b5gRGHO?rC)H_@}xfQ)GTtL(Qm9w-N1*hf{wb(b~fX_I9t'
    'D7tR$kH>?`DTn(QZ~F?YM9&mO8!uD-R6CTfE=#ongO06tILk-Z9Ecc7-x4;98}<FhlF|TT_%B0D^lm8>i0*gR-x^t>PvuwQ=vk2b'
    'm7pZID4-Ir-t!XH`YqpY?xFqAvc~6~9p7zK`qh0bhN1La7WCx$A&0>_zh!>cw|EPpAteKd11f1(2nD-L&I|a6zLi_$ZN8_J!}VU('
    'cVWvy1i0bm`^a+TIt;-?n%WUOLUwJ-rFxCwill(<yMP6PfZXRuzwh_#2<mZ{fhOo(u>>!uEDIu_$_iz>`!=Y+B?T$)I1wb`K2z1u'
    '4ZmA<<Gs|vPzk>qx9l&L;S0f76?|c<Uc>gf7fSl>cHX;PkdBHd?B*M^fsbg68RX>e&M-jgZXbV^0ZwU~s_iwd^o<i7q$;Ag%Udcf'
    '{C>Z`z)JkPARry~zaS}H+5nns?3-_`6q~o~M?)p#t#3id3TNMl_qR-+^{oeVB}+j-PWwjJUOxK&|E*}ND?4(B@zzAR0yNkdeuvVu'
    '8#mg}Ol=RWKc(EehkcF&8N#R*W!WwK;=Rf$NMcE@tos(0y8^A=!}21+>4^7mqt5kbMMGucAFW*S9_X<MBe(7Hz1+0-X<AWt1pn9q'
    'd)yCSRj((mkV7ljLh81IE?5Gm`s#qq-@=Dh!1jBf{UT@cJ&OI))pWryUN|x13IzAm-Mq({BXM?)4L71EDCcRw$T{jH?Cz?*XlNf$'
    '5b7x9e}zi-FSt_y1r)G>tRy(UY8Y3?kQF>tw_~O*q6%U6@ncwXe<t0r$vj9wFc35_uyGZ_2U(UGK%+X)3gO`B0rG7dOXR6a@)oxB'
    'dT*_SWnFC<J=}6sMrLed_l&FD#z15Goq>7m;6;Zkn1N{<o2dh04WB9Abr^xsw`$CYfI}6_D;0Y;e%V;B>;fD^>c(5I)$8<I6XID>'
    '=-v2CPQSm0SGHgHax>Rk9(!Pg=inY~f3h*&90b<N``_-cf(1ICs_l+y1;gN4__*PwUH~KwR`eGxJW%8#2+Gvm%$30kHfxj6K$=98'
    'SXWW{ea!f%4_w!X0_wX9AWHAtQr~8c-YL(=*uM1up&Hi4EkmEXGJ(rtfUFXBOm9{b(DYkTc>cWJ6jK$y4-|NLyy;AXYXIgrxbVxZ'
    '(Q9y4j6}_jvw;ln7)VOROOWn&t#zgW)<+dK-kHBa!C$1KG?Y(^u(`ahL7*~W*f4hSx=0cBxCcs%LG0$_z%Ok;y?}=lI`|YUsVvdJ'
    'o1g#{t)|xpe7-JL+HBzQ1~i!-fLQTC1^mm_z)R?s*@7y<25<`Qf&n{$;bM0SDrVu6Es$7Xt~<b5*)3OO#K4x%>d$zCQx=UyM-Rb$'
    '=fTjd2>U=DnZV-^x%F5!vm?OQvrVx%u-3JDGhW~W*oPZHH+mH|6QB>f^Z_QuzTXUTOx4lA52cNqAv5@G5Z3BS)In@IIc=ajlW|Z0'
    ')eY>$*GW`<#o8e-;p<Fb`2u2cNQI_$aP{6ydBV^)?fM788(>AVE9D(4fP~Oz>}rOMZw36v!2Oul2KzLW#`(CmFwH^oU1)NH&qH7n'
    '5<_iFg~g1x5@F|S4P?^54H&?@WCpL)So5^>YJkz5VKe&=`&TvW0<o1u|LG#u33L#HMD83BN*xeWu*x2!1VG1$$Qr82ToYq;w1Sm`'
    'l!3$)>e^_%RMkHWGpzbz#{fIm`=9RU2q8tnyRzRcsdA8j%W+q9oP!-PY!r0RpbGd>nIE&fqK?k^vkyMc`u4@Jk+L35v}$B;Y+}qU'
    'pY&wL{8sc9^T`$STLynA?>6Q>oC4)Z3?yrXBZ3nu5T|MDRdDki1vkB(rlq_X*QoIzls@%-E8HK(bpa~rzjSXV+@X*>F>Rl!n>CF*'
    ')M>jr)w~uQR5c6NT=^h+rrsQbQExX7wP^MK=sVBDcX$~onyZC0-2l1BlPt@U-%49fib#@9S5|i31w<s{yA)H3gyI8y$TcsJ<QT<8'
    '$nzL^l2D-<(8)-a%x{J4RCq-xERB>xE=C!O-*MD{qR(^CZ3`~N$UP2z03COFGL1xZMXXs$$-fn>2sD67hYyA@o5b@dKHr;Elh<=c'
    'mhz1Xev8VsaNuq)w=43m%lO(Caz@I`-s?mKet#EQHudL_Nz)XEctIq(vkkz#NYcP~;l3dc6d;A5hwr57#b=UTyk~WswY#I}jQOZR'
    'XOP*`xZxeaHjS*KWCzJs>4F@nAi8X)K)SnsJ%C>|iVx43PVC5xJp{DchZEjf#w1}&xUf9};U>~hU1k#}=tgaYTmXf0S?w^sb=(bj'
    'K|%Tkj-kG7V4sTDw{%Jw1AFlv6$W?iZyRSRXU;>(b~4X3ArKlys7?gsBUC5akT8{pG>J&U1iNy7J9L7b*MB2?PCpUy@6dS0yI&i5'
    '!VX;rikApcY||iX0pKntgI$gl+r}D?QWczq%wRFu08-C65<W6#4>D$|37XQT*&P$<&>RrDUHBWqbE~|Z$C#gSUE5$(WTyJ&{Ab>a'
    '$m_6NH@V9KaR6S2-dxHfk%EbabQ%y}1(zUO-asWlpar%B0A=>1c7sk?VQl9z>cZ5f1!Ydc|7%kvDQ{V5M@z5^1Zhclp%s(zeGK8#'
    'HiNo3$j<g??ZO0^)_$m`_v0#9ZV%a;in*7|Ow$`v-}g++-R;$M-P0n=%cAw5I!;1OZvC1ou#9O-(z~&R)XnGkexC_Fd4H|nc`1^i'
    'y1|u4Lf*>@vm>A<paN`pRMPBd37W17DYf_{^DhYbu0h8m_Adk!XcLqLU_XA3@%CNoJXr*ws|c7NOfacpBLjh$g6^^`k}T-pZGa7e'
    'tk7LXAO+0LwAjRXWPXk^7@3F{%xXkO%56;Ko#_dJ2<{b9An|&FkUvl#=)2%IpbAf{u|CQia?|9@vF%zC5nnd8!V0V)iMXE-16EXs'
    'IGB8k$(8D}h(XB2Y$4LED4l9!ZlkNAaAGy23J~wGkrx61J4LOjZY=DxY(xf;4^a}x`A9iHhQ~l`A)JA1#3THthzadLZZBLtggl`6'
    'iNua++aG}st_HFKh`)&HY2j`VTSAdef%R6O&wg-qtoHc5WAS&Au8gf8SML+GLF8r->2C~iAn9tkQ;pad*+dey2BevTowO#e5?UDf'
    'CD_l@MrI-dT7h^d*$#Htnw=h*sgW4eJ8pCcqqhB3qKM;~ly=g9X(JLOeC8I9WSI0RiFif3TXV*#{NpZ&n6oTl9-S}r*(rmDE%Swm'
    'xUnc}Q)~({N|<DA)j6hQihtV-nOKodlIXVH*9WsIeV_DR#^fswhE}Sg9gMD=Jd!~bl^cn^6V(dD?0prHknVy^JP;hk`doO@P?mV`'
    'y%97JQX{F9!FO?EsKnPAo9{g{%QPT8W0lSqcVl2jB?-}`5=o>UPH|!OM1N$W^v0FNlq3X1lrepuWP5pDHRh616<I*)Ul17e{6Pvv'
    'u3oujFl(f93vxq^dq0gca)cs$nJL~##B8AZi|B@TzY@kuE+W4Jd70;%K2s(0_M&Q%e?)j;gUhJ#vX4|~8PlsWRuE)R%zG?w#IoKQ'
    '$V%%Cq3aOBTfK@IVoKFjvb%h;&0MY!X&IM#XOInT{NA(A-@V}8fJ{g<a7jc8?2sD;DOw=L0R0M>bw1lV%zu`k`~e<p!$rYvZa@ZW'
    'DK}$5liichV!N0CKEB&0ID(;*$hX+ySv2GrS<%g;`Bw^!;6ogb1*{%w(`?x+QgDp2xKXQSQbdF08P*y7Yue9_*IF@Jqdg%8vxI0L'
    '%n2>DXNMKoc2SXUrt|eQ@`^OzI(e1CR5F&Cx%6ke<~SA%56^2QqJ~VU-XcZ-R)-k4fj8h9GU+y^du5`;%RI+e`Ar+9bz=-H3PZX9'
    'BQG)s5E>naJ2iz2vU5BLqpy?;fr1zMz6bdk=DQU<31ma=%X*Z&08IfO5jqp?G(>Hf*>Uxtp_Y|I%0L{%RKIA+DGVy+^|SecR2div'
    'LQ&=&%)5zavKouk9!>E;U$fl%dGZ{LNn5MwE|mClEU2Lve_>ySz>pbFEzQC5cd+~cBz-&{X2Ig`GyCEW@)UEvi<ue%Y6rMI&pRmu'
    '<_;tp3gSv7M-*n2IV%YWRC-#W(d9eD;(MwJGmZ}xmPQ@{Vy@=uFiySfaj#_z1aI#T#R%2W{k!a(4&ON8T8;=yx!g+5!A_KCY=?<E'
    '&&gRH)B2sK@!qGi$Q{VWz!m|@M=a1<O~U%?Jtt@Kmbv38mzAdgDWYfoV1}r7%7MqKcg<M{5lyuZGHmL6c`w}f<OF7alk2AQok}?Q'
    'J?Jc1Nyp8b<0;vT-=r3ng;zkRDXU!?lh)$DoO}~qpnA_&(QUfrvwstC-$gStO&5ynS0j0Z%9Jkn@hk712*F7;BYC|WVXBF;Xe3tj'
    'vik-PgQ+&lFe%^Ftn;*4A@_p|36CmS=SC?{XxF>)@!T=*3M%uBz*WGYyYk*ozm;G`33`NWNFTQ5ymzoqV$5!WzRy1^My|0!0!>BO'
    '?|kwESMM(g*k-(3%=Q0R`8vOT8S86gk|v{oF#qH&ahEdF9^0?b9IUqC&gwc0l9lw2;NKoDd?ldyo6o9B{9PF@m*1R<=Mp2kK#IJf'
    '+0dzU7*VKNvI(u-l@8wfoDb}9rEn?+T;#Z_nXf)m)fM?ZRJ^=`dU}JSMByJdERvCbF0ZdQT9<Kv5|A@Iz#OUrt^#BM2@9kNbl`)U'
    'J%Udf<aedKqA#=H(3`WOJBW@&T`B&9FoHf?8~j3jcrW?m?9~5x9bI4qG>Q);<aY`>rz+;ctvCwL!wf)pB%&{t$!LY}QOq7f7$J!0'
    'A<P(j{Q!%ruL)Q?WF6SvqEW&$stg!1!NH_TRHZ7$L@q-a3t@g<Ke-pSV#IpVav+lHKs-Qmg$_1Xacf5^Lp6lXw^P42N6NBPgF3G-'
    'U)L84`JzrNh5cpo#k_5j8`oU&>yP_~HLPxX2Ztz2W?2Gw7T00CDJJd9Kml2JB8`GIrsYj!nVD^71m61-y(0cO>wT%?%x5IaSbTyF'
    'UnY~>HSq&zc{1}7yCFLl5OBGtu!Ot~L^ft7pvu?DW5IwV$14<i_U-jK<#HviHZU9<WHFHQA=e}8DlFYKQwglzYoxi`(yJD8H;|f>'
    'yqoH2fL$y-i~kmqw_(X!ZI9$F=7!|IWOa>=re6P`Al#-TRmjdva+G@aPYY)@%nH9gBVehl&4eF9rUvZ39`=b%xn7zxkP227;$0tK'
    'FXzT8#(OJ;>8A5o7#Z@rq<1In>O3uWijf{N!ZP_=WQ=s4>rVz~kv*!I9i9}#`}&agka_9RdBjMuCtrJg*fEoY1@JqknxcmoPA&x%'
    'zGjM&FndHab|O3~!ITNXRkM<b2&uP0h)S82ZP7s8H$-E(Kl_RB@>0Zpb25_Y57~0}InT>3Z`k^g-$dvU+9ookuw@$3%k$4R80pRR'
    'GK2bdI;%Js&y36(DS^!0ckUH|#=#w1$RQ&dy;HV%L9&FUgTN#$ciWX}dyd5@$aC?z!7-v;XEX<y&YJSenE~bMlHWxxl`N*7(nh}4'
    'GQUVqw<oEp=Q)qXa{B~zbNS>bK->%*I?vwMPxm|E`XMDzzv_VBR4kMKU+W3Z2BcPGb`gywtIr&#CU^B;^RUCLoNe;@+qd;6h6^K!'
    '=a|fVz|)5lPnI5r;OSrO;PL6Q#^Krrb&&oE^#%;~JYRnUh&76ZLg$;~jaJ_r5||Z2L`Y#Vv&C{+HrZjxnRA{5AmThoIr1fiJ5l1*'
    ';jvD37655IvW3iA2R6;o69|&qo5MqFT;FWg%lr;^h-^!qqt6+>6kioAC=Y$&%cGtRv)_S$s2`u0CzjFycLd=hs;Sg|r&UgN3T=6g'
    'uIz<bHHxY8xY?iVAK^S8lliKp?uXtApgmm8M3abWrlqnJiA+n!b~7we98;(%<YIFhGB6Vv*_s#&q_1L9kt|Q}3%T0>ho-8(uSNTl'
    'y`f$_oEvF5siDgMoug;&t{iJ73jj;srBlrjx)VBJD$27QVi8S=4iFJ#){bR>HDfLNRY?!=TSIc6jF={U{#_G8FW<Xyn7V;Bczw&5'
    'm$<o#Z$y@yI21W5aXyeBGgkw?h2l_V;PU<xD;G3nbyYUNxjSPH|H;SV*hM)a)$D%DRMRe3r8tj#@#m#bm0S89ac%Lq5gZXWr%)mC'
    'G8L{VL$z!lCr-L}=e`Z>K#>a@8eqSP4|dJceQJ|O`idOQlafk}^u|q3H)7X2rEh4&-c$}uVa0Xd(Ht01k*2yuVwaCe8nA<7RH*&7'
    'b^L7C`;Oz7W7PiXW5a_&dJtWyhO2p2<D2!e_FEjXI3A`Ceuwav#QA)9=bh!`^}6On?UxyXXDSCd_h=d6E=4zi?8mbFY0F9})37z2'
    'tB(|KcRv61A-`@7QCIX+dWyO_XHKLIh6EAwjWFvI-MPyxE8PgZQi=BVFz*g_>v^n<H1WJubS?H8Rfq}eUU|<sF(@w>6l;4$V-tyJ'
    'N2zWn+jSjsNVpA&{w-@~;u!6-iB5Hl_##!zIuz@C?PL_%UgQN5-Ubyx<RQ^d>i9(TDC6iG_3i7nuwoE?W!gqS=Yqttmo=F6Ry9*s'
    '7?-zi{j>aLJ3YRLAA=9c+QBAQ8pTnwx5@1@N>kK&5LYelD`?dma2*)kA9|$Ga<H98VubjaFD3DI8}E0jCx<!~Y+~*<UqnQ=sqY~6'
    'n*DMAiftLNq^E+ex;4?ZY|S-~c5Fyh4GRc8wMcOz%5Dn?350vOx}E(@DzJU<^28`3zGcN^#H6op{z_MWwMm2)E3sJe5nUfpV4SN_'
    'JP05+<btV$U?>1SDBJW`ah4}Ke-P0YU-D4r6o2#Z#aCYTxVEA%-rLvNr{ct04M3VtV*a;c*ql&`uo9tn6hby*`39;vfF@sSZoa}0'
    'Rt)xA;o~pZXFcY<ZXm<!gYq6QXq$c7D=pP(L>GTSp^q$KJyedA>G}?uDa}BleUX||@C1UlS&WpF<|epOGP}XjaGI7xpDyHYrOt=)'
    '%)d-l@;E*)z6u+C&_$hI(L?1C-?nvDjEzKb#lgqSAm;8H@zB-0^5y9^7|Nx@`0hAazk_BkOb?ti58iG-ibDOHq+&-fy0h7i$EG10'
    'uu-R6zz{0jDaG@MQvW%QL!yL7iv~yrTeLpYldqfzle!q9D8B8zKcJ3jDldB8qNFMmQdqN4m!D8ob9?W9oW;F=qhQ@d3B@BBI=Dxk'
    'D4T~+T9ia1X*=}#W$@siHhl~x+%ci?9#0Rep*u05b*FUQ{eAnhXzI77V1L@lYu8hfsCvpT84gdqp)vQ8GHlgiW|C<6_RZ@E@JiH7'
    'rrwg<VL%GxRqZac8uV*#ul>(5ZfpE=aHqK+pcA(Lxhh6L>aJs{H~5X0W8t87eSm5FH{(7)%5M4TYd51<8|+-gqEgco$mJgGeYZI_'
    '9s099-4sB`s!3?hjH{Z2bWOkB!;y9e2iKN6RsD8zFFKgFo0l%+GjdsR@{QdKcHrW|mj*(tN?S>hz<oX5?T2lj?((<2mt|D^j>Q?^'
    'HF774hl~V^jeve!?Wy9}{@`A8!<cV3wX#w2x^Tliu_L<v&xlrDxa+p^QRvrU0hv8*OBBCFWCMRE^EF1>YbKCoFYo%>2=Atw%-v)I'
    'FF9)iDpcEVH}??fnXBjpYVHz3_be$WwaGQg?fPx2U2Q~hGlA<ECU6z(_{K_RZyneA7T=L8uKw*tD<DGd@W*G!i_9MDU5td!E+|_!'
    '7D`EsE5P?!P=jH=ovU(&JxP(;n}t}f#2cz8ef(o@j5~OZYq+zquxjpgo>tyc)0(ec9mWEc_s)&&XZ-3~@=W|jH+u4VBfu`xO&P#t'
    '*)3bhS3|STX!T_3@}*Tk{#`Rk$1Ll{U4o_E4C*9W6K`Fk^XMIfKB3eWEKP?n3y07VyVYA{@$Ku>W=_j2{434ZZmfZ^(DHF^zC^DT'
    'zHlTP<Rt{!8_Q|n8MVDdNz<dme80YLH?Bp`+kz>P>~?ETbb(OZciA+?2-J*3ijm%x($O*!;<e@;0uGYydU@CyU_7F)G_W%_41;o@'
    'sZg`d%vK#K(`V#aiYiBl(%_sutVjGW3gQBAzXLOXSi35$Ac%fLQ&m!3-@(;y*Kc`QZ*VX<{@rBf;O<{tbGd4;ZqH|Fg#MOvJXo6>'
    'D>Y5Vk#o=G8**4)UaXh~+DKWeR$R=zwX|xi&|Pk}2fCNWc095JAoHjiaWjH`OUfSn5wi9e!Q3(0ifg^<54p4(9^~@9>fgD%Sgs@s'
    'gW-)GRJC5snomBy!AmDEMP!uwW<WY?qM#}PjIz<yXfQ>H+;C;AD;YGs3~KXhkBYq8$@~V$y*mavPnE)(jy0HO9G&YXlf;hfG!_YO'
    'U(YJR0Bami8!u_5KEM&GCi1q))C~A$?W2esbCP*5_Ylg|zq@wg?LE`0VC#%Xtlu(hgVUcCBl^eHs+K!*2ZU;0J8uTL9%~?lk@vGO'
    'gtJiKGAM+5+vooA6yj}x1v|pntUnfEN-nKp^r5x~afq!Jy5(4hWt(}7IrGJ2%Pou=$z57}KK4T?GCsj>Tdp4qwrnTSt_s7NtucF6'
    'O(rU}zCXIYdL2#p8bS$u`@TMEJNm-Rm*x4g6aL+HXYq_^&_h|MdcStw{h6;iRj5<4-62yk-~B<Ej~RH!WEFG9^`{@$wCN5Z@%`G*'
    'aYjA1?6kXk%hZSav*bh(@ok?qyWXC;qHAzicwx@Np47sy|H|z%<Kk-+&xX>=+#hT`$z14~AluEge*3yD>6O_Ql+4hC{Xy*d)DG`*'
    'x2oaeag+=#)6{dK2zuhNC)kVXEIf-j=eMQ#8Za*u*=Bz-14;niays+)C;MZPz|zSuaz9(qg`T6$Gg}%EeQjn&)4}a|>>SI9&+^sK'
    '=4ZP}CJ$yz5MajqU;a(D`HnFy2K{PkvA$*h^3?=gS%WKtR<*^!rv<40w3*o>I|9VRvw{bz_9cjiD}@_|EryljoR9wO*TDS0HnX-z'
    '+N7_9jT!4|oE4livmWb1@50N4$A%}vYsXnBpjjHbr#mMkmL(X@`>Y(Je^p~4Awb>c?VD*MT;x=`{(b*RvA}VL#ONFnbr$E4R{#86'
    '>N09fnWE+5-(sac^xStGAGiaBQgrfLA&(>uc884%^}G|KL1fR(lsD<nVWE+Kt!487e#sVf8yQZH^}(Qje<4|-<V&{?A}u;H%9VKB'
    '<F9ESYfv!KEbO;}S{f@52z6ENU*gY7?vssB*%R05n(R#HW2C5{3g@&A#48nENrsjNLVp^_7B%aa_S!w3raDTR-xbw{5lIpGPI7IM'
    'f4~M1s~@#!H40`e)~R?+=dsY8P^RL_yNvHxHa9$IZ=fM@_CJ63$wus2heiyReYt|dL?B*qcJfJo``~CNMBUfGYZ3UGB|(t)=yYo!'
    '77_?$fSf~mM5EivA+LeN5W??Y2(1+NQ8+z9-~~>P3iSfP2Z}cTk_~*f>13h)UVm1&XMI|b@2(6)Hn>e^{*m9v240KtJRa|+x>aGV'
    '5OrAT*`ZU6g`5VGFCsElEF!Q5{L|XI1bQ)W=_bWuR8A%{0^z!@xC<B6>OI~k8xa_8OIq`)3yrMN{a&<Xq$A;PCBQo(b_4)UqhJnd'
    'Wd}*4Z0)=8{VBf80x%Mx8zMf2;6#FMp;vS~_I=~sKz2AVoWqz(T0*-agR>8Ie(2~ZKMb(?+vh*`-+uvMaIiW'
)


@lru_cache(maxsize=1)
def _world_basemap():
    return json.loads(zlib.decompress(base64.b85decode(_WORLD_DATA)))


def _map_path(row):
    """The preview and the PDF use the same geometry."""
    if row.get("route_coords"):
        return [tuple(point[:2]) for point in row["route_coords"]]
    start = row["lon_o"], row["lat_o"]
    end = row["lon_d"], row["lat_d"]
    if mode_to_category(row["Mode"]) != "aerien":
        return [start, end]

    # Sample the minor great-circle arc used for the air distance.
    longitude, latitude = np.radians(np.array([start, end]).T)
    vectors = np.array([
        np.cos(latitude) * np.cos(longitude),
        np.cos(latitude) * np.sin(longitude),
        np.sin(latitude),
    ]).T
    a, b = vectors
    angle = math.acos(float(np.clip(np.dot(a, b), -1, 1)))
    if abs(math.sin(angle)) < 1e-7:
        return [start, end]
    fractions = np.linspace(0, 1, max(8, math.ceil(math.degrees(angle)) + 1))[:, None]
    arc = (np.sin((1 - fractions) * angle) * a + np.sin(fractions * angle) * b) / math.sin(angle)
    lon = np.degrees(np.arctan2(arc[:, 1], arc[:, 0]))
    lat = np.degrees(np.arctan2(arc[:, 2], np.hypot(arc[:, 0], arc[:, 1])))
    return list(zip(lon.tolist(), lat.tolist()))


def _mercator(points):
    lons, lats = np.asarray(points).T
    latitude = np.radians(np.clip(lats, -85.05112878, 85.05112878))
    return (lons + 180) / 360, 0.5 + np.arcsinh(np.tan(latitude)) / (2 * np.pi)


def _map_bounds(rows, paths, aspect):
    route_x, route_y = _mercator([point for path in paths for point in path])
    pin_x, pin_y = _mercator([
        (row[f"lon_{side}"], row[f"lat_{side}"])
        for row in rows for side in ("o", "d")
    ])
    world_x = np.sort(np.mod(np.concatenate([route_x, pin_x]), 1))
    gaps = np.diff(np.append(world_x, world_x[0] + 1))
    seam = int(np.argmax(gaps))
    start = world_x[(seam + 1) % len(world_x)]
    span_x = max(1e-5, float(1 - gaps[seam]))
    world_y = np.concatenate([route_y, pin_y])
    center_x, center_y = start + span_x / 2, (world_y.min() + world_y.max()) / 2
    height = min(1, max(0.00005, (world_y.max() - world_y.min()) * 1.30, span_x * 1.23 / aspect))
    # Limit redundant ocean or repeated continents in wide landscape reports.
    width = min(1, height * aspect, max(span_x * 1.65, height * 2.4))
    center_y = float(np.clip(center_y, height / 2, 1 - height / 2))
    return center_x - width / 2, center_x + width / 2, center_y - height / 2, center_y + height / 2


@st.cache_data(show_spinner=False, ttl=86400, max_entries=256)
def _carto_tile(style, z, x, y, key):
    side = 2 ** z
    url = f"https://a.basemaps.cartocdn.com/{style}/{z}/{x % side}/{y}@2x.png"
    response = requests.get(url, params={"key": key}, timeout=(3, 5))
    response.raise_for_status()
    if response.headers.get("ETag", "").lstrip('W/"').startswith("wm-"):
        raise ValueError("Fond CARTO avec filigrane.")
    return response.content


def _carto_mosaic(bounds, style, key, width_in, dpi):
    x0, x1, y0, y1 = bounds
    zoom = max(0, min(14, math.ceil(math.log2(width_in * dpi / (512 * (x1 - x0))))))
    while True:
        side = 2 ** zoom
        cols = range(math.floor(x0 * side), math.ceil(x1 * side))
        rows = range(max(0, math.floor((1 - y1) * side)), min(side, math.ceil((1 - y0) * side)))
        if len(cols) * len(rows) <= 18 or zoom == 0:
            break
        zoom -= 1

    mosaic = PILImage.new("RGB", (512 * len(cols), 512 * len(rows)))
    tiles = [(x, y) for y in rows for x in cols]
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending = [pool.submit(_carto_tile, style, zoom, x, y, key) for x, y in tiles]
        for (x, y), task in zip(tiles, pending):
            with PILImage.open(BytesIO(task.result())) as tile:
                if tile.size != (512, 512):
                    raise ValueError("Tuile CARTO Retina indisponible.")
                mosaic.paste(tile.convert("RGB"), (512 * (x - cols.start), 512 * (y - rows.start)))
    return mosaic, (cols.start / side, cols.stop / side, 1 - rows.stop / side, 1 - rows.start / side)


def _draw_world(ax, bounds, palette):
    _, _, land, border, label_color = palette
    x0, x1, y0, y1 = bounds
    patches = []
    labels = []
    label_rank_limit = 4 if x1 - x0 > 0.35 else 6
    for label, polygons in _world_basemap():
        for polygon in polygons:
            rings = [_mercator(np.asarray(ring).reshape((-1, 2)) / 100) for ring in polygon]
            outer_x, outer_y = rings[0]
            if outer_y.max() < y0 or outer_y.min() > y1:
                continue
            for offset in range(math.ceil(x0 - outer_x.max()), math.floor(x1 - outer_x.min()) + 1):
                projected = [np.column_stack((x + offset, y)) for x, y in rings]
                codes = np.concatenate([
                    [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY]
                    for ring in projected
                ])
                patches.append(PathPatch(MplPath(np.concatenate(projected), codes)))
        if label:
            name, lon, lat, rank = label
            x, y = _mercator([(lon, lat)])
            x = float(x[0] + round((x0 + x1) / 2 - x[0]))
            y = float(y[0])
            if (
                rank <= label_rank_limit
                and x0 + (x1 - x0) * 0.06 < x < x1 - (x1 - x0) * 0.06
                and y0 + (y1 - y0) * 0.07 < y < y1 - (y1 - y0) * 0.07
            ):
                labels.append((rank, name, x, y))

    ax.add_collection(PatchCollection(patches, facecolor=land, edgecolor=border, linewidth=0.38, zorder=1))
    placed = []
    for _, name, x, y in sorted(labels):
        if len(placed) >= 22:
            break
        if any(abs(x - x2) < (x1 - x0) * 0.10 and abs(y - y2) < (y1 - y0) * 0.13 for x2, y2 in placed):
            continue
        ax.text(x, y, name.upper(), color=label_color, fontsize=6, alpha=0.84,
                ha="center", va="center", clip_on=True, zorder=2)
        placed.append((x, y))


def _short_map_label(display):
    name = display.split(" · ", 1)[0].split(",", 1)[0]
    return name if len(name) <= 31 else name[:30].rstrip() + "…"


@st.cache_data(show_spinner=False, ttl=1800, max_entries=16)
def render_pdf_map(rows, width_in, height_in, map_style, dpi, carto_key=""):
    paths = [_map_path(row) for row in rows]
    bounds = _map_bounds(rows, paths, width_in / height_in)
    x0, x1, y0, y1 = bounds
    center_x = (x0 + x1) / 2
    palette = MAP_THEME[map_style]
    style, water, _, _, _ = palette
    fig = Figure(figsize=(width_in, height_in), dpi=dpi, facecolor=water)
    ax = fig.add_axes((0, 0, 1, 1), facecolor=water)
    ax.set_axis_off()

    attribution = "Fond : Natural Earth"
    mosaic = None
    if carto_key:
        try:
            mosaic = _carto_mosaic(bounds, style, carto_key, width_in, dpi)
        except (requests.RequestException, OSError, ValueError):
            pass
    if mosaic:
        image, extent = mosaic
        ax.imshow(image, extent=extent, origin="upper", interpolation="bilinear", zorder=1)
        image.close()
        attribution = "Fond : © OpenStreetMap contributors · © CARTO"
    else:
        _draw_world(ax, bounds, palette)

    endpoints = {}
    dark = map_style == "Carto Dark Matter (sombre)"
    halo = "#182435" if dark else "#ffffff"
    text_color = "#f2f6fb" if dark else "#203548"
    for row, path in zip(rows, paths):
        category = mode_to_category(row["Mode"])
        x, y = _mercator(path)
        x = np.unwrap(x * (2 * np.pi)) / (2 * np.pi)
        x += round(center_x - float(np.mean(x)))
        indicative = not row.get("route_coords") and category != "aerien"
        for offset in (-1, 0, 1):
            ax.plot(x + offset, y, color=ROUTE_COLORS[category], linewidth=2.1, zorder=4,
                    linestyle=(0, (2, 2)) if indicative else "-",
                    solid_capstyle="round", solid_joinstyle="round",
                    path_effects=[patheffects.Stroke(linewidth=4, foreground=halo), patheffects.Normal()])
        for side, field in (("o", "Origine"), ("d", "Destination")):
            coord = (row[f"lon_{side}"], row[f"lat_{side}"])
            key = tuple(round(value, 5) for value in coord)
            pin = endpoints.setdefault(key, {"coord": coord, "name": _short_map_label(row[field]), "roles": set()})
            pin["roles"].add(side)

    placed = []
    for pin in endpoints.values():
        x, y = _mercator([pin["coord"]])
        x = float(x[0] + round(center_x - x[0]))
        y = float(y[0])
        origin, destination = "o" in pin["roles"], "d" in pin["roles"]
        pin_color = "#7549ad" if origin and destination else "#1678cc" if origin else "#c73c3c"
        ax.scatter(x, y, s=40, color=pin_color, edgecolor=halo, linewidth=1.3, zorder=6)
        if len(endpoints) > 14 or any(abs(x - px) < (x1 - x0) * 0.16 and abs(y - py) < (y1 - y0) * 0.11 for px, py in placed):
            continue
        dx = -8 if x > center_x else 8
        dy = -8 if y > (y0 + y1) / 2 else 8
        ax.annotate(pin["name"], (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else "left", va="top" if dy < 0 else "bottom",
                    fontsize=8, fontweight="semibold", color=text_color, zorder=7,
                    path_effects=[patheffects.Stroke(linewidth=2.8, foreground=halo), patheffects.Normal()])
        placed.append((x, y))

    ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="auto")
    output = BytesIO()
    fig.savefig(output, format="png", dpi=dpi)
    return output.getvalue(), attribution


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
    map_style="Carto Voyager", dpi=210, carto_key="",
):
    from reportlab.pdfgen import canvas as pdfcanvas

    if not dossier_val.strip():
        raise ValueError("Le n° de dossier transport est obligatoire pour générer le rapport.")

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
        ["Date du rapport", datetime.now().strftime("%d/%m/%Y %H:%M"), "Émissions totales", f"{total_emissions:.2f} kg CO2e"],
    ]
    sum_col_w = [4.5*cm, 5.5*cm, 4.5*cm, 5.5*cm]
    info_tbl = Table(info_summary_data, colWidths=sum_col_w)
    info_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f7')),
        ('BACKGROUND', (2, 0), (2, 1), colors.HexColor('#fff4e6')),
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

    footer_h = 0.6 * cm
    mapped_rows = [row for row in rows if row["lat_o"] is not None and row["lat_d"] is not None]
    if mapped_rows:
        map_h = min(8.2 * cm, max(4.5 * cm, (y - M - footer_h) * 0.48))
        image, attribution = render_pdf_map(mapped_rows, AVAIL_W / 72, map_h / 72, map_style, dpi, carto_key)
        c.drawImage(ImageReader(BytesIO(image)), M, y - map_h, width=AVAIL_W, height=map_h)
        c.setStrokeColor(colors.HexColor("#cbd6e0"))
        c.setLineWidth(0.5)
        c.rect(M, y - map_h, AVAIL_W, map_h, stroke=1, fill=0)
        y -= map_h + 11

        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#486075"))
        c.drawRightString(M + AVAIL_W, y, attribution)
        legend_x = M
        used_modes = dict.fromkeys(mode_to_category(row["Mode"]) for row in mapped_rows)
        for category in used_modes:
            c.setStrokeColor(colors.HexColor(ROUTE_COLORS[category]))
            c.setLineWidth(1.8)
            c.line(legend_x, y + 2, legend_x + 12, y + 2)
            c.drawString(legend_x + 16, y, ROUTE_LABELS[category])
            legend_x += 23 + c.stringWidth(ROUTE_LABELS[category], "Helvetica", 7)
        y -= 10

    table_h_avail = y - M - footer_h - 15

    heading_para = Paragraph(f"Détail des segments ({len(rows)})", heading_style)
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
def unified_location_input(
    side_key: str, seg_index: int, label_prefix: str,
    show_airports: bool = True, show_ports: bool = False, show_stations: bool = False,
):
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

    location_types = "Gare / Ville / Code UIC" if show_stations else "Adresse / Ville / Pays"
    label = f"{label_prefix} — {location_types}"
    query_val = st.text_input(label, value=st.session_state.get(q_key, ""), key=q_key)
    q_raw = (query_val or "").strip()
    is_iata_mode = show_airports and bool(re.fullmatch(r"[A-Za-z]{3}", q_raw))
    q_iata = q_raw.upper() if is_iata_mode else None

    airports = pd.DataFrame()
    ports = pd.DataFrame()
    stations = search_stations(q_raw) if show_stations and q_raw else []
    station_options = {}
    oc_opts = {}

    if q_raw:
        if show_airports or is_iata_mode:
            airports = search_airports(q_raw, limit=20)
            if is_iata_mode and not airports.empty:
                exact  = airports[airports["iata_code"].str.upper() == q_iata]
                others = airports[airports["iata_code"].str.upper() != q_iata]
                airports = pd.concat([exact, others], ignore_index=True)

        if show_ports and not is_iata_mode:
            ports = search_ports(q_raw, limit=12)

        if not is_iata_mode and not stations:
            oc = geocode_cached(q_raw, limit=5)
            oc_opts = {r["formatted"]: r for r in oc if r.get("formatted")}

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

    for station in stations:
        place = station["city"] or station["country"]
        freight = " · Fret" if station["freight"] else ""
        code = f" | UIC {station['code']}" if station["code"] else ""
        display = f"{station['name']} · {place}{freight}{code}"
        station_options[f"🚉 {display}"] = station
    options.extend(station_options)

    if oc_opts:
        options.extend(f"📍 {label}" for label in oc_opts)

    if not options:
        options = ["— Aucun résultat —"]
        known = st.session_state.segments[seg_index].get(side_key, {})
        if q_raw and q_raw != known.get("display"):
            st.caption("Aucun lieu trouvé. Précisez le nom ou le code, ou indiquez la distance du transporteur.")

    # An exact code or a single match needs no extra click.
    sel = options[0]
    if len(options) > 1:
        sel = st.selectbox("Résultats", options, key=c_key)
    elif q_raw and sel != "— Aucun résultat —":
        st.caption(sel)

    coord = None; display = ""; sel_iata = ""; sel_unlo = ""; country = ""
    if sel != "— Aucun résultat —":
        if sel in station_options:
            station = station_options[sel]
            coord, country = station["coord"], station["country"]
            display = sel.removeprefix("🚉 ")
        elif sel.startswith("✈️"):
            idx_in_all = options.index(sel)
            r = airport_rows[idx_in_all] if 0 <= idx_in_all < len(airport_rows) else airports.iloc[0]
            coord = (float(r["lat"]), float(r["lon"]))
            display = r["label"]
            sel_iata = r["iata_code"]
            country = str(r.get("iso_country") or "")
            sel_unlo = ""
        elif sel.startswith("⚓"):
            idx_global = options.index(sel)
            idx = idx_global - (len(airport_rows))
            r = port_rows[idx] if 0 <= idx < len(port_rows) else ports.iloc[0]
            coord = (float(r["lat"]), float(r["lon"]))
            display = r["label"]
            sel_unlo = str(r.get("unlocode") or "")
            country = sel_unlo[:2]
            sel_iata = ""
        else:
            formatted = sel[2:].strip() if sel.startswith("📍") else sel
            record = oc_opts[formatted]
            geometry = record.get("geometry") or {}
            if "lat" in geometry and "lng" in geometry:
                coord = (float(geometry["lat"]), float(geometry["lng"]))
            country = str((record.get("components") or {}).get("country_code") or "").upper()
            display = formatted
            sel_iata = ""; sel_unlo = ""

    known = st.session_state.segments[seg_index].get(side_key, {})
    if coord is None and q_raw and q_raw == known.get("display"):
        coord = known.get("coord")
        country = known.get("country", "")
    st.session_state[crd_key]  = coord
    st.session_state[disp_key] = display
    st.session_state[iata_key] = sel_iata
    st.session_state[unlo_key] = sel_unlo

    return {
        "coord": coord, "display": display or q_raw, "iata": sel_iata,
        "unlocode": sel_unlo, "country": country, "query": query_val, "choice": sel,
    }

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
        cur["origin"]["iata"]    = prev["dest"].get("iata", "")
        cur["origin"]["country"] = prev["dest"].get("country", "")
        cur["origin"]["query"]   = prev["dest"]["display"]
        st.session_state[f"origin_autofill_{i}"] = True
        st.session_state[f"chain_src_signature_{i}"] = _normalize_signature(prev["dest"].get("display"), prev["dest"].get("coord"))
        st.session_state[f"origin_user_edited_{i}"] = False

col_title, col_reset = st.columns([8, 2])
with col_title:
    st.subheader("Saisie des segments")
with col_reset:
    st.button("Reinitialiser", type="secondary", use_container_width=True, help="Effacer tous les segments", key="btn_reset_segments", on_click=reset_segments)

weight_val = st.number_input(
    "Masse brute totale (kg)", min_value=0.001, value=None, step=100.0,
    format="%.3f", key="weight_0", help="Masse de la marchandise avec ses emballages.",
)
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
            o = unified_location_input("origin", i, "Origine", show_airports=(mode == "Aerien"), show_ports=(mode == "Maritime"), show_stations=(mode == "Ferroviaire"))
        with c2:
            st.markdown("**Destination**")
            d = unified_location_input("dest", i, "Destination", show_airports=(mode == "Aerien"), show_ports=(mode == "Maritime"), show_stations=(mode == "Ferroviaire"))

        # Si l'utilisateur modifie l'origine par rapport à la source de chainage, enlever le badge et verrouiller
        if st.session_state.get(f"origin_autofill_{i}", False) and i > 0:
            prev_sig = st.session_state.get(f"chain_src_signature_{i}")
            cur_sig  = _normalize_signature(o["display"], o["coord"])
            if prev_sig and cur_sig != prev_sig:
                st.session_state[f"origin_autofill_{i}"] = False
                st.session_state[f"origin_user_edited_{i}"] = (not _is_location_empty({"display": o["display"], "coord": o["coord"]}))

        st.session_state.segments[i]["weight"] = weight_val

        st.session_state.segments[i]["origin"] = o
        st.session_state.segments[i]["dest"] = d

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
                next_seg["origin"]["country"] = d["country"]
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

        # The main form needs only mass, mode and locations. Overrides remain available per segment.
        profile = None
        source = ""
        carrier_total = specific_factor = volume = None
        additional_emissions = 0.0
        with st.expander("Données du transporteur (facultatif)"):
            distance_km = st.number_input(
                "Distance réelle (km)", min_value=0.001, value=None, format="%.3f",
                key=f"distance_km_{i}",
                help="Laissez vide pour calculer automatiquement la distance à partir des lieux sélectionnés.",
            )
            method = st.selectbox(
                "Donnée d’émissions",
                ("Estimation par profil", "Total du transporteur", "Facteur spécifique"),
                format_func=lambda value: "Estimation automatique" if value == "Estimation par profil" else value,
                key=f"emission_method_{i}",
            )
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
                    help="Utilisez un facteur représentatif couvrant l’amont et le fonctionnement.",
                )
            elif mode == "Ferroviaire":
                profile = st.selectbox(
                    "Traction réelle", [None, *RAIL_FACTORS],
                    format_func=lambda value: value or "Automatique", key=f"rail_traction_{i}",
                )
                volume = st.number_input(
                    "Volume brut (m³)", min_value=0.001, value=None,
                    format="%.3f", key=f"rail_volume_{i}",
                )
            elif mode == "Routier":
                profile = st.selectbox(
                    "Véhicule routier", [None, *ROAD_PROFILES],
                    format_func=lambda value: value or "Automatique (selon le poids)",
                    key=f"road_profile_{i}",
                )
                if profile is None and weight_val is not None:
                    profile = automatic_road_profile(weight_val)
            elif mode == "Maritime":
                profile = st.selectbox("Profil du navire", list(SEA_FACTORS), key=f"sea_profile_{i}")
            if method != "Estimation par profil":
                source = st.text_input("Source / référence du document", key=f"emission_source_{i}").strip()
            if method != "Total du transporteur":
                additional_emissions = st.number_input(
                    "Fuites frigorigènes (kg CO₂e)",
                    min_value=0.0, value=0.0, format="%.3f", key=f"additional_emissions_{i}",
                    help="À saisir si les fuites attribuées à l’envoi sont absentes du facteur.",
                )

        segments_out.append({
            "origin_display": o["display"],
            "destination_display": d["display"],
            "mode": mode,
            "weight": weight_val,
            "coord_o": o["coord"],
            "coord_d": d["coord"],
            "country_o": o["country"],
            "country_d": d["country"],
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

unit = "kg"
st.caption(
    "Calcul segment par segment en kg CO₂e, amont de l’énergie et fonctionnement. "
    f"[Guide ministériel Information GES]({GES_GUIDE_URL}). "
    "Si le transporteur fournit un total pour le segment, reprenez-le tel quel. "
    "Pour un transport combiné, saisissez aussi les pré- et post-acheminements."
)
calculation_inputs = segments_out
if st.button("Calculer les émissions totales", type="primary"):
    st.session_state.pop("calculation_result", None)
    st.session_state.pop("generated_report", None)
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
            st.caption("Carte indicative · itinéraires OpenStreetMap et SNCF Réseau selon le mode.")
        route_paths = []
        for row in mapped_rows:
            category = mode_to_category(row["Mode"])
            color = list(bytes.fromhex(ROUTE_COLORS[category][1:]))
            route_paths.append({
                "path": _map_path(row), "name": f"Segment {row['Segment']} - {row['Mode']}",
                "color": [*color, 230],
            })

        layers = []
        if route_paths:
            layers.append(pdk.Layer(
                "PathLayer", data=route_paths, get_path="path", get_color="color",
                width_scale=1, width_min_pixels=4, pickable=True, wrap_longitude=True,
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
                coordinates = _map_path(r)
                lon_mid, lat_mid = coordinates[len(coordinates) // 2]
                if len(coordinates) == 2:
                    lon_mid = (coordinates[0][0] + coordinates[1][0]) / 2
                    lat_mid = (coordinates[0][1] + coordinates[1][1]) / 2
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

        st.subheader("Exporter")
        dossier_val = (st.text_input(
            "N° de dossier transport", key="dossier_transport",
            help="Obligatoire pour générer le rapport PDF.",
        ) or "").strip()

        df_export = df.drop(columns=["lat_o", "lon_o", "lat_d", "lon_d", "route_coords"]).copy()
        df_export.insert(0, "N° dossier Transport", dossier_val)
        csv = df_export.to_csv(index=False).encode("utf-8")
        safe_suffix = "".join(c if c.isalnum() or c in "-_" else "_" for c in dossier_val)
        safe_suffix = f"_{safe_suffix}" if safe_suffix else ""
        filename_csv = f"resultats_co2_multimodal{safe_suffix}.csv"
        filename_pdf = f"rapport_co2_multimodal{safe_suffix}.pdf"

        pdf_base_choice = st.selectbox("Fond de carte du PDF", [PDF_BASEMAP_MATCH, PDF_BASEMAP_LOCAL])
        pdf_style = map_style_label if pdf_base_choice == PDF_BASEMAP_MATCH else "Carto Positron (clair)"
        carto_key = read_secret("CARTO_BASEMAPS_KEY") if pdf_base_choice == PDF_BASEMAP_MATCH else ""
        report_inputs = (dossier_val, pdf_style, pdf_base_choice)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Télécharger le détail (CSV)", data=csv, file_name=filename_csv, mime="text/csv", on_click="ignore")
        with c2:
            if st.button("Générer le rapport PDF", disabled=not dossier_val):
                st.session_state.pop("generated_report", None)
                try:
                    with st.spinner("Génération du rapport PDF…"):
                        pdf_buffer = generate_pdf_report(
                            dossier_val, total_distance, total_emissions, unit, rows,
                            map_style=pdf_style, carto_key=carto_key,
                        )
                    st.session_state.generated_report = (report_inputs, pdf_buffer.getvalue())
                except Exception as exc:
                    st.error(f"Erreur lors de la génération du PDF : {exc}")
            generated = st.session_state.get("generated_report")
            if dossier_val and generated and generated[0] == report_inputs:
                st.download_button(
                    "Télécharger le rapport PDF", data=generated[1], file_name=filename_pdf,
                    mime="application/pdf", on_click="ignore",
                )
        if not dossier_val:
            st.caption("Renseignez le n° de dossier transport pour générer le PDF.")
