
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
st.set_page_config(page_title="Calculateur CO2 multimodal - NILEY EXPERTS", page_icon="🌍", layout="centered")

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

DEFAULT_EMISSION_FACTORS = {
    "Routier": 0.100,
    "Aerien": 0.500,
    "Maritime": 0.015,
    "Ferroviaire": 0.030,
}

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
        mode = list(DEFAULT_EMISSION_FACTORS.keys())[0]
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
    if prev_dest.get("display") and prev_dest.get("coord"):
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

def remove_last_segment():
    if "segments" not in st.session_state or not st.session_state.segments:
        st.session_state.segments = [_default_segment()]
        return
    if len(st.session_state.segments) <= 1:
        st.info("Au moins un segment doit rester. Reinitialisation du segment.")
        st.session_state.segments = [_default_segment()]
        for k in list(st.session_state.keys()):
            if any(pat in k for pat in [
                "origin_query_0","dest_query_0","origin_choice_0","dest_choice_0",
                "origin_coord_0","dest_coord_0","origin_display_0","dest_display_0",
                "origin_iata_0","dest_iata_0","origin_unlo_0","dest_unlo_0","mode_select_0",
                "origin_autofill_0","origin_user_edited_0","chain_src_signature_0"
            ]):
                st.session_state.pop(k, None)
        return

    last_idx = len(st.session_state.segments) - 1
    st.session_state.segments.pop()
    for k in list(st.session_state.keys()):
        if any(pat in k for pat in [
            f"origin_query_{last_idx}", f"dest_query_{last_idx}",
            f"origin_choice_{last_idx}", f"dest_choice_{last_idx}",
            f"origin_coord_{last_idx}",  f"dest_coord_{last_idx}",
            f"origin_display_{last_idx}",f"dest_display_{last_idx}",
            f"origin_iata_{last_idx}",   f"dest_iata_{last_idx}",
            f"origin_unlo_{last_idx}",   f"dest_unlo_{last_idx}",
            f"mode_select_{last_idx}",
            f"origin_autofill_{last_idx}",
            f"origin_user_edited_{last_idx}",
            f"chain_src_signature_{last_idx}",
        ]):
            st.session_state.pop(k, None)

def reset_segments():
    try:
        st.session_state.segments = [_default_segment()]
        for i in range(MAX_SEGMENTS):
            for prefix in ["origin", "dest"]:
                for suffix in ["query", "choice", "coord", "display", "iata", "unlo", "autofill", "user_edited"]:
                    st.session_state.pop(f"{prefix}_{suffix}_{i}", None)
            st.session_state.pop(f"mode_select_{i}", None)
            st.session_state.pop(f"weight_{i}", None)
            st.session_state.pop(f"chain_src_signature_{i}", None)
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
    s = _normalize_no_diacritics(mode_str)
    if "routier" in s or "road" in s or "truck" in s:
        return "routier"
    if "aerien" in s or "air" in s or "plane" in s:
        return "aerien"
    if any(k in s for k in ["maritime","mer","bateau","sea","ship"]):
        return "maritime"
    if "ferroviaire" in s or "rail" in s or "train" in s:
        return "ferroviaire"
    return "routier"

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
    disp = (loc.get("display") or "").strip()
    coord = loc.get("coord")
    return (not disp) or (not coord)

# --------------------------
# Rapport PDF mono-page
# --------------------------
@st.cache_data(show_spinner=True, ttl=24*60*60)
\1
    df, dossier_val, total_distance, total_emissions, unit, rows,
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

    title_para = Paragraph("RAPPORT D'EMPREINTE Co2 TRANSPORT", title_style)
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
        ["N° dossier Transport:", dossier_val, "Distance totale:", f"{total_distance:.1f} km"],
        ["Date du rapport:", datetime.now().strftime("%d/%m/%Y %H:%M"), "Emissions totales:", f"{total_emissions:.2f} kg CO2e"],
        ["Nombre de segments:", str(len(rows)), "Emissions moyennes:", f"{(total_emissions/total_distance):.3f} kg CO2e/km" if total_distance>0 else "N/A"],
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
    try:
        all_lats = [r["lat_o"] for r in rows] + [r["lat_d"] for r in rows]
        all_lons = [r["lon_o"] for r in rows] + [r["lon_d"] for r in rows]
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
            for r in rows:
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
            for r in rows:
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

    heading_para = Paragraph("Detail des segments", heading_style)
    hw, hh = heading_para.wrap(AVAIL_W, AVAIL_H)
    heading_para.drawOn(c, M, y - hh)
    y = y - hh - 0.10*cm

    headers = ["Seg.", "Origine", "Destination", "Mode", "Dist.\n(km)", f"Poids\n({unit})", "Facteur\n(kg CO2e/t.km)", "Emissions\n(kg CO2e)"]
    col_widths = [1.2*cm, 4.8*cm, 4.8*cm, 3.0*cm, 1.8*cm, 1.8*cm, 2.2*cm, 2.2*cm]

    # Correctif : version sécurisée de la cellule Paragraph
    def _p_cell_dyn(s, fs):
        stl = ParagraphStyle('CellWrapDyn', parent=styles['Normal'], fontSize=fs, leading=max(8, fs+2), alignment=0)
        txt = "" if s is None else str(s)
        txt = xml_escape(txt, entities={"'": "&apos;", '"': "&quot;"})
        return Paragraph(txt, stl)

    data_rows = []
    for _, row in df.iterrows():
        data_rows.append([
            str(row["Segment"]),
            _p_cell_dyn(row["Origine"], 8),
            _p_cell_dyn(row["Destination"], 8),
            _p_cell_dyn(row["Mode"], 8),
            f"{row['Distance (km)']:.1f}",
            f"{row[f'Poids ({unit})']}",
            f"{row['Facteur (kg CO2e/t.km)']:.3f}",
            f"{row['Emissions (kg CO2e)']:.2f}",
        ])

    total_row = ["TOTAL", "", "", "", f"{total_distance:.1f}", "", "", f"{total_emissions:.2f}"]

    def build_table(font_size, rows_limit=None, show_notice=False, hidden_count=0):
        body = [headers]
        body += (data_rows if rows_limit is None else data_rows[:rows_limit])
        body.append(total_row)
        if show_notice and hidden_count > 0:
            notice = Paragraph(
                f"... {hidden_count} ligne(s) non affichee(s) pour tenir sur 1 page ...",
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
if not API_KEY:
    st.error("Clee API OpenCage absente. Ajoutez OPENCAGE_KEY.")
    st.stop()
geocoder = OpenCageGeocode(API_KEY)

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
    is_iata_mode = bool(re.fullmatch(r"[A-Za-z]{3}", q_raw))
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

    return {"coord": coord, "display": display, "iata": sel_iata, "unlocode": sel_unlo, "query": query_val, "choice": sel}

# --------------------------
# UI
# --------------------------
st.image(LOGO_URL, width=620)
st.markdown("### Calculateur d'empreinte CO2 multimodal")

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
            mode_options = ["Routier", "Maritime", "Ferroviaire", "Aerien"]
            current_mode = st.session_state.segments[i].get("mode", mode_options[0])
            if current_mode not in mode_options:
                current_mode = mode_options[0]
            mode = st.selectbox("Mode de transport", options=mode_options, index=mode_options.index(current_mode), key=f"mode_select_{i}")
            st.session_state.segments[i]["mode"] = mode

        c1, c2 = st.columns(2)
        with c1:
            if st.session_state.get(f"origin_autofill_{i}", False):
                st.markdown("**Origine** (repris du segment precedent)", unsafe_allow_html=True)
            else:
                st.markdown("**Origine**")
            o = unified_location_input("origin", i, "Origine", show_airports=("aerien" in _normalize_no_diacritics(mode)))
        with c2:
    # Ajout du cache intelligent et bouton reset
    if st.button("Réinitialiser le PDF", type="secondary", help="Force la régénération du PDF"):
        st.cache_data.clear()
        st.session_state.pop("pdf_bytes", None)
        st.success("Cache PDF réinitialisé. Cliquez à nouveau pour générer.")

    # Génération avec cache
    try:
        with st.spinner("Préparation du PDF..."):
            pdf_buffer = generate_pdf_report(
                df=df,
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
            st.download_button("Télécharger le rapport PDF", data=pdf_buffer.getvalue(), file_name=filename_pdf, mime="application/pdf")
    except Exception as e:
        st.error(f"Erreur lors de la génération du PDF : {e}")
        import traceback; st.code(traceback.format_exc())
            except Exception as e:
                st.error(f"Erreur lors de la generation du PDF : {e}")
                import traceback; st.code(traceback.format_exc())
    else:
        st.info("Aucun segment valide n'a ete calcule. Verifiez les entrees.")
