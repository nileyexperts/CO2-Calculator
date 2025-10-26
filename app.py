# -*- coding: utf-8 -*-
# Calculateur CO2 multimodal - NILEY EXPERTS
# Version : Sélecteur de mode par logos (cases exclusives) + Logos sur la carte PDF
# + Fond WEB #DFEDF5 + Contours #BB9357 + Natural Earth (Cartopy) + Auth + Export CSV/PDF
# + Paramètres/Apparence/Fond PDF MASQUÉS (valeurs par défaut)
# + Gestion avancée des segments : ajouter/insérer/dupliquer/supprimer
# + Fond de carte PDF amélioré (thèmes internes : voyager/minimal/terrain) + sélecteur masquable

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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import io
import tempfile
import numpy as np

# -- Cache Cartopy (Natural Earth)
os.environ.setdefault("CARTOPY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "cartopy_cache"))

# =========================
# Paramètres globaux & Config page
# =========================
st.set_page_config(
    page_title="Calculateur CO2 multimodal - NILEY EXPERTS",
    page_icon="🌍",
    layout="centered"
)

# Fond de l'app WEB : #DFEDF5 (main + sidebar)
st.markdown(
    """
    <style>
    .stApp { background-color: #DFEDF5; }
    </style>
    """,
    unsafe_allow_html=True
)

# -- Constantes visuelles/ressources
LOGO_URL = "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/NILEY-EXPERTS-logo-removebg-preview.png"

# =========================
# Apparence PDF (défauts & panneau masquable)
# =========================
# Thème par défaut pour la carte PDF : "voyager" | "minimal" | "terrain"
# 👉 "Beau rendu détaillé" = "terrain"
PDF_THEME_DEFAULT = "terrain"
# Niveau de détail Natural Earth par défaut : "110m" | "50m" | "10m"
# 👉 "Beau rendu détaillé" = "50m" (détaillé, bon compromis perf/qualité)
NE_SCALE_DEFAULT = "50m"
# Afficher/masquer le panneau UI d'apparence PDF (mettez True pour l'afficher)
SHOW_PDF_APPEARANCE_PANEL = False

# Helpers boîtes visuelles
def open_box(title: str = ""):
    if title:
        st.markdown(f"##### {title}\n", unsafe_allow_html=True)
    else:
        st.markdown("\n", unsafe_allow_html=True)

def close_box():
    st.markdown("\n", unsafe_allow_html=True)

DEFAULT_EMISSION_FACTORS = {
    "Routier": 0.100,
    "Aerien": 0.500,
    "Maritime": 0.015,
    "Ferroviaire": 0.030,
}
MAX_SEGMENTS = 50

# =========================
# Entête simple avec logo
# =========================
st.image(LOGO_URL, width=620)
st.markdown("### Calculateur d'empreinte Co2 multimodal")

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
        raise ValueError("Aucune route retournée par OSRM")
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

# Métadonnées UI pour les modes (label + icône)
MODE_META = [
    {"id": "Routier", "label": "Routier", "emoji": "🚚", "icon": ICON_URLS["routier"]},
    {"id": "Maritime", "label": "Maritime", "emoji": "🚢", "icon": ICON_URLS["maritime"]},
    {"id": "Ferroviaire", "label": "Ferroviaire", "emoji": "🚆", "icon": ICON_URLS["ferroviaire"]},
    {"id": "Aerien", "label": "Aérien", "emoji": "✈️", "icon": ICON_URLS["aerien"]},
]

def midpoint_on_path(route_coords, lon_o, lat_o, lon_d, lat_d):
    if route_coords and isinstance(route_coords, list) and len(route_coords) >= 2:
        idx = len(route_coords) // 2
        pt = route_coords[idx]
        return [float(pt[0]), float(pt[1])]
    return [(lon_o + lon_d) / 2.0, (lat_o + lat_d) / 2.0]

# -------- Emprise & ratio pour carte PDF --------
def _compute_extent_and_ratio(all_lats, all_lons, margin_ratio=0.12, min_span_deg=1e-3):
    if not all_lats or not all_lons:
        return (-10, 30, 30, 60)  # Europe Ouest par défaut
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    span_lat = max(max_lat - min_lat, min_span_deg)
    span_lon = max(max_lon - min_lon, min_span_deg)
    min_lat -= span_lat * margin_ratio; max_lat += span_lat * margin_ratio
    min_lon -= span_lon * margin_ratio; max_lon += span_lon * margin_ratio
    return (min_lon, max_lon, min_lat, max_lat)

def fit_extent_to_aspect(min_lon, max_lon, min_lat, max_lat, target_aspect_w_over_h):
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
    # clamp
    min_lon = max(-180.0, min_lon); max_lon = min(180.0, max_lon)
    min_lat = max(-90.0, min_lat); max_lat = min(90.0, max_lat)
    return (min_lon, max_lon, min_lat, max_lat)

# ------ Helper : logos sur carte PDF ------
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
# =========================
# Génération du PDF (fond de carte enrichi)
# =========================
def generate_pdf_report(
    df, dossier_val, total_distance, total_emissions, unit, rows,
    pdf_basemap_mode='auto',  # 'auto' | 'simple' | 'naturalearth'
    ne_scale='110m',
    pdf_theme='voyager',       # 'voyager' | 'minimal' | 'terrain'
    pdf_icon_size_px=28
):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        rightMargin=1*cm, leftMargin=1*cm, topMargin=1*cm, bottomMargin=1*cm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                 fontSize=14, textColor=colors.HexColor('#1f4788'),
                                 spaceAfter=6, alignment=1)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'],
                                   fontSize=11, textColor=colors.HexColor('#2c5aa0'),
                                   spaceAfter=4, spaceBefore=6)
    normal_style = styles['Normal']; normal_style.fontSize = 8
    cell_style = ParagraphStyle('CellSmall', parent=styles['Normal'],
                                fontSize=7, leading=8, alignment=TA_LEFT)

    def two_line_place(text: str, max_line=28):
        if not text:
            return ""
        s = text.strip()
        if len(s) <= max_line:
            return s
        preferred_seps = [' - ', ' - ', ' - ', ' / ', ', ']
        cut_idx = -1
        for sep in preferred_seps:
            i = s.rfind(sep, 0, max_line + 1)
            if i > cut_idx:
                cut_idx = i + len(sep)
        if cut_idx <= 0:
            cut_idx = s.rfind(' ', 0, max_line + 1)
        if cut_idx <= 0:
            cut_idx = max_line
        line1 = s[:cut_idx].rstrip(); line2 = s[cut_idx:].lstrip()
        if len(line2) > max_line:
            line2 = line2[:max_line - 3].rstrip() + '...'
        def esc(t):
            return (t.replace('&', '&').replace('<', '<').replace('>', '>'))
        return f"{esc(line1)}\n{esc(line2)}"

    story = []

    # Logo PDF (robuste : HTTP + fallback local)
    logo = None
    try:
        resp = requests.get(LOGO_URL, timeout=10)
        if resp.ok:
            logo_img = PILImage.open(io.BytesIO(resp.content))
            logo_buffer = io.BytesIO()
            logo_img.save(logo_buffer, format='PNG')
            logo_buffer.seek(0)
            logo = RLImage(logo_buffer, width=3*cm, height=1.5*cm)
    except Exception:
        # Fallback local si disponible
        try:
            with open("assets/NILEY-EXPERTS-logo-removebg-preview.png", "rb") as f:
                logo = RLImage(io.BytesIO(f.read()), width=3*cm, height=1.5*cm)
        except Exception:
            logo = None
    if logo:
        story.append(logo)

    story.append(Paragraph("RAPPORT D'EMPREINTE CARBONE MULTIMODAL", title_style))
    story.append(Spacer(1, 0.2*cm))

    info_summary_data = [
        ["N° dossier Transport:", dossier_val, "Distance totale:", f"{total_distance:.1f} km"],
        ["Date du rapport:", datetime.now().strftime("%d/%m/%Y %H:%M"), "Emissions totales:", f"{total_emissions:.2f} kg CO2"],
        ["Nombre de segments:", str(len(rows)), "Emissions moyennes:", f"{(total_emissions/total_distance):.3f} kg CO2/km" if total_distance > 0 else "N/A"],
    ]
    info_summary_table = Table(info_summary_data, colWidths=[4.5*cm, 5.5*cm, 4.5*cm, 5.5*cm])
    info_summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f7')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#fff4e6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(info_summary_table)
    story.append(Spacer(1, 0.3*cm))

    # Carte PDF
    try:
        use_cartopy = (pdf_basemap_mode in ('auto', 'naturalearth'))
        if pdf_basemap_mode == 'simple':
            use_cartopy = False

        all_lats = [r["lat_o"] for r in rows] + [r["lat_d"] for r in rows]
        all_lons = [r["lon_o"] for r in rows] + [r["lon_d"] for r in rows]

        min_lon, max_lon, min_lat, max_lat = _compute_extent_and_ratio(all_lats, all_lons)
        target_width_cm = 20.0; target_height_cm = 7.5; dpi = 150
        fig_w_in = target_width_cm / 2.54; fig_h_in = target_height_cm / 2.54

        min_lon, max_lon, min_lat, max_lat = fit_extent_to_aspect(
            min_lon, max_lon, min_lat, max_lat,
            target_aspect_w_over_h=(target_width_cm / target_height_cm)
        )

        map_buffer = None

        if use_cartopy:
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature
                from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

                # ---------- Styles par thème ----------
                if pdf_theme == 'minimal':
                    colors_cfg = {
                        'ocean':    '#F5F7FA',
                        'land':     '#FAFAF8',
                        'lakes_fc': '#F5F7FA',
                        'lakes_ec': '#D9DEE7',
                        'coast':    '#B5BBC6',
                        'borders0': '#C3C8D2',
                        'borders1': '#E0E5EC',
                        'rivers':   '#D0D6E2',
                        'grid':     '#E6EAF0',
                    }
                    widths = {'coast':0.3, 'b0':0.3, 'b1':0.25, 'rivers':0.3, 'grid':0.35}
                    grid_labels = True
                elif pdf_theme == 'terrain':
                    colors_cfg = {
                        'ocean':    '#E8F2FF',
                        'land':     '#F2EFE9',
                        'lakes_fc': '#E8F2FF',
                        'lakes_ec': '#9FC3EB',
                        'coast':    '#556270',
                        'borders0': '#6C7A89',
                        'borders1': '#A0AABA',
                        'rivers':   '#6DA6E2',
                        'grid':     '#CBD5E3',
                    }
                    widths = {'coast':0.5, 'b0':0.6, 'b1':0.4, 'rivers':0.6, 'grid':0.5}
                    grid_labels = True
                else:  # 'voyager' (défaut)
                    colors_cfg = {
                        'ocean':    '#EAF4FF',
                        'land':     '#F7F5F2',
                        'lakes_fc': '#EAF4FF',
                        'lakes_ec': '#B3D4F5',
                        'coast':    '#818892',
                        'borders0': '#8F98A3',
                        'borders1': '#B3BAC4',
                        'rivers':   '#9ABFEA',
                        'grid':     '#DDE3EA',
                    }
                    widths = {'coast':0.4, 'b0':0.5, 'b1':0.35, 'rivers':0.45, 'grid':0.4}
                    grid_labels = False  # labels discrets (off par défaut)

                fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
                ax = plt.axes(projection=ccrs.PlateCarree())

                # Fonds
                ax.add_feature(cfeature.OCEAN.with_scale(ne_scale), facecolor=colors_cfg['ocean'], edgecolor='none', zorder=0)
                ax.add_feature(cfeature.LAND.with_scale(ne_scale),  facecolor=colors_cfg['land'],  edgecolor='none', zorder=0)
                ax.add_feature(cfeature.LAKES.with_scale(ne_scale), facecolor=colors_cfg['lakes_fc'],
                                edgecolor=colors_cfg['lakes_ec'], linewidth=0.3, zorder=1)
                # Côtes & frontières
                ax.add_feature(cfeature.COASTLINE.with_scale(ne_scale), edgecolor=colors_cfg['coast'],
                                linewidth=widths['coast'], zorder=2)
                ax.add_feature(cfeature.BORDERS.with_scale(ne_scale),   edgecolor=colors_cfg['borders0'],
                                linewidth=widths['b0'], zorder=2)
                # Frontières admin-1 (si dispo aux échelles fines)
                try:
                    admin1 = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', ne_scale,
                                                          edgecolor=colors_cfg['borders1'], facecolor='none')
                    ax.add_feature(admin1, linewidth=widths['b1'], zorder=2)
                except Exception:
                    pass
                # Rivières
                try:
                    ax.add_feature(cfeature.RIVERS.with_scale(ne_scale), edgecolor=colors_cfg['rivers'],
                                   facecolor='none', linewidth=widths['rivers'], zorder=2)
                except Exception:
                    pass

                ax.set_extent((min_lon, max_lon, min_lat, max_lat), crs=ccrs.PlateCarree())

                # Graticule
                gl = ax.gridlines(draw_labels=grid_labels, linewidth=widths['grid'], color=colors_cfg['grid'],
                                  alpha=1.0, linestyle='--', zorder=1)
                if grid_labels:
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlabel_style = {'size': 6, 'color': '#7A808A'}
                    gl.ylabel_style = {'size': 6, 'color': '#7A808A'}
                    gl.xformatter = LONGITUDE_FORMATTER
                    gl.yformatter = LATITUDE_FORMATTER

                mode_colors = {"routier": "#0066CC", "aerien": "#CC0000", "maritime": "#009900", "ferroviaire": "#9900CC"}
                for r in rows:
                    cat = mode_to_category(r["Mode"]); color = mode_colors.get(cat, "#666666")
                    ax.plot([r["lon_o"], r["lon_d"]], [r["lat_o"], r["lat_d"]],
                            color=color, linewidth=2.0, alpha=0.85, transform=ccrs.PlateCarree(), zorder=3)
                    ax.scatter([r["lon_o"]], [r["lat_o"]], s=22, c="#0A84FF", edgecolors='white', linewidths=0.8,
                               transform=ccrs.PlateCarree(), zorder=4)
                    ax.scatter([r["lon_d"]], [r["lat_d"]], s=22, c="#FF3B30", edgecolors='white', linewidths=0.8,
                               transform=ccrs.PlateCarree(), zorder=4)
                    mid_lon = (r["lon_o"] + r["lon_d"]) / 2; mid_lat = (r["lat_o"] + r["lat_d"]) / 2
                    _pdf_add_mode_icon(ax, mid_lon, mid_lat, cat, pdf_icon_size_px, transform=ccrs.PlateCarree())

                ax.set_title("")
                map_buffer = io.BytesIO()
                plt.tight_layout()
                plt.savefig(map_buffer, format='png', dpi=dpi, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                plt.close(fig)
                map_buffer.seek(0)

            except Exception:
                use_cartopy = False

        if not use_cartopy:
            fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
            ax.set_facecolor('#F7F8FA')
            for spine in ax.spines.values():
                spine.set_edgecolor('#D0D4DA'); spine.set_linewidth(0.8)
            ax.set_xlim(min_lon, max_lon); ax.set_ylim(min_lat, max_lat)

            def _nice_step(span_deg):
                for step in (1, 2, 5, 10, 20, 30, 45, 60):
                    if span_deg / step <= 12:
                        return step
                return 90

            lat_span = max_lat - min_lat; lon_span = max_lon - min_lon
            lat_step = _nice_step(lat_span); lon_step = _nice_step(lon_span)
            lats = np.arange(math.floor(min_lat / lat_step) * lat_step, math.ceil(max_lat / lat_step) * lat_step + 1e-9, lat_step)
            lons = np.arange(math.floor(min_lon / lon_step) * lon_step, math.ceil(max_lon / lon_step) * lon_step + 1e-9, lon_step)
            for y in lats:
                ax.plot([min_lon, max_lon], [y, y], color='#E6E9EF', linewidth=0.6, zorder=0)
            for x in lons:
                ax.plot([x, x], [min_lat, max_lat], color='#E6E9EF', linewidth=0.6, zorder=0)

            mode_colors = {"routier": "#0066CC", "aerien": "#CC0000", "maritime": "#009900", "ferroviaire": "#9900CC"}
            for r in rows:
                cat = mode_to_category(r["Mode"]); color = mode_colors.get(cat, "#666666")
                ax.plot([r["lon_o"], r["lon_d"]], [r["lat_o"], r["lat_d"]],
                        color=color, linewidth=2.0, alpha=0.85, zorder=2)
                ax.scatter(r["lon_o"], r["lat_o"], s=22, c="#0A84FF", edgecolors='white', linewidths=0.8, zorder=3)
                ax.scatter(r["lon_d"], r["lat_d"], s=22, c="#FF3B30", edgecolors='white', linewidths=0.8, zorder=3)
                mid_lon = (r["lon_o"] + r["lon_d"]) / 2; mid_lat = (r["lat_o"] + r["lat_d"]) / 2
                _pdf_add_mode_icon(ax, mid_lon, mid_lat, cat, pdf_icon_size_px, transform=None)

            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title("")
            map_buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(map_buffer, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            map_buffer.seek(0)

        map_image = RLImage(map_buffer, width=target_width_cm*cm, height=target_height_cm*cm)
        story.append(map_image); story.append(Spacer(1, 0.3*cm))

    except Exception:
        story.append(Paragraph("_Carte non disponible_", normal_style)); story.append(Spacer(1, 0.2*cm))

    # Tableau détails
    story.append(Paragraph("Détail des segments", heading_style))

    facteur_col = None; emissions_col = None; poids_col = None
    for col in df.columns:
        col_lower = col.lower()
        if "facteur" in col_lower:
            facteur_col = col
        if "mission" in col_lower:
            emissions_col = col
        if "poids" in col_lower:
            poids_col = col
    if not poids_col:
        for col in df.columns:
            if unit in col:
                poids_col = col; break

    table_data = [["Seg.", "Origine", "Destination", "Mode",
                   "Dist.\n(km)", f"Poids\n({unit})",
                   "Facteur\n(kg CO2/t.km)", "Emissions\n(kg CO2)"]]

    for _, row in df.iterrows():
        mode_clean = row["Mode"]
        try:
            facteur_val = f"{row[facteur_col]:.3f}" if facteur_col else "N/A"
        except Exception:
            facteur_val = "N/A"
        try:
            emissions_val = f"{row[emissions_col]:.2f}" if emissions_col else "N/A"
        except Exception:
            emissions_val = "N/A"
        try:
            poids_val = f"{row[poids_col]:.1f}" if poids_col else "N/A"
        except Exception:
            poids_val = "N/A"

        def two_line(text):
            return Paragraph(two_line_place(text, max_line=28), style=cell_style)

        table_data.append([
            str(row["Segment"]),
            two_line(row["Origine"]),
            two_line(row["Destination"]),
            mode_clean,
            f"{row['Distance (km)']:.1f}",
            poids_val,
            facteur_val,
            emissions_val
        ])

    table_data.append(["TOTAL", "", "", "", f"{total_distance:.1f}", "", "", f"{total_emissions:.2f}"])

    col_widths = [1.2*cm, 4.5*cm, 4.5*cm, 3*cm, 1.8*cm, 1.8*cm, 2.2*cm, 2.2*cm]
    detail_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    detail_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('BOTTOMPADDING', (0,0), (-1,0), 4),
        ('TOPPADDING', (0, 0), (-1, 0), 4),

        ('BACKGROUND', (0, 1), (-1, -2), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -2), colors.black),
        ('ALIGN', (0, 1), (0, -1), 'CENTER'),
        ('ALIGN', (4, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 1), (-1, -2), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -2), 7),
        ('VALIGN', (0, 1), (-1, -1), 'MIDDLE'),

        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#fff4e6')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, -1), (-1, -1), 8),

        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#1f4788')),
        ('LINEABOVE', (0, -1), (-1, -1), 2, colors.HexColor('#1f4788')),
        ('BOTTOMPADDING', (0,1), (-1,-1), 3),
        ('TOPPADDING', (0,1), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(detail_table); story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        f"_Document généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} par le Calculateur CO2 multimodal - NILEY EXPERTS_",
        ParagraphStyle('Footer', parent=normal_style, fontSize=7, textColor=colors.grey, alignment=1)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# =========================
# Vérification du mot de passe
# =========================
PASSWORD_KEY = "APP_PASSWORD"
if PASSWORD_KEY not in st.secrets:
    st.error("Mot de passe non configuré. Ajoutez APP_PASSWORD dans .streamlit/secrets.toml.")
    st.stop()

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

# 👉 On n'affiche le formulaire QUE si non authentifié
if not st.session_state.auth_ok:
    st.markdown("## Accès sécurisé")
    with st.form("login_form", clear_on_submit=True):
        # key explicite pour pouvoir nettoyer la valeur
        password_input = st.text_input(
            "Entrez le mot de passe pour accéder à l'application :",
            type="password",
            placeholder="Votre mot de passe",
            key="__pwd__",
        )
        submitted = st.form_submit_button("Valider")

    if submitted:
        if password_input == st.secrets[PASSWORD_KEY]:
            st.session_state.auth_ok = True
            # Nettoyage de la valeur en mémoire de session (bonne pratique)
            try:
                del st.session_state["__pwd__"]
            except KeyError:
                pass
            # Nouveau cycle : le formulaire ne sera plus rendu
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")
    else:
        st.info("Veuillez saisir le mot de passe puis cliquer sur Valider.")
    # Empêche l'accès au reste tant que non authentifié
    st.stop()
else:
    st.success("Accès autorisé. Bienvenue dans l'application !")

# =========================
# API OpenCage
# =========================
API_KEY = read_secret("OPENCAGE_KEY")
if not API_KEY:
    st.error("Clé API OpenCage absente. Ajoutez OPENCAGE_KEY à st.secrets ou à vos variables d'environnement.")
    st.stop()

geocoder = OpenCageGeocode(API_KEY)

# =========================
# Explicatif
# =========================
st.markdown("Ajoutez plusieurs segments (origine → destination), choisissez le mode et le poids. Le mode Routier utilise OSRM (distance réelle + trace).")

# =========================
# Informations générales
# =========================
open_box("Informations générales")
dossier_transport = st.text_input(
    "N° dossier Transport (obligatoire) *",
    value=st.session_state.get("dossier_transport", ""),
    placeholder="ex : TR-2025-001",
    help="Renseignez un identifiant de dossier pour lancer le calcul."
)
st.session_state["dossier_transport"] = (dossier_transport or "").strip()
close_box()

# =========================
# Paramètres MASQUÉS : valeurs par défaut (pas d'UI)
# =========================
default_mode_label = "Envoi unique (même poids sur tous les segments)"
if "weight_mode" not in st.session_state:
    st.session_state["weight_mode"] = default_mode_label
weight_mode = st.session_state["weight_mode"]

factors = {
    "Routier": float(DEFAULT_EMISSION_FACTORS["Routier"]),
    "Aerien": float(DEFAULT_EMISSION_FACTORS["Aerien"]),
    "Maritime": float(DEFAULT_EMISSION_FACTORS["Maritime"]),
    "Ferroviaire": float(DEFAULT_EMISSION_FACTORS["Ferroviaire"]),
}

if "unit" not in st.session_state:
    st.session_state["unit"] = "kg"  # "kg" | "tonnes"
unit = st.session_state["unit"]

if "osrm_base_url" not in st.session_state:
    st.session_state["osrm_base_url"] = "https://router.project-osrm.org"
osrm_base_url = st.session_state["osrm_base_url"]

if "dynamic_radius" not in st.session_state:
    st.session_state["dynamic_radius"] = True
dynamic_radius = st.session_state["dynamic_radius"]
radius_m = 20000 if dynamic_radius else None
radius_px = None if dynamic_radius else 8

if "icon_size_px" not in st.session_state:
    st.session_state["icon_size_px"] = 28
icon_size_px = st.session_state["icon_size_px"]

if "pdf_basemap_choice" not in st.session_state:
    st.session_state["pdf_basemap_choice"] = "Automatique (recommandé)"
pdf_basemap_choice = st.session_state["pdf_basemap_choice"]

# Appliquer défauts "Beau rendu détaillé"
if "ne_scale" not in st.session_state:
    st.session_state["ne_scale"] = NE_SCALE_DEFAULT
if "pdf_theme" not in st.session_state:
    st.session_state["pdf_theme"] = PDF_THEME_DEFAULT

# =========================
# Apparence du rapport (PDF) — Sélecteurs (masquables)
# =========================
if SHOW_PDF_APPEARANCE_PANEL:
    open_box("Apparence du rapport (PDF)")
    st.session_state["pdf_theme"] = st.selectbox(
        "Thème de la carte PDF",
        options=["voyager", "minimal", "terrain"],
        index=["voyager", "minimal", "terrain"].index(st.session_state["pdf_theme"]),
        help="Style graphique du fond Natural Earth utilisé dans le rapport PDF."
    )
    st.session_state["ne_scale"] = st.selectbox(
        "Niveau de détail Natural Earth",
        options=["110m", "50m", "10m"],
        index=["110m", "50m", "10m"].index(st.session_state["ne_scale"]),
        help="110m = rapide • 50m = équilibré • 10m = très détaillé (plus lent)."
    )
    close_box()

ne_scale = st.session_state["ne_scale"]

# =========================
# Saisie des segments
# =========================
def _default_segment(origin_raw="", origin_sel="", dest_raw="", dest_sel="", mode=None, weight=1000.0):
    if mode is None:
        mode = list(DEFAULT_EMISSION_FACTORS.keys())[0]
    return {
        "origin_raw": origin_raw, "origin_sel": origin_sel,
        "dest_raw": dest_raw, "dest_sel": dest_sel,
        "mode": mode, "weight": weight
    }

# Initialisation
if "segments" not in st.session_state or not st.session_state.segments:
    st.session_state.segments = [_default_segment()]

# Auto-lien origine = dest précédent
for i in range(1, len(st.session_state.segments)):
    prev, cur = st.session_state.segments[i-1], st.session_state.segments[i]
    if prev.get("dest_sel") and not cur.get("origin_raw") and not cur.get("origin_sel"):
        cur["origin_raw"] = prev["dest_sel"]; cur["origin_sel"] = prev["dest_sel"]

# Actions segments
def add_segment_end():
    last = st.session_state.segments[-1]
    st.session_state.segments.append(
        _default_segment(
            origin_raw=last.get("dest_sel") or last.get("dest_raw") or "",
            origin_sel=last.get("dest_sel") or "",
            mode=last.get("mode", list(DEFAULT_EMISSION_FACTORS.keys())[0]),
            weight=last.get("weight", 1000.0)
        )
    )
    st.rerun()

def remove_segment_end():
    if len(st.session_state.segments) > 1:
        st.session_state.segments.pop()
    st.rerun()

def insert_after(index: int):
    base = st.session_state.segments[index]
    st.session_state.segments.insert(
        index+1,
        _default_segment(
            origin_raw=base.get("dest_sel") or base.get("dest_raw") or "",
            origin_sel=base.get("dest_sel") or "",
            mode=base.get("mode"),
            weight=base.get("weight", 1000.0)
        )
    )
    st.rerun()

def duplicate_segment(index: int):
    base = st.session_state.segments[index].copy()
    st.session_state.segments.insert(index+1, base)
    st.rerun()

def delete_segment(index: int):
    if len(st.session_state.segments) > 1:
        st.session_state.segments.pop(index)
    st.rerun()

# Exclusivité : callback + rendu "logos + cases"
def _on_mode_check(seg_idx: int, clicked_id: str):
    for m in MODE_META:
        st.session_state[f"mode_chk_{seg_idx}_{m['id']}"] = (m["id"] == clicked_id)
    if "segments" in st.session_state and 0 <= seg_idx < len(st.session_state.segments):
        st.session_state.segments[seg_idx]["mode"] = clicked_id

def select_mode_with_icons(segment_index: int, current_value: str) -> str:
    st.markdown("**Mode de transport**")
    cols = st.columns(len(MODE_META))
    all_ids = [m["id"] for m in MODE_META]
    selected_id = current_value if current_value in all_ids else MODE_META[0]["id"]

    # Init des cases au premier rendu
    if not any(k.startswith(f"mode_chk_{segment_index}_") for k in st.session_state.keys()):
        for m in MODE_META:
            st.session_state[f"mode_chk_{segment_index}_{m['id']}"] = (m["id"] == selected_id)

    # Affichage
    for i, m in enumerate(MODE_META):
        with cols[i]:
            st.image(m["icon"], width=42)
            st.caption(f"{m['emoji']} {m['label']}")
            st.checkbox(
                " ",
                key=f"mode_chk_{segment_index}_{m['id']}",
                help=f"Sélectionner {m['label']}",
                on_change=_on_mode_check,
                args=(segment_index, m["id"]),
            )

    # Valeur active
    for m in MODE_META:
        if st.session_state.get(f"mode_chk_{segment_index}_{m['id']}", False):
            return m["id"]
    _on_mode_check(segment_index, selected_id)
    return selected_id

open_box("Saisie des segments")

# Barre d'actions globale (haut)
a1, a2, _ = st.columns([2,2,6])
with a1:
    st.button("➕ Ajouter un segment (fin)", on_click=add_segment_end, key="btn_add_end_top")
with a2:
    st.button("🗑️ Supprimer le dernier", on_click=remove_segment_end, disabled=(len(st.session_state.segments) <= 1), key="btn_del_end_top")

segments_out = []

for i in range(len(st.session_state.segments)):
    st.markdown(f"##### Segment {i+1}")

    # Actions locales pour ce segment
    b1, b2, b3, _ = st.columns([2,2,2,6])
    with b1:
        st.button("➕ Insérer après", key=f"btn_ins_{i}", on_click=insert_after, args=(i,))
    with b2:
        st.button("📄 Dupliquer", key=f"btn_dup_{i}", on_click=duplicate_segment, args=(i,))
    with b3:
        st.button("🗑️ Supprimer", key=f"btn_del_{i}", on_click=delete_segment, args=(i,), disabled=(len(st.session_state.segments) <= 1))

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

    # Sélecteur de mode : logos + cases exclusives
    mode = select_mode_with_icons(segment_index=i, current_value=st.session_state.segments[i]["mode"])

    # Poids
    if weight_mode == "Poids par segment":
        default_weight = st.session_state.segments[i]["weight"]
        weight_val = st.number_input(
            f"Poids transporté pour le segment {i+1}",
            min_value=0.001,
            value=float(default_weight),
            step=100.0 if unit == "kg" else 0.1,
            key=f"weight_{i}"
        )
    else:
        default_weight = st.session_state.segments[0]["weight"]
        if i == 0:
            weight_val = st.number_input(
                "Poids transporté (appliqué à tous les segments)",
                min_value=0.001,
                value=float(default_weight),
                step=100.0 if unit == "kg" else 0.1,
                key="weight_0"
            )
        else:
            weight_val = st.session_state.get("weight_0", default_weight)

    # Enregistrer
    st.session_state.segments[i] = {
        "origin_raw": origin_raw, "origin_sel": origin_sel,
        "dest_raw": dest_raw, "dest_sel": dest_sel,
        "mode": mode, "weight": weight_val
    }

    segments_out.append({
        "origin": origin_sel or origin_raw or "",
        "destination": dest_sel or dest_raw or "",
        "mode": mode,
        "weight": weight_val
    })

# Barre d'actions globale (bas)
a3, a4, _ = st.columns([2,2,6])
with a3:
    st.button("➕ Ajouter un segment (fin)", on_click=add_segment_end, key="btn_add_end_bot")
with a4:
    st.button("🗑️ Supprimer le dernier", on_click=remove_segment_end, disabled=(len(st.session_state.segments) <= 1), key="btn_del_end_bot")

close_box()

# =========================
# Calcul + Carte
# =========================
def _compute_auto_view(all_lats, all_lons, viewport_px=(900, 600), padding_px=80):
    if not all_lats or not all_lons:
        return pdk.ViewState(latitude=48.8534, longitude=2.3488, zoom=3)
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    mid_lat = (min_lat + max_lat) / 2.0; mid_lon = (min_lon + max_lon) / 2.0
    span_lat = max(1e-6, max_lat - min_lat); span_lon = max(1e-6, max_lon - min_lon)
    span_lon_equiv = span_lon * max(0.1, math.cos(math.radians(mid_lat)))
    world_deg_width = 360.0
    zoom_x = math.log2(world_deg_width / max(1e-6, span_lon_equiv))
    zoom_y = math.log2(180.0 / max(1e-6, span_lat))
    zoom = max(1.0, min(15.0, min(zoom_x, zoom_y)))
    return pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=float(zoom), bearing=0, pitch=0)

can_calculate = bool(st.session_state.get("dossier_transport"))
if not can_calculate:
    st.warning("Veuillez renseigner le N° dossier Transport avant de lancer le calcul.")

if st.button("Calculer l'empreinte carbone totale", disabled=not can_calculate):
    if not st.session_state.get("dossier_transport"):
        st.error("Le N° dossier Transport est obligatoire pour calculer.")
        st.stop()

    rows = []; total_emissions = 0.0; total_distance = 0.0

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
            if "routier" in _normalize_no_diacritics(seg["mode"]):
                try:
                    r = osrm_route(coord1, coord2, st.session_state["osrm_base_url"], overview="full")
                    distance_km = r["distance_km"]; route_coords = r["coords"]
                except Exception as e:
                    st.warning(f"Segment {idx}: OSRM indisponible ({e}). Distance à vol d'oiseau utilisée.")
                    distance_km = compute_distance_km(coord1, coord2)
            else:
                distance_km = compute_distance_km(coord1, coord2)

            weight_tonnes = seg["weight"] if unit == "tonnes" else seg["weight"]/1000.0
            factor = float(factors.get(seg["mode"], DEFAULT_EMISSION_FACTORS.get(seg["mode"], 0.0)))
            emissions = compute_emissions(distance_km, weight_tonnes, factor)

            total_distance += distance_km; total_emissions += emissions

            rows.append({
                "Segment": idx,
                "Origine": seg["origin"],
                "Destination": seg["destination"],
                "Mode": seg["mode"],
                "Distance (km)": round(distance_km, 1),
                f"Poids ({unit})": round(seg["weight"], 3 if unit=="tonnes" else 1),
                "Facteur (kg CO2e/t.km)": factor,
                "Emissions (kg CO2e)": round(emissions, 2),
                "lat_o": coord1[0], "lon_o": coord1[1],
                "lat_d": coord2[0], "lon_d": coord2[1],
                "route_coords": route_coords,
            })

    if rows:
        df = pd.DataFrame(rows)
        st.success(f"{len(rows)} segment(s) calculé(s) • Distance totale : {total_distance:.1f} km • Emissions totales : {total_emissions:.2f} kg CO2e")

        dossier_val = st.session_state.get("dossier_transport", "")
        if dossier_val:
            st.info(f"N° dossier Transport : {dossier_val}")

        st.dataframe(
            df[["Segment", "Origine", "Destination", "Mode", "Distance (km)", f"Poids ({unit})", "Facteur (kg CO2e/t.km)", "Emissions (kg CO2e)"]],
            use_container_width=True
        )

        st.subheader("Carte des segments")

        # -- Préparation des couches (layers) pydeck --
        route_paths = []
        for r in rows:
            if "routier" in _normalize_no_diacritics(r["Mode"]) and r.get("route_coords"):
                route_paths.append({"path": r["route_coords"], "name": f"Segment {r['Segment']} - {r['Mode']}"})

        layers = []

        # Traces OSRM (PathLayer)
        if route_paths:
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=route_paths,
                    get_path="path",
                    get_color=[187, 147, 87, 220],
                    width_scale=1,
                    width_min_pixels=4,
                    pickable=True,
                )
            )

        # Lignes droites pour les autres modes
        straight_lines = []
        for r in rows:
            if not ("routier" in _normalize_no_diacritics(r["Mode"]) and r.get("route_coords")):
                straight_lines.append({
                    "from": [r["lon_o"], r["lat_o"]],
                    "to": [r["lon_d"], r["lat_d"]],
                    "name": f"Segment {r['Segment']} - {r['Mode']}",
                })
        if straight_lines:
            layers.append(
                pdk.Layer(
                    "LineLayer",
                    data=straight_lines,
                    get_source_position="from",
                    get_target_position="to",
                    get_width=3,
                    get_color=[120, 120, 120, 160],
                    pickable=True,
                )
            )

        # Points O/D + étiquettes
        points, labels = [], []
        for r in rows:
            points.append({"position": [r["lon_o"], r["lat_o"]], "name": f"S{r['Segment']} - Origine", "color": [0, 122, 255, 220]})
            labels.append({"position": [r["lon_o"], r["lat_o"]], "text": f"S{r['Segment']} O", "color": [0, 122, 255, 255]})
            points.append({"position": [r["lon_d"], r["lat_d"]], "name": f"S{r['Segment']} - Destination", "color": [220, 66, 66, 220]})
            labels.append({"position": [r["lon_d"], r["lat_d"]], "text": f"S{r['Segment']} D", "color": [220, 66, 66, 255]})

        if points:
            if dynamic_radius:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=points,
                        get_position="position",
                        get_fill_color="color",
                        get_radius=radius_m if radius_m is not None else 20000,
                        radius_min_pixels=2,
                        radius_max_pixels=60,
                        pickable=True,
                        stroked=True,
                        get_line_color=[255, 255, 255],
                        line_width_min_pixels=1,
                    )
                )
            else:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=points,
                        get_position="position",
                        get_fill_color="color",
                        get_radius=radius_px if radius_px is not None else 8,
                        radius_units="pixels",
                        pickable=True,
                        stroked=True,
                        get_line_color=[255, 255, 255],
                        line_width_min_pixels=1,
                    )
                )

        if labels:
            layers.append(
                pdk.Layer(
                    "TextLayer",
                    data=labels,
                    get_position="position",
                    get_text="text",
                    get_color="color",
                    get_size=16,
                    size_units="pixels",
                    get_text_anchor="start",
                    get_alignment_baseline="top",
                    background=False,
                )
            )

        # Icônes (logos) au milieu de chaque segment
        icons = []
        for r in rows:
            cat = mode_to_category(r["Mode"])
            url = ICON_URLS.get(cat)
            if not url:
                continue
            if r.get("route_coords"):
                coords_poly = r["route_coords"]
                mid_index = len(coords_poly) // 2
                lon_mid, lat_mid = coords_poly[mid_index][0], coords_poly[mid_index][1]
            else:
                lon_mid = (r["lon_o"] + r["lon_d"]) / 2.0
                lat_mid = (r["lat_o"] + r["lat_d"]) / 2.0
            icons.append({
                "position": [lon_mid, lat_mid],
                "name": f"S{r['Segment']} - {cat.capitalize()}",
                "icon": {"url": url, "width": 64, "height": 64, "anchorY": 64, "anchorX": 32},
            })

        if icons:
            layers.append(
                pdk.Layer(
                    "IconLayer",
                    data=icons,
                    get_icon="icon",
                    get_position="position",
                    get_size=icon_size_px,
                    size_units="pixels",
                    pickable=True,
                )
            )

        # -- Vue auto pour centrer/zoomer --
        all_lats, all_lons = [], []
        if route_paths and any(d["path"] for d in route_paths):
            all_lats.extend([pt[1] for d in route_paths for pt in d["path"]])
            all_lons.extend([pt[0] for d in route_paths for pt in d["path"]])

        all_lats.extend([r["lat_o"] for r in rows] + [r["lat_d"] for r in rows])
        all_lons.extend([r["lon_o"] for r in rows] + [r["lon_d"] for r in rows])

        view = _compute_auto_view(all_lats, all_lons, viewport_px=(900, 600), padding_px=80)

        # -- Rendu pydeck --
        st.pydeck_chart(
            pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
                initial_view_state=view,
                layers=layers,
                tooltip={"text": "{name}"},
            )
        )

        # Exports
        df_export = df.drop(columns=["lat_o","lon_o","lat_d","lon_d","route_coords"]).copy()
        dossier_val = st.session_state.get("dossier_transport", "")
        df_export.insert(0, "N° dossier Transport", dossier_val)

        csv = df_export.to_csv(index=False).encode("utf-8")
        raw_suffix = dossier_val.strip()
        safe_suffix = "".join(c if (c.isalnum() or c in "-_") else "_" for c in raw_suffix)
        safe_suffix = f"_{safe_suffix}" if safe_suffix else ""
        filename_csv = f"resultats_co2_multimodal{safe_suffix}.csv"
        filename_pdf = f"rapport_co2_multimodal{safe_suffix}.pdf"

        mode_map = {
            "Automatique (recommandé)": "auto",
            "Côtes/continents (Natural Earth)": "naturalearth",
            "Simple (sans côtes)": "simple"
        }
        pdf_basemap_param = mode_map[pdf_basemap_choice]

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Télécharger le détail (CSV)", data=csv, file_name=filename_csv, mime="text/csv")
        with col2:
            try:
                with st.spinner("Génération du PDF en cours..."):
                    pdf_buffer = generate_pdf_report(
                        df=df,
                        dossier_val=dossier_val,
                        total_distance=total_distance,
                        total_emissions=total_emissions,
                        unit=unit,
                        rows=rows,
                        pdf_basemap_mode=pdf_basemap_param,
                        ne_scale=st.session_state.get("ne_scale", NE_SCALE_DEFAULT),
                        pdf_theme=st.session_state.get("pdf_theme", PDF_THEME_DEFAULT),
                        pdf_icon_size_px=icon_size_px
                    )
                st.download_button("Télécharger le rapport PDF", data=pdf_buffer, file_name=filename_pdf, mime="application/pdf")
            except Exception as e:
                st.error(f"Erreur lors de la génération du PDF : {e}")
                import traceback; st.code(traceback.format_exc())
    else:
        st.info("Aucun segment valide n'a été calculé. Vérifiez les entrées ou les sélections.")
