# -*- coding: utf-8 -*-
# Calculateur CO2 multimodal - NILEY EXPERTS
# App : Champ unique (Adresse/Ville/Pays ou IATA) • IATA robuste • Cadres arrondis
# • Sélecteur Mode à droite • Bouton discret "➕ Ajouter un segment"
# • Carte interactive pydeck améliorée (fond au choix + focus segment)
# • Rapport PDF : carte détaillée via Cartopy (Stamen Terrain / OSM) avec fallback Natural Earth
# • Auth + Export CSV/PDF

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
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage

# Matplotlib pour rendu carte PDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# -- Cache Cartopy (Natural Earth)
os.environ.setdefault("CARTOPY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "cartopy_cache"))

# =========================
# Paramètres globaux & Config page
# =========================
st.set_page_config(page_title="Calculateur CO2 multimodal - NILEY EXPERTS", page_icon="🌍", layout="centered")

# Fond & style
st.markdown(
    """
    """,
    unsafe_allow_html=True
)

LOGO_URL = "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/NILEY-EXPERTS-logo-removebg-preview.png"

# =========================
# Apparence PDF
# =========================
PDF_THEME_DEFAULT = "terrain"  # "voyager" \
# "minimal" \
# "terrain"
NE_SCALE_DEFAULT = "50m"  # "110m" \
# "50m" \
# "10m"

# Options de basemap PDF
PDF_BASEMAP_LABELS = [
    "Auto (Stamen → OSM → NaturalEarth)",
    "Stamen Terrain (détaillé, internet)",
    "OSM (détaillé, internet)",
    "Natural Earth (vectoriel, offline possible)"
]
PDF_BASEMAP_MODES = {
    "Auto (Stamen → OSM → NaturalEarth)": "auto",
    "Stamen Terrain (détaillé, internet)": "stamen",
    "OSM (détaillé, internet)": "osm",
    "Natural Earth (vectoriel, offline possible)": "naturalearth",
}

# =========================
# Constantes / Factors
# =========================
DEFAULT_EMISSION_FACTORS = {
    "Routier": 0.100,
    "Aerien": 0.500,
    "Maritime": 0.015,
    "Ferroviaire": 0.030,
}
MAX_SEGMENTS = 50
ICON_URLS = {
    "routier": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/truck.png",
    "aerien": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/plane.png",
    "maritime":"https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/ship.png",
    "ferroviaire":"https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/train.png",
}
def reset_segments():
    """
    Réinitialise la saisie des segments et les champs transitoires associés,
    puis relance l'application.
    """
    try:
        # Réinitialiser la liste de segments à 1 segment par défaut
        st.session_state.segments = [ _default_segment() ]

        # Nettoyer quelques états de saisie globaux
        for k in list(st.session_state.keys()):
            # Effacer les champs d'input des lieux (query, choice, coord, display, iata, unlo)
            if any(pat in k for pat in ["origin_query_", "dest_query_", 
                                        "origin_choice_", "dest_choice_",
                                        "origin_coord_", "dest_coord_", 
                                        "origin_display_", "dest_display_",
                                        "origin_iata_", "dest_iata_",
                                        "origin_unlo_", "dest_unlo_"]):
                st.session_state.pop(k, None)

        # Réinitialiser le poids global du premier segment s'il existe
        st.session_state.pop("weight_0", None)

        # (optionnel) garder le n° de dossier; si vous voulez aussi le vider, décommentez :
        # st.session_state.pop("dossier_transport", None)

        # (optionnel) reset du focus carte
        # st.session_state.pop("Focus segment", None)

    finally:
        st.rerun()
        
# =========================
# Entête simple avec logo
# =========================
st.image(LOGO_URL, width=620)
st.markdown("### Calculateur d'empreinte Co2 multimodal")

# =========================
# Utilitaires
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
        # petite pause anti-rate limiting
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
    if any(k in s for k in ["maritime","mer","bateau","sea","ship"]):
        return "maritime"
    if "ferroviaire" in s or "rail" in s or "train" in s:
        return "ferroviaire"
    return "routier"

# =========================
# PDF : helpers pour l'emprise
# =========================
def _compute_extent_from_coords(all_lats, all_lons, margin_ratio=0.12, min_span_deg=1e-3):
    if not all_lats or not all_lons:
        return (-10, 30, 30, 60)  # Ouest Europe par défaut
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
    # clamp
    min_lon = max(-180.0, min_lon); max_lon = min(180.0, max_lon)
    min_lat = max(-90.0, min_lat); max_lat = min(90.0, max_lat)
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

# =========================
# Génération du PDF (fond détaillé quand possible)
# =========================
def generate_pdf_report(
    df, dossier_val, total_distance, total_emissions, unit, rows,
    pdf_basemap_choice_label, ne_scale='50m', pdf_theme='terrain', pdf_icon_size_px=28
):
    """ pdf_basemap_choice_label ∈ PDF_BASEMAP_LABELS (auto, stamen, osm, naturalearth) """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        rightMargin=1*cm, leftMargin=1*cm, topMargin=1*cm, bottomMargin=1*cm
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=14,
                                 textColor=colors.HexColor('#1f4788'), spaceAfter=6, alignment=1)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=11,
                                   textColor=colors.HexColor('#2c5aa0'), spaceAfter=4, spaceBefore=6)
    normal_style = styles['Normal']; normal_style.fontSize = 8

    story = []

    # Logo
    logo = None
    try:
        resp = requests.get(LOGO_URL, timeout=10)
        if resp.ok:
            pil_logo = PILImage.open(io.BytesIO(resp.content))
            lb = io.BytesIO(); pil_logo.save(lb, format='PNG'); lb.seek(0)
            logo = RLImage(lb, width=3*cm, height=1.5*cm)
    except Exception:
        pass
    if logo:
        story.append(logo)

    story.append(Paragraph("RAPPORT D'EMPREINTE CARBONE MULTIMODAL", title_style))
    story.append(Spacer(1, 0.2*cm))

    # Résumé
    info_summary_data = [
        ["N° dossier Transport:", dossier_val, "Distance totale:", f"{total_distance:.1f} km"],
        ["Date du rapport:", datetime.now().strftime("%d/%m/%Y %H:%M"), "Emissions totales:", f"{total_emissions:.2f} kg CO2e"],
        ["Nombre de segments:", str(len(rows)), "Emissions moyennes:",
         f"{(total_emissions/total_distance):.3f} kg CO2e/km" if total_distance>0 else "N/A"],
    ]
    info_summary_table = Table(info_summary_data, colWidths=[4.5*cm, 5.5*cm, 4.5*cm, 5.5*cm])
    info_summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f7')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#fff4e6')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))
    story.append(info_summary_table)
    story.append(Spacer(1, 0.3*cm))

    # ----------- Carte PDF détaillée -----------
    try:
        all_lats = [r["lat_o"] for r in rows] + [r["lat_d"] for r in rows]
        all_lons = [r["lon_o"] for r in rows] + [r["lon_d"] for r in rows]
        min_lon, max_lon, min_lat, max_lat = _compute_extent_from_coords(all_lats, all_lons)

        target_width_cm, target_height_cm = 20.0, 7.5
        dpi = 150
        fig_w_in = target_width_cm / 2.54
        fig_h_in = target_height_cm / 2.54
        min_lon, max_lon, min_lat, max_lat = _fit_extent_to_aspect(
            min_lon, max_lon, min_lat, max_lat, target_aspect_w_over_h=target_width_cm/target_height_cm
        )

        map_buffer = None
        use_cartopy = True
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.io.img_tiles import Stamen, OSM
        except Exception:
            use_cartopy = False
        mode_label = pdf_basemap_choice_label
        mode = PDF_BASEMAP_MODES.get(mode_label, "auto")

        def _draw_overlays(ax, ccrs, theme='terrain'):
            # éléments vectoriels supplémentaires (côtes, frontières, rivières) par-dessus un raster
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

        if use_cartopy:
            # Choix fond raster si possible
            fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
            ax = None
            raster_ok = False
            try:
                if mode in ("auto","stamen"):
                    tiler = Stamen('terrain-background')  # détaillé + relief
                    ax = plt.axes(projection=tiler.crs)
                    ax.set_extent((min_lon, max_lon, min_lat, max_lat), crs=ccrs.PlateCarree())
                    # niveau de zoom empirique
                    zoom = 6 if (max_lon-min_lon < 5 and max_lat-min_lat < 3) else (5 if (max_lon-min_lon < 15) else 4)
                    ax.add_image(tiler, zoom)
                    raster_ok = True
                if not raster_ok and mode in ("auto","osm"):
                    tiler = OSM()
                    ax = plt.axes(projection=tiler.crs)
                    ax.set_extent((min_lon, max_lon, min_lat, max_lat), crs=ccrs.PlateCarree())
                    zoom = 6 if (max_lon-min_lon < 5 and max_lat-min_lat < 3) else (5 if (max_lon-min_lon < 15) else 4)
                    ax.add_image(tiler, zoom)
                    raster_ok = True
            except Exception:
                raster_ok = False

            if not raster_ok:
                # Fallback Natural Earth vectoriel (offline possible)
                ax = plt.axes(projection=ccrs.PlateCarree())
                # Thèmes rapides
                colors_cfg = {
                    'ocean':'#EAF4FF','land':'#F7F5F2','lakes_fc':'#EAF4FF','lakes_ec':'#B3D4F5',
                    'coast':'#818892','borders0':'#8F98A3'
                }
                ax.add_feature(cfeature.OCEAN.with_scale(ne_scale), facecolor=colors_cfg['ocean'], edgecolor='none', zorder=0)
                ax.add_feature(cfeature.LAND.with_scale(ne_scale), facecolor=colors_cfg['land'], edgecolor='none', zorder=0)
                ax.add_feature(cfeature.LAKES.with_scale(ne_scale), facecolor=colors_cfg['lakes_fc'], edgecolor=colors_cfg['lakes_ec'], linewidth=0.3, zorder=1)
                ax.add_feature(cfeature.COASTLINE.with_scale(ne_scale), edgecolor=colors_cfg['coast'], linewidth=0.4, zorder=2)
                ax.add_feature(cfeature.BORDERS.with_scale(ne_scale), edgecolor=colors_cfg['borders0'], linewidth=0.5, zorder=2)
                ax.set_extent((min_lon, max_lon, min_lat, max_lat), crs=ccrs.PlateCarree())
            else:
                _draw_overlays(ax, ccrs)

            # Tracé des segments
            mode_colors = {"routier":"#0066CC","aerien":"#CC0000","maritime":"#009900","ferroviaire":"#9900CC"}
            for r in rows:
                cat = mode_to_category(r["Mode"]); color = mode_colors.get(cat, "#666666")
                ax.plot([r["lon_o"], r["lon_d"]], [r["lat_o"], r["lat_d"]],
                        color=color, linewidth=2.0, alpha=0.9,
                        transform=ccrs.PlateCarree(), zorder=3)
                ax.scatter([r["lon_o"]], [r["lat_o"]], s=22, c="#0A84FF", edgecolors='white', linewidths=0.8,
                           transform=ccrs.PlateCarree(), zorder=4)
                ax.scatter([r["lon_d"]], [r["lat_d"]], s=22, c="#FF3B30", edgecolors='white', linewidths=0.8,
                           transform=ccrs.PlateCarree(), zorder=4)
                mid_lon = (r["lon_o"] + r["lon_d"]) / 2.0
                mid_lat = (r["lat_o"] + r["lat_d"]) / 2.0
                _pdf_add_mode_icon(ax, mid_lon, mid_lat, cat, pdf_icon_size_px, transform=ccrs.PlateCarree())

            # Sauvegarde image
            map_buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(map_buffer, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            map_buffer.seek(0)
        else:
            # Cartopy indisponible → fallback simple (grille)
            fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
            ax.set_facecolor('#F7F8FA')
            ax.set_xlim(min_lon, max_lon); ax.set_ylim(min_lat, max_lat)
            # grille soft
            lat_step = (max_lat-min_lat)/6.0 if (max_lat-min_lat)>0 else 1
            lon_step = (max_lon-min_lon)/8.0 if (max_lon-min_lon)>0 else 1
            for y in np.arange(min_lat, max_lat+1e-9, lat_step):
                ax.plot([min_lon, max_lon],[y,y], color='#E6E9EF', lw=0.6)
            for x in np.arange(min_lon, max_lon+1e-9, lon_step):
                ax.plot([x,x],[min_lat,max_lat], color='#E6E9EF', lw=0.6)
            mode_colors = {"routier":"#0066CC","aerien":"#CC0000","maritime":"#009900","ferroviaire":"#9900CC"}
            for r in rows:
                cat = mode_to_category(r["Mode"]); color = mode_colors.get(cat, "#666666")
                ax.plot([r["lon_o"], r["lon_d"]], [r["lat_o"], r["lat_d"]], color=color, lw=2.0, alpha=0.9)
                ax.scatter([r["lon_o"]],[r["lat_o"]], s=22, c="#0A84FF", edgecolor='white', lw=0.8)
                ax.scatter([r["lon_d"]],[r["lat_d"]], s=22, c="#FF3B30", edgecolor='white', lw=0.8)
            map_buffer = io.BytesIO()
            plt.tight_layout(); plt.savefig(map_buffer, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(); map_buffer.seek(0)

        map_image = RLImage(map_buffer, width=target_width_cm*cm, height=target_height_cm*cm)
        story.append(map_image)
        story.append(Spacer(1, 0.3*cm))
    except Exception as e:
        story.append(Paragraph(f"_Carte non disponible : {e}_", normal_style))
        story.append(Spacer(1, 0.2*cm))

    # Détail des segments
    story.append(Paragraph("Détail des segments", heading_style))
    table_data = [["Seg.", "Origine", "Destination", "Mode", "Dist.\n(km)", f"Poids\n({unit})",
                   "Facteur\n(kg CO2e/t.km)", "Emissions\n(kg CO2e)"]]
    for _, row in df.iterrows():
        table_data.append([
            str(row["Segment"]),
            str(row["Origine"]),
            str(row["Destination"]),
            row["Mode"],
            f"{row['Distance (km)']:.1f}",
            f"{row[f'Poids ({unit})']}",
            f"{row['Facteur (kg CO2e/t.km)']:.3f}",
            f"{row['Emissions (kg CO2e)']:.2f}"
        ])
    table_data.append(["TOTAL", "", "", "", f"{total_distance:.1f}", "", "", f"{total_emissions:.2f}"])
    col_widths = [1.2*cm, 4.5*cm, 4.5*cm, 3*cm, 1.8*cm, 1.8*cm, 2.2*cm, 2.2*cm]
    detail_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    detail_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#fff4e6')),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
    ]))
    story.append(detail_table); story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        f"_Document généré le {datetime.now().strftime('%d/%m/%Y %H:%M')} — Calculateur CO2 multimodal - NILEY EXPERTS_",
        ParagraphStyle('Footer', parent=normal_style, fontSize=7, textColor=colors.grey, alignment=1)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# =========================
# Auth
# =========================
PASSWORD_KEY = "APP_PASSWORD"
if PASSWORD_KEY not in st.secrets:
    st.error("Mot de passe non configuré. Ajoutez APP_PASSWORD dans .streamlit/secrets.toml.")
    st.stop()

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.markdown("## Accès sécurisé")
    with st.form("login_form", clear_on_submit=True):
        password_input = st.text_input("Entrez le mot de passe pour accéder à l'application :", type="password",
                                       placeholder="Votre mot de passe", key="__pwd__")
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
# IATA/Aéroports : chargement + recherche
# =========================
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
        st.error(f"Colonnes manquantes dans airport-codes.csv : {', '.join(sorted(missing))}")
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
        base = f"{(r['iata_code'] or '').strip()} — {(r['name'] or 'Sans nom').strip()}"
        extra = " · ".join([p for p in [r['municipality'], r['iso_country']] if p])
        return f"{base} {extra}" if extra else base
    df["label"] = df.apply(_label, axis=1)
    cols = ["iata_code","name","municipality","iso_country","lat","lon","label"]
    if "type" in df.columns:
        cols.append("type")
    return df[cols]

@st.cache_data(show_spinner=False, ttl=24*60*60)
def search_airports(query: str, limit: int = 10) -> pd.DataFrame:
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
    
# =========================
# Ports : chargement + recherche (UN/LOCODE ou équivalent)
# =========================
@st.cache_data(show_spinner=False, ttl=7*24*60*60)
def load_ports_csv(path: str = "ports.csv") -> pd.DataFrame:
    """
    Charge un fichier de ports (lat/lon obligatoires).
    Colonnes tolérées : unlocode|locode|UNLOCODE, name|port_name,
                        country|iso_country|country_code, lat, lon
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.warning(f"Impossible de charger '{path}': {e}")
        return pd.DataFrame(columns=["unlocode", "name", "country", "lat", "lon", "label"])

    # Normalisation des colonnes attendues
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

    required_missing = [k for k,v in {
        "lat": col_lat, "lon": col_lon
    }.items() if v is None]
    if required_missing:
        st.error("Colonnes manquantes dans ports.csv : " + ", ".join(required_missing))
        return pd.DataFrame(columns=["unlocode", "name", "country", "lat", "lon", "label"])

    # Construire le DataFrame cible
    out = pd.DataFrame()
    out["unlocode"] = df[col_unlo] if col_unlo else ""
    out["name"]     = df[col_name] if col_name else ""
    out["country"]  = df[col_ctry] if col_ctry else ""
    out["lat"]      = pd.to_numeric(df[col_lat], errors="coerce")
    out["lon"]      = pd.to_numeric(df[col_lon], errors="coerce")

    out = out.dropna(subset=["lat","lon"]).copy()
    out["unlocode"] = out["unlocode"].astype(str).str.upper().str.strip()
    out["name"]     = out["name"].astype(str).replace({"nan":""}).fillna("").str.strip()
    out["country"]  = out["country"].astype(str).replace({"nan":""}).fillna("").str.strip()

    # Étiquette pour l’UI
    def _label_port(r):
        base = (r["name"] or "Port sans nom").strip()
        extras = " · ".join([p for p in [r["unlocode"], r["country"]] if p])
        return f"{base} {extras}" if extras else base
    out["label"] = out.apply(_label_port, axis=1)

    # Colonnes finales
    return out[["unlocode","name","country","lat","lon","label"]]


@st.cache_data(show_spinner=False, ttl=24*60*60)
def search_ports(query: str, limit: int = 12) -> pd.DataFrame:
    df = load_ports_csv()
    q = (query or "").strip()
    if df.empty:
        return df
    if not q:
        return df.head(limit)

    # Si l'utilisateur tape un UN/LOCODE
    if len(q) in (5,6):  # certains jeux contiennent 6 (avec séparateur)
        res = df[df["unlocode"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
                  .str.startswith(re.sub(r"[^A-Za-z0-9]", "", q.upper()))]
        if not res.empty:
            return res.head(limit)

    # Recherche par nom de port et pays
    ql = q.lower()
    res = df[
        df["name"].str.lower().str.contains(ql, na=False)
        | df["country"].str.lower().str.contains(ql, na=False)
        | df["unlocode"].str.lower().str.contains(ql, na=False)
    ]
    return res.head(limit)
# =========================
# Champ unifié (Adresse/Ville/Pays ou IATA)
# =========================
def unified_location_input(side_key: str, seg_index: int, label_prefix: str,
                           show_airports: bool = True,
                           show_ports: bool = False):
    """
    Text input unique + selectbox de résultats combinés :
    ✈️ aéroports (si show_airports=True)
    ⚓ ports (si show_ports=True)
    puis 📍 OpenCage
    Renvoie dict {coord:(lat,lon)|None, display:str, iata:str, unlocode:str, query:str, choice:str}
    """
    q_key   = f"{side_key}_query_{seg_index}"
    c_key   = f"{side_key}_choice_{seg_index}"
    crd_key = f"{side_key}_coord_{seg_index}"
    disp_key= f"{side_key}_display_{seg_index}"
    iata_key= f"{side_key}_iata_{seg_index}"
    unlo_key= f"{side_key}_unlo_{seg_index}"

    query_val = st.text_input(
        f"{label_prefix} — Adresse / Ville / Pays"
        + (" ou IATA (3 lettres)" if show_airports else "")
        + (" ou UN/LOCODE (5 lettres)" if show_ports else ""),
        value=st.session_state.get(q_key, ""),
        key=q_key
    )

    airports = pd.DataFrame()
    ports    = pd.DataFrame()
    oc_opts  = []

    if query_val:
        # AEROPORTS
        if show_airports:
            airports = search_airports(query_val, limit=10)
        # PORTS
        if show_ports:
            ports = search_ports(query_val, limit=12)

        oc = geocode_cached(query_val, limit=5)
        oc_opts = [r['formatted'] for r in oc] if oc else []

    options = []
    airport_rows = []
    port_rows = []

    # Injecter d'abord les aéroports (si activé)
    if show_airports and not airports.empty:
        for _, r in airports.iterrows():
            label = f"✈️ {r['label']} (IATA {r['iata_code']})"
            options.append(label); airport_rows.append(r)

    # Puis les ports (si activé)
    if show_ports and not ports.empty:
        for _, r in ports.iterrows():
            suffix = f" (UN/LOCODE {r['unlocode']})" if r.get("unlocode") else ""
            label = f"⚓ {r['label']}{suffix}"
            options.append(label); port_rows.append(r)

    # Enfin OpenCage
    if oc_opts:
        options += [f"📍 {o}" for o in oc_opts]

    if not options:
        options = ["— Aucun résultat —"]

    sel = st.selectbox("Résultats", options, index=0, key=c_key)

    coord = None; display = ""; sel_iata = ""; sel_unlo = ""
    if sel != "— Aucun résultat —":
        if sel.startswith("✈️"):
            idx = options.index(sel)
            r = airport_rows[idx] if idx < len(airport_rows) else airports.iloc[0]
            coord = (float(r["lat"]), float(r["lon"]))
            display = r["label"]; sel_iata = r["iata_code"]; sel_unlo = ""
        elif sel.startswith("⚓"):
            # PORT
            # retrouver l'index relatif aux ports
            # (options = [air..., port..., oc...], on recalcule l'offset)
            first_port_idx = 0
            if show_airports and not airports.empty:
                first_port_idx = len(airport_rows)
            idx_global = options.index(sel)
            idx = idx_global - (len(airport_rows))  # position relative dans port_rows
            r = port_rows[idx] if 0 <= idx < len(port_rows) else ports.iloc[0]
            coord = (float(r["lat"]), float(r["lon"]))
            display = r["label"]; sel_unlo = str(r.get("unlocode") or ""); sel_iata = ""
        else:
            formatted = sel[2:].strip() if sel.startswith("📍") else sel
            coord = coords_from_formatted(formatted)
            display = formatted
            sel_iata = ""; sel_unlo = ""

    st.session_state[crd_key]  = coord
    st.session_state[disp_key] = display
    st.session_state[iata_key] = sel_iata
    st.session_state[unlo_key] = sel_unlo

    return {"coord": coord, "display": display, "iata": sel_iata, "unlocode": sel_unlo,
            "query": query_val, "choice": sel}

# =========================
# Saisie des segments (UI)
# =========================
def _default_segment(mode=None, weight=1000.0):
    if mode is None:
        mode = list(DEFAULT_EMISSION_FACTORS.keys())[0]
    return {
        "origin": {"query":"", "display":"", "iata":"", "coord":None},
        "dest": {"query":"", "display":"", "iata":"", "coord":None},
        "mode": mode,
        "weight": weight,
    }

if "segments" not in st.session_state or not st.session_state.segments:
    st.session_state.segments = [_default_segment()]

# auto chaînage O = D précédent si vide
for i in range(1, len(st.session_state.segments)):
    prev, cur = st.session_state.segments[i-1], st.session_state.segments[i]
    if prev.get("dest",{}).get("display") and not cur.get("origin",{}).get("display"):
        cur["origin"]["display"] = prev["dest"]["display"]
        cur["origin"]["coord"] = prev["dest"]["coord"]
        cur["origin"]["iata"] = prev["dest"]["iata"]
        cur["origin"]["query"] = prev["dest"]["display"]

# Titre avec bouton Réinitialiser à droite
col_title, col_reset = st.columns([8, 2])
with col_title:
    st.subheader("Saisie des segments")
with col_reset:
    st.button(
        "↺ Réinitialiser",
        type="secondary",
        use_container_width=True,
        help="Effacer tous les segments saisis et recommencer",
        key="btn_reset_segments",
        on_click=reset_segments
    )
segments_out = []

for i in range(len(st.session_state.segments)):
    with st.container(border=True):
        # En-tête : Titre + Mode à droite
        hl, hr = st.columns([6, 4])
        with hl:
            st.markdown(f"##### Segment {i+1}")
        with hr:
            mode_options = ["Routier", "Maritime", "Ferroviaire", "Aerien"]
            current_mode = st.session_state.segments[i].get("mode", mode_options[0])
            if current_mode not in mode_options:
                current_mode = mode_options[0]
            mode = st.selectbox("Mode de transport", options=mode_options,
                                index=mode_options.index(current_mode), key=f"mode_select_{i}")
            st.session_state.segments[i]["mode"] = mode

        c1, c2 = st.columns(2)

        # IMPORTANT : blocs with bien indentés
        with c1:
            st.markdown("**Origine**")
            # Proposer les aéroports UNIQUEMENT si mode Aérien pour l'Origine
            o = unified_location_input(
                "origin", i, "Origine",
                show_airports=("aerien" in _normalize_no_diacritics(mode))
            )

        with c2:
            st.markdown("**Destination**")
            # Jamais d'aéroports pour la Destination (selon la demande)
            d = unified_location_input(
                "dest", i, "Destination",
                show_airports=False
            )

        # Poids (mode global "envoi unique")
        if "weight_0" not in st.session_state:
            st.session_state["weight_0"] = st.session_state.segments[0]["weight"]
        if i == 0:
            weight_val = st.number_input(
                "Poids transporté (appliqué à tous les segments)",
                min_value=0.001,
                value=float(st.session_state["weight_0"]),
                step=100.0,
                key="weight_0"
            )
        else:
            weight_val = st.session_state["weight_0"]
        st.session_state.segments[i]["weight"] = weight_val

        # Persist & sortie
        st.session_state.segments[i]["origin"] = {
            "query": o["query"], "display": o["display"], "iata": o["iata"], "coord": o["coord"]
        }
        st.session_state.segments[i]["dest"] = {
            "query": d["query"], "display": d["display"], "iata": d["iata"], "coord": d["coord"]
        }
        segments_out.append({
            "origin_display": o["display"],
            "destination_display": d["display"],
            "origin_iata": o["iata"],
            "dest_iata": d["iata"],
            "mode": mode,
            "weight": weight_val,
            "coord_o": o["coord"],
            "coord_d": d["coord"],
            "origin": o["display"],
            "destination": d["display"],
        })

# =========================
# ➕ Bouton discret "Ajouter un segment" (bas)
# =========================
def add_segment_end():
    if len(st.session_state.segments) >= MAX_SEGMENTS:
        st.warning(f"Nombre maximal de segments atteint ({MAX_SEGMENTS}).")
        return
    last = st.session_state.segments[-1]
    new_seg = _default_segment(mode=last.get("mode","Routier"), weight=last.get("weight",1000.0))
    if last.get("dest",{}).get("display"):
        new_seg["origin"] = last["dest"].copy()
    st.session_state.segments.append(new_seg)

with st.container():
    st.markdown('\n', unsafe_allow_html=True)
    if st.button("➕ Ajouter un segment", use_container_width=False, key="btn_add_bottom"):
        add_segment_end(); st.rerun()
    st.markdown('\n', unsafe_allow_html=True)

# =========================
# Carte interactive — contrôles
# =========================
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
    show_icons = st.checkbox("Afficher les icônes mode", value=True)

# =========================
# Calcul + Affichage des résultats
# =========================
st.subheader("Calcul")
dossier_transport = st.text_input("N° dossier Transport (obligatoire) *",
                                  value=st.session_state.get("dossier_transport",""),
                                  placeholder="ex : TR-2025-001")
st.session_state["dossier_transport"] = (dossier_transport or "").strip()
unit = "kg"

factors = {
    "Routier": float(DEFAULT_EMISSION_FACTORS["Routier"]),
    "Aerien": float(DEFAULT_EMISSION_FACTORS["Aerien"]),
    "Maritime": float(DEFAULT_EMISSION_FACTORS["Maritime"]),
    "Ferroviaire": float(DEFAULT_EMISSION_FACTORS["Ferroviaire"]),
}

can_calculate = bool(st.session_state["dossier_transport"])
if not can_calculate:
    st.warning("Veuillez renseigner le N° dossier Transport avant de lancer le calcul.")

if st.button("Calculer l'empreinte carbone totale", disabled=not can_calculate):
    rows = []; total_emissions = 0.0; total_distance = 0.0
    with st.spinner("Calcul en cours..."):
        for idx, seg in enumerate(segments_out, start=1):
            if not seg["origin_display"] or not seg["destination_display"]:
                st.warning(f"Segment {idx} : origine/destination manquante(s).")
                continue
            coord1, coord2 = seg["coord_o"], seg["coord_d"]
            if not coord1 or not coord2:
                st.error(f"Segment {idx} : lieu introuvable ou ambigu.")
                continue

            route_coords = None
            if "routier" in _normalize_no_diacritics(seg["mode"]):
                try:
                    r = osrm_route(coord1, coord2, overview="full")
                    distance_km = r["distance_km"]; route_coords = r["coords"]
                except Exception as e:
                    st.warning(f"Segment {idx}: OSRM indisponible ({e}). Distance à vol d'oiseau utilisée.")
                    distance_km = compute_distance_km(coord1, coord2)
            else:
                distance_km = compute_distance_km(coord1, coord2)

            weight_tonnes = seg["weight"] / 1000.0
            factor = float(factors.get(seg["mode"], 0.0))
            emissions = compute_emissions(distance_km, weight_tonnes, factor)

            total_distance += distance_km; total_emissions += emissions
            rows.append({
                "Segment": idx,
                "Origine": seg["origin_display"],
                "Destination": seg["destination_display"],
                "Mode": seg["mode"],
                "Distance (km)": round(distance_km, 1),
                f"Poids ({unit})": round(seg["weight"], 1),
                "Facteur (kg CO2e/t.km)": factor,
                "Emissions (kg CO2e)": round(emissions, 2),
                "lat_o": coord1[0], "lon_o": coord1[1],
                "lat_d": coord2[0], "lon_d": coord2[1],
                "route_coords": route_coords,
            })

    if rows:
        df = pd.DataFrame(rows)
        st.success(f"{len(rows)} segment(s) • Distance totale : {total_distance:.1f} km • Emissions : {total_emissions:.2f} kg CO2e")

        # Tableau
        st.dataframe(
            df[["Segment","Origine","Destination","Mode","Distance (km)",f"Poids ({unit})","Facteur (kg CO2e/t.km)","Emissions (kg CO2e)"]],
            use_container_width=True
        )

        # Carte interactive pydeck
        # - Routes OSRM (PathLayer)
        route_paths = [
            {"path": r["route_coords"], "name": f"Segment {r['Segment']} - {r['Mode']}"}
            for r in rows
            if ("routier" in _normalize_no_diacritics(r["Mode"]) and r.get("route_coords"))
        ]

        layers = []
        if route_paths:
            layers.append(pdk.Layer(
                "PathLayer",
                data=route_paths,
                get_path="path",
                get_color=[187,147,87,220],
                width_scale=1,
                width_min_pixels=4,
                pickable=True
            ))

        # - Lignes droites autres modes
        straight_lines = []
        for r in rows:
            if not ("routier" in _normalize_no_diacritics(r["Mode"]) and r.get("route_coords")):
                straight_lines.append({
                    "from":[r["lon_o"], r["lat_o"]],
                    "to":[r["lon_d"], r["lat_d"]],
                    "name": f"Segment {r['Segment']} - {r['Mode']}",
                })
        if straight_lines:
            layers.append(pdk.Layer(
                "LineLayer",
                data=straight_lines,
                get_source_position="from",
                get_target_position="to",
                get_width=3,
                get_color=[120,120,120,160],
                pickable=True
            ))

        # - Points O/D
        points, labels = [], []
        for r in rows:
            points += [
                {"position":[r["lon_o"], r["lat_o"]], "name":f"S{r['Segment']} - Origine", "color":[0,122,255,220]},
                {"position":[r["lon_d"], r["lat_d"]], "name":f"S{r['Segment']} - Destination", "color":[220,66,66,220]},
            ]
            labels += [
                {"position":[r["lon_o"], r["lat_o"]], "text":f"S{r['Segment']} O", "color":[0,122,255,255]},
                {"position":[r["lon_d"], r["lat_d"]], "text":f"S{r['Segment']} D", "color":[220,66,66,255]},
            ]
        if points:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=points,
                get_position="position",
                get_fill_color="color",
                get_radius=20000,
                radius_min_pixels=2,
                radius_max_pixels=60,
                pickable=True,
                stroked=True,
                get_line_color=[255,255,255],
                line_width_min_pixels=1
            ))
        if labels:
            layers.append(pdk.Layer(
                "TextLayer",
                data=labels,
                get_position="position",
                get_text="text",
                get_color="color",
                get_size=16,
                size_units="pixels",
                get_text_anchor="start",
                get_alignment_baseline="top",
                background=False
            ))

        # - Icônes au milieu
        if show_icons:
            icons = []
            for r in rows:
                cat = mode_to_category(r["Mode"]); url = ICON_URLS.get(cat)
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
                    "IconLayer",
                    data=icons,
                    get_icon="icon",
                    get_position="position",
                    get_size=28,
                    size_units="pixels",
                    pickable=True
                ))

        # Vue : focus sur segment choisi sinon fit global
        def compute_view_for_segment(r):
            mid_lat = (r["lat_o"] + r["lat_d"]) / 2.0
            mid_lon = (r["lon_o"] + r["lon_d"]) / 2.0
            # zoom heuristique
            span_deg = max(
                abs(r["lat_d"]-r["lat_o"]),
                abs(r["lon_d"]-r["lon_o"])*max(0.3, math.cos(math.radians(mid_lat)))
            )
            zoom = 6 if span_deg < 1.5 else (5 if span_deg < 4 else (4 if span_deg < 10 else 3))
            return pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=zoom)

        if focus_sel != "— Tous —":
            idx = int(focus_sel.split()[-1])
            sel_row = next((r for r in rows if r["Segment"]==idx), None)
            view = compute_view_for_segment(sel_row) if sel_row else pdk.ViewState(latitude=48.85, longitude=2.35, zoom=3)
        else:
            all_lats, all_lons = [], []
            for r in rows:
                all_lats += [r["lat_o"], r["lat_d"]]
                all_lons += [r["lon_o"], r["lon_d"]]
            if route_paths:
                for dct in route_paths:
                    if dct.get("path"):
                        all_lats += [pt[1] for pt in dct["path"]]
                        all_lons += [pt[0] for pt in dct["path"]]
            # fit global
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

        # Légende
        st.caption("**Légende** — O: point bleu / D: point rouge • Tracé routier : brun (OSRM) • Autres : gris • Icône au milieu : mode.")

        # Exports
        df_export = df.drop(columns=["lat_o","lon_o","lat_d","lon_d","route_coords"]).copy()
        dossier_val = st.session_state.get("dossier_transport","")
        df_export.insert(0, "N° dossier Transport", dossier_val)
        csv = df_export.to_csv(index=False).encode("utf-8")
        safe_suffix = "".join(c if (c.isalnum() or c in "-_") else "_" for c in dossier_val.strip())
        safe_suffix = f"_{safe_suffix}" if safe_suffix else ""
        filename_csv = f"resultats_co2_multimodal{safe_suffix}.csv"
        filename_pdf = f"rapport_co2_multimodal{safe_suffix}.pdf"

        # Choix de fond PDF détaillé
        st.subheader("Exporter")
        pdf_base_choice = st.selectbox(
            "Fond de carte du PDF",
            options=PDF_BASEMAP_LABELS,
            index=0,
            help="Stamen/OSM = très détaillé (nécessite internet & Cartopy). Fallback automatique sur Natural Earth."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Télécharger le détail (CSV)", data=csv, file_name=filename_csv, mime="text/csv")
        with c2:
            try:
                with st.spinner("Génération du PDF en cours..."):
                    pdf_buffer = generate_pdf_report(
                        df=df, dossier_val=dossier_val,
                        total_distance=total_distance, total_emissions=total_emissions,
                        unit=unit, rows=rows,
                        pdf_basemap_choice_label=pdf_base_choice,
                        ne_scale=NE_SCALE_DEFAULT, pdf_theme=PDF_THEME_DEFAULT,
                        pdf_icon_size_px=28
                    )
                st.download_button("Télécharger le rapport PDF", data=pdf_buffer, file_name=filename_pdf, mime="application/pdf")
            except Exception as e:
                st.error(f"Erreur lors de la génération du PDF : {e}")
                import traceback; st.code(traceback.format_exc())
    else:
        st.info("Aucun segment valide n'a été calculé. Vérifiez les entrées ou les sélections.")
