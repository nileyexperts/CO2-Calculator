# -*- coding: utf-8 -*-
# Calculateur CO2 multimodal - NILEY EXPERTS
# Version : Champ unique (Adresse/Ville/Pays ou IATA) par côté + IATA corrigé
# + Cadres segments arrondis (#002E49) + Sélecteur de mode à droite du titre
# + Bouton discret «➕ Ajouter un segment» en bas
# + Natural Earth (Cartopy), Carte Web, Export CSV/PDF, Auth

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
import re

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
      .stApp {background-color: #DFEDF5;}
      section[data-testid="stSidebar"] {background-color: #DFEDF5;}
      /* Cadres segments */
      div:where([data-testid="stContainer"])[style*="border: 1px"] {
        border-color: #002E49 !important;
        border-radius: 12px !important;
        margin-bottom: 0.6rem;
      }
      /* Bouton discret en bas */
      .add-seg .stButton>button {
        background-color: white; color: #002E49; border: 1px solid #002E49;
        padding: .3rem .6rem; border-radius: 999px; font-size: 0.9rem;
      }
      .add-seg .stButton>button:hover { background-color: #002E49; color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

LOGO_URL = "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/NILEY-EXPERTS-logo-removebg-preview.png"

# =========================
# Apparence PDF
# =========================
PDF_THEME_DEFAULT = "terrain"  # "voyager" | "minimal" | "terrain"
NE_SCALE_DEFAULT = "50m"       # "110m" | "50m" | "10m"
SHOW_PDF_APPEARANCE_PANEL = False

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
    if "routier" in s or "road" in s or "truck" in s: return "routier"
    if "aerien" in s or "air" in s or "plane" in s:   return "aerien"
    if "maritime" in s or "mer" in s or "bateau" in s or "sea" in s or "ship" in s: return "maritime"
    if "ferroviaire" in s or "rail" in s or "train" in s: return "ferroviaire"
    return "routier"

ICON_URLS = {
    "routier": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/truck.png",
    "aerien": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/plane.png",
    "maritime": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/ship.png",
    "ferroviaire": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/train.png",
}

# =========================
# PDF
# =========================
def generate_pdf_report(
    df, dossier_val, total_distance, total_emissions, unit, rows,
    pdf_basemap_mode='auto', ne_scale='110m', pdf_theme='voyager', pdf_icon_size_px=28
):
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
            logo_img = PILImage.open(io.BytesIO(resp.content))
            lb = io.BytesIO(); logo_img.save(lb, format='PNG'); lb.seek(0)
            logo = RLImage(lb, width=3*cm, height=1.5*cm)
    except Exception:
        pass
    if logo: story.append(logo)
    story.append(Paragraph("RAPPORT D'EMPREINTE CARBONE MULTIMODAL", title_style))
    story.append(Spacer(1, 0.2*cm))

    info_summary_data = [
        ["N° dossier Transport:", dossier_val, "Distance totale:", f"{total_distance:.1f} km"],
        ["Date du rapport:", datetime.now().strftime("%d/%m/%Y %H:%M"), "Emissions totales:", f"{total_emissions:.2f} kg CO2"],
        ["Nombre de segments:", str(len(rows)), "Emissions moyennes:", f"{(total_emissions/total_distance):.3f} kg CO2/km" if total_distance>0 else "N/A"],
    ]
    info_summary_table = Table(info_summary_data, colWidths=[4.5*cm, 5.5*cm, 4.5*cm, 5.5*cm])
    info_summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f7')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#fff4e6')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_summary_table); story.append(Spacer(1, 0.3*cm))

    # Carte (simple fallback si Cartopy indispo)
    try:
        import cartopy.crs as ccrs, cartopy.feature as cfeature
        target_width_cm = 20.0; target_height_cm = 7.5; dpi = 150
        fig_w_in = target_width_cm/2.54; fig_h_in = target_height_cm/2.54
        all_lats = [r["lat_o"] for r in rows]+[r["lat_d"] for r in rows]
        all_lons = [r["lon_o"] for r in rows]+[r["lon_d"] for r in rows]
        if not all_lats or not all_lons:
            min_lon, max_lon, min_lat, max_lat = (-10, 30, 30, 60)
        else:
            min_lon, max_lon = min(all_lons), max(all_lons)
            min_lat, max_lat = min(all_lats), max(all_lats)
        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())
        # thèmes rapides
        ax.add_feature(cfeature.OCEAN.with_scale(ne_scale), facecolor='#EAF4FF', edgecolor='none')
        ax.add_feature(cfeature.LAND.with_scale(ne_scale), facecolor='#F7F5F2', edgecolor='none')
        ax.add_feature(cfeature.LAKES.with_scale(ne_scale), facecolor='#EAF4FF', edgecolor='#B3D4F5', linewidth=0.3)
        ax.add_feature(cfeature.COASTLINE.with_scale(ne_scale), edgecolor='#818892', linewidth=0.4)
        ax.add_feature(cfeature.BORDERS.with_scale(ne_scale), edgecolor='#8F98A3', linewidth=0.5)
        ax.set_extent((min_lon, max_lon, min_lat, max_lat), crs=ccrs.PlateCarree())
        mode_colors = {"routier":"#0066CC","aerien":"#CC0000","maritime":"#009900","ferroviaire":"#9900CC"}
        for r in rows:
            cat = mode_to_category(r["Mode"])
            color = mode_colors.get(cat, "#666666")
            ax.plot([r["lon_o"], r["lon_d"]], [r["lat_o"], r["lat_d"]],
                    color=color, linewidth=2.0, alpha=0.9, transform=ccrs.PlateCarree())
            ax.scatter([r["lon_o"]], [r["lat_o"]], s=22, c="#0A84FF", edgecolors='white', linewidths=0.8, transform=ccrs.PlateCarree())
            ax.scatter([r["lon_d"]], [r["lat_d"]], s=22, c="#FF3B30", edgecolors='white', linewidths=0.8, transform=ccrs.PlateCarree())
            mid_lon = (r["lon_o"]+r["lon_d"])/2; mid_lat = (r["lat_o"]+r["lat_d"])/2
            try:
                url = ICON_URLS.get(cat); resp = requests.get(url, timeout=6)
                pil = PILImage.open(io.BytesIO(resp.content)).convert('RGBA')
                from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                imgbox = OffsetImage(pil, zoom=28/max(1,pil.width))
                ax.add_artist(AnnotationBbox(imgbox, (mid_lon, mid_lat), frameon=False))
            except Exception:
                pass
        buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight');
        plt.close(fig); buf.seek(0)
        story.append(RLImage(buf, width=target_width_cm*cm, height=target_height_cm*cm)); story.append(Spacer(1,0.3*cm))
    except Exception:
        story.append(Paragraph("_Carte non disponible_", normal_style)); story.append(Spacer(1,0.2*cm))

    # Tableau
    table_data = [["Seg.","Origine","Destination","Mode","Dist.\n(km)",f"Poids\n({unit})","Facteur\n(kg CO2e/t.km)","Emissions\n(kg CO2e)"]]
    for _, row in df.iterrows():
        table_data.append([
            str(row["Segment"]), str(row["Origine"]), str(row["Destination"]),
            row["Mode"], f"{row['Distance (km)']:.1f}",
            f"{row[f'Poids ({unit})']}", f"{row['Facteur (kg CO2e/t.km)']:.3f}",
            f"{row['Emissions (kg CO2e)']:.2f}",
        ])
    table_data.append(["TOTAL","","","", f"{total_distance:.1f}","","", f"{total_emissions:.2f}"])
    col_widths = [1.2*cm, 4.5*cm, 4.5*cm, 3*cm, 1.8*cm, 1.8*cm, 2.2*cm, 2.2*cm]
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f4788')), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#fff4e6')),
    ]))
    story.append(t); story.append(Spacer(1,0.3*cm))
    story.append(Paragraph(
        f"_Document généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}_",
        ParagraphStyle('Footer', parent=normal_style, fontSize=7, textColor=colors.grey, alignment=1)
    ))
    doc.build(story); buffer.seek(0); return buffer
# =========================
# Auth
# =========================
PASSWORD_KEY = "APP_PASSWORD"
if PASSWORD_KEY not in st.secrets:
    st.error("Mot de passe non configuré. Ajoutez APP_PASSWORD dans .streamlit/secrets.toml.")
    st.stop()
if "auth_ok" not in st.session_state: st.session_state.auth_ok = False
if not st.session_state.auth_ok:
    st.markdown("## Accès sécurisé")
    with st.form("login_form", clear_on_submit=True):
        password_input = st.text_input("Entrez le mot de passe pour accéder à l'application :", type="password", placeholder="Votre mot de passe", key="__pwd__")
        submitted = st.form_submit_button("Valider")
        if submitted:
            if password_input == st.secrets[PASSWORD_KEY]:
                st.session_state.auth_ok = True
                try: del st.session_state["__pwd__"]
                except KeyError: pass
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
    if not required.issubset(df.columns):
        st.error(f"Colonnes manquantes dans airport-codes.csv : {', '.join(sorted(required - set(df.columns)))}")
        return pd.DataFrame(columns=["iata_code","name","municipality","iso_country","lat","lon","label","type"])
    df = df[df["iata_code"].astype(str).str.len()==3].copy()
    df["iata_code"] = df["iata_code"].astype(str).str.upper()
    if "type" in df.columns:
        df = df[df["type"].isin(["large_airport","medium_airport"])].copy()
    coord_series = df["coordinates"].astype(str).str.replace('"','').str.strip()
    parts = coord_series.str.split(",", n=1, expand=True)
    if parts.shape[1] < 2: parts = pd.DataFrame({0:coord_series, 1:None})
    df["lat"] = pd.to_numeric(parts[0].astype(str).str.strip(), errors="coerce")
    df["lon"] = pd.to_numeric(parts[1].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["lat","lon"]).copy()
    for col in ["municipality","iso_country","name"]:
        if col not in df.columns: df[col] = ""
    for col in ["name","municipality","iso_country"]:
        df[col] = df[col].astype(str).replace({"nan":""}).fillna("").str.strip()
    def _label(r):
        base = f"{(r['iata_code'] or '').strip()} — {(r['name'] or 'Sans nom').strip()}"
        extra = " · ".join([p for p in [r['municipality'], r['iso_country']] if p])
        return f"{base} {extra}" if extra else base
    df["label"] = df.apply(_label, axis=1)
    cols = ["iata_code","name","municipality","iso_country","lat","lon","label"]
    if "type" in df.columns: cols.append("type")
    return df[cols]

@st.cache_data(show_spinner=False, ttl=24*60*60)
def airport_by_iata(code: str):
    if not code: return None
    code = code.strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", code): return None
    df = load_airports_iata(); m = df[df["iata_code"]==code]
    return None if m.empty else m.iloc[0].to_dict()

@st.cache_data(show_spinner=False, ttl=24*60*60)
def search_airports(query: str, limit: int = 10) -> pd.DataFrame:
    df = load_airports_iata()
    q = (query or "").strip()
    if df.empty: return df
    if not q: return df.head(limit)
    if len(q) <= 3:
        res = df[df["iata_code"].str.startswith(q.upper())]
        if res.empty:
            res = df[df["name"].str.lower().str.contains(q.lower()) | df["municipality"].str.lower().str.contains(q.lower())]
    else:
        res = df[df["name"].str.lower().str.contains(q.lower()) | df["municipality"].str.lower().str.contains(q.lower())]
    return res.head(limit)

def is_iata_3(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{3}", (s or "").strip()))

# =========================
# Saisie unifiée (Adresse OU Aéroport) – logique commune
# =========================
def unified_location_input(side_key: str, seg_index: int, label_prefix: str):
    """
    Affiche :
      - un text_input unique (adresse/ville/pays ou IATA)
      - une selectbox de résultats combinés : ✈️ aéroports puis 📍 OpenCage
    Stocke dans st.session_state:
      - f"{side_key}_query_i" : saisie libre
      - f"{side_key}_choice_i" : libellé choisi (str, avec préfixe ✈️/📍)
      - f"{side_key}_coord_i"  : tuple (lat, lon)
      - f"{side_key}_display_i": texte lisible pour tableaux/PDF
      - f"{side_key}_iata_i"   : code IATA si aéroport, sinon ""
    """
    q_key = f"{side_key}_query_{seg_index}"
    c_key = f"{side_key}_choice_{seg_index}"
    crd_key = f"{side_key}_coord_{seg_index}"
    disp_key = f"{side_key}_display_{seg_index}"
    iata_key = f"{side_key}_iata_{seg_index}"

    # Saisie
    query_val = st.text_input(
        f"{label_prefix} — Adresse / Ville / Pays ou IATA (3 lettres)",
        value=st.session_state.get(q_key, ""),
        key=q_key
    )

    airports = pd.DataFrame()
    oc_opts = []
    if query_val:
        # Résultats aéroports d'abord
        airports = search_airports(query_val, limit=10)
        # Suggestions OpenCage
        oc = geocode_cached(query_val, limit=5)
        oc_opts = [r['formatted'] for r in oc] if oc else []

    # Build options list: ✈️ + 📍
    options = []
    airport_rows = []
    if not airports.empty:
        for _, r in airports.iterrows():
            label = f"✈️ {r['label']} (IATA {r['iata_code']})"
            options.append(label); airport_rows.append(r)
    if oc_opts:
        options += [f"📍 {o}" for o in oc_opts]

    if not options:
        options = ["— Aucun résultat —"]
    # Sélection
    sel = st.selectbox("Résultats", options, index=0, key=c_key)
    coord = None; display = ""; sel_iata = ""
    if sel != "— Aucun résultat —":
        if sel.startswith("✈️"):
            # Trouver ligne correspondante
            idx = options.index(sel)
            r = airport_rows[idx] if idx < len(airport_rows) else airports.iloc[0]
            coord = (float(r["lat"]), float(r["lon"]))
            display = r["label"]
            sel_iata = r["iata_code"]
        else:
            # OpenCage: enlever "📍 "
            formatted = sel[2:].strip() if sel.startswith("📍") else sel
            coord = coords_from_formatted(formatted)
            display = formatted

    # Persistance
    st.session_state[crd_key] = coord
    st.session_state[disp_key] = display
    st.session_state[iata_key] = sel_iata

    # Pour compat aval, renvoyer dict
    return {
        "coord": coord,
        "display": display,
        "iata": sel_iata,
        "query": query_val,
        "choice": sel,
    }

# =========================
# Saisie des segments
# =========================
def _default_segment(mode=None, weight=1000.0):
    if mode is None:
        mode = list(DEFAULT_EMISSION_FACTORS.keys())[0]
    return {
        # champs unifiés
        "origin": {"query":"", "display":"", "iata":"", "coord":None},
        "dest":   {"query":"", "display":"", "iata":"", "coord":None},
        "mode": mode,
        "weight": weight,
    }

if "segments" not in st.session_state or not st.session_state.segments:
    st.session_state.segments = [_default_segment()]

# Chaînage auto O = D précédent si vide
for i in range(1, len(st.session_state.segments)):
    prev, cur = st.session_state.segments[i-1], st.session_state.segments[i]
    if prev.get("dest",{}).get("display") and not cur.get("origin",{}).get("display"):
        cur["origin"]["display"] = prev["dest"]["display"]
        cur["origin"]["coord"] = prev["dest"]["coord"]
        cur["origin"]["iata"] = prev["dest"]["iata"]
        cur["origin"]["query"] = prev["dest"]["display"]

open_box("Saisie des segments")

# Sélection / Affichage segments
segments_out = []
for i in range(len(st.session_state.segments)):
    with st.container(border=True):
        # Titre + mode (à droite)
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

        # Entrées unifiées O / D
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Origine**")
            o = unified_location_input("origin", i, "Origine")
        with c2:
            st.markdown("**Destination**")
            d = unified_location_input("dest", i, "Destination")

        # Persister pour ce segment (utile si rerun)
        st.session_state.segments[i]["origin"] = {
            "query": o["query"], "display": o["display"], "iata": o["iata"], "coord": o["coord"]
        }
        st.session_state.segments[i]["dest"] = {
            "query": d["query"], "display": d["display"], "iata": d["iata"], "coord": d["coord"]
        }

        # Poids (mode global « envoi unique »)
        if "weight_0" not in st.session_state:
            st.session_state["weight_0"] = st.session_state.segments[0]["weight"]
        if i == 0:
            weight_val = st.number_input(
                "Poids transporté (appliqué à tous les segments)",
                min_value=0.001, value=float(st.session_state["weight_0"]),
                step=100.0, key="weight_0"
            )
        else:
            weight_val = st.session_state["weight_0"]
        st.session_state.segments[i]["weight"] = weight_val

        # Sortie
        segments_out.append({
            "origin_display": o["display"], "destination_display": d["display"],
            "origin_iata": o["iata"], "dest_iata": d["iata"],
            "mode": mode, "weight": weight_val,
            "coord_o": o["coord"], "coord_d": d["coord"],
            "origin": o["display"], "destination": d["display"],
        })

close_box()

# =========================
# ➕ Bouton discret "Ajouter un segment" (bas)
# =========================
def add_segment_end():
    if len(st.session_state.segments) >= MAX_SEGMENTS:
        st.warning(f"Nombre maximal de segments atteint ({MAX_SEGMENTS}).")
        return
    new_seg = _default_segment(mode=st.session_state.segments[-1].get("mode","Routier"),
                               weight=st.session_state.segments[-1].get("weight",1000.0))
    # Chaînage O = dernière D si dispo
    last = st.session_state.segments[-1]
    if last.get("dest",{}).get("display"):
        new_seg["origin"] = last["dest"].copy()
    st.session_state.segments.append(new_seg)

with st.container():
    st.markdown('<div class="add-seg">', unsafe_allow_html=True)
    if st.button("➕ Ajouter un segment", use_container_width=False, key="btn_add_bottom"):
        add_segment_end(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

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
    zoom_x = math.log2(360.0 / max(1e-6, span_lon_equiv))
    zoom_y = math.log2(180.0 / max(1e-6, span_lat))
    zoom = max(1.0, min(15.0, min(zoom_x, zoom_y)))
    return pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=float(zoom), bearing=0, pitch=0)

st.markdown("---")
st.markdown("#### Calcul")

dossier_transport = st.text_input("N° dossier Transport (obligatoire) *",
                                  value=st.session_state.get("dossier_transport",""),
                                  placeholder="ex : TR-2025-001")
st.session_state["dossier_transport"] = (dossier_transport or "").strip()
unit = "kg"   # on conserve kg (UI masquée)
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
            coord1 = seg["coord_o"]; coord2 = seg["coord_d"]
            if not coord1 or not coord2:
                st.error(f"Segment {idx} : lieu introuvable ou ambigu.")
                continue

            route_coords = None
            if "routier" in _normalize_no_diacritics(seg["mode"]):
                try:
                    r = osrm_route(coord1, coord2, "https://router.project-osrm.org", overview="full")
                    distance_km = r["distance_km"]; route_coords = r["coords"]
                except Exception as e:
                    st.warning(f"Segment {idx}: OSRM indisponible ({e}). Distance à vol d'oiseau utilisée.")
                    distance_km = compute_distance_km(coord1, coord2)
            else:
                distance_km = compute_distance_km(coord1, coord2)

            weight_tonnes = seg["weight"]/1000.0
            factor = float(factors.get(seg["mode"], 0.0))
            emissions = compute_emissions(distance_km, weight_tonnes, factor)

            total_distance += distance_km; total_emissions += emissions
            rows.append({
                "Segment": idx,
                "Origine": seg["origin_display"], "Destination": seg["destination_display"],
                "Mode": seg["mode"], "Distance (km)": round(distance_km, 1),
                f"Poids ({unit})": round(seg["weight"], 1),
                "Facteur (kg CO2e/t.km)": factor,
                "Emissions (kg CO2e)": round(emissions, 2),
                "lat_o": coord1[0], "lon_o": coord1[1],
                "lat_d": coord2[0], "lon_d": coord2[1],
                "route_coords": route_coords,
            })

    if rows:
        df = pd.DataFrame(rows)
        st.success(f"{len(rows)} segment(s) calculé(s) • Distance totale : {total_distance:.1f} km • Emissions totales : {total_emissions:.2f} kg CO2e")

        st.dataframe(
            df[["Segment","Origine","Destination","Mode","Distance (km)",f"Poids ({unit})","Facteur (kg CO2e/t.km)","Emissions (kg CO2e)"]],
            use_container_width=True
        )

        # Carte
        st.subheader("Carte des segments")
        route_paths = []
        for r in rows:
            if "routier" in _normalize_no_diacritics(r["Mode"]) and r.get("route_coords"):
                route_paths.append({"path": r["route_coords"], "name": f"Segment {r['Segment']} - {r['Mode']}"})
        layers = []
        if route_paths:
            layers.append(pdk.Layer("PathLayer", data=route_paths, get_path="path",
                                    get_color=[187,147,87,220], width_scale=1, width_min_pixels=4, pickable=True))
        straight_lines = []
        for r in rows:
            if not ("routier" in _normalize_no_diacritics(r["Mode"]) and r.get("route_coords")):
                straight_lines.append({"from":[r["lon_o"],r["lat_o"]],"to":[r["lon_d"],r["lat_d"]],"name":f"S{r['Segment']} - {r['Mode']}"})
        if straight_lines:
            layers.append(pdk.Layer("LineLayer", data=straight_lines, get_source_position="from", get_target_position="to",
                                    get_width=3, get_color=[120,120,120,160], pickable=True))
        points, labels = [], []
        for r in rows:
            points += [
                {"position":[r["lon_o"],r["lat_o"]], "name":f"S{r['Segment']} - Origine", "color":[0,122,255,220]},
                {"position":[r["lon_d"],r["lat_d"]], "name":f"S{r['Segment']} - Destination", "color":[220,66,66,220]},
            ]
            labels += [
                {"position":[r["lon_o"],r["lat_o"]], "text":f"S{r['Segment']} O", "color":[0,122,255,255]},
                {"position":[r["lon_d"],r["lat_d"]], "text":f"S{r['Segment']} D", "color":[220,66,66,255]},
            ]
        if points:
            layers.append(pdk.Layer("ScatterplotLayer", data=points, get_position="position", get_fill_color="color",
                                    get_radius=20000, radius_min_pixels=2, radius_max_pixels=60, pickable=True,
                                    stroked=True, get_line_color=[255,255,255], line_width_min_pixels=1))
        if labels:
            layers.append(pdk.Layer("TextLayer", data=labels, get_position="position", get_text="text",
                                    get_color="color", get_size=16, size_units="pixels", get_text_anchor="start",
                                    get_alignment_baseline="top", background=False))
        # Icons milieu
        icons = []
        for r in rows:
            cat = mode_to_category(r["Mode"]); url = ICON_URLS.get(cat)
            if not url: continue
            if r.get("route_coords"):
                coords_poly = r["route_coords"]; mid = coords_poly[len(coords_poly)//2]
                lon_mid, lat_mid = mid[0], mid[1]
            else:
                lon_mid = (r["lon_o"] + r["lon_d"])/2.0; lat_mid = (r["lat_o"] + r["lat_d"])/2.0
            icons.append({"position":[lon_mid,lat_mid], "name":f"S{r['Segment']} - {cat.capitalize()}",
                          "icon":{"url":url, "width":64, "height":64, "anchorY":64, "anchorX":32}})
        if icons:
            layers.append(pdk.Layer("IconLayer", data=icons, get_icon="icon", get_position="position",
                                    get_size=28, size_units="pixels", pickable=True))
        all_lats = [pt["position"][1] for pt in points] if points else []
        all_lons = [pt["position"][0] for pt in points] if points else []
        view = _compute_auto_view(all_lats, all_lons)
        st.pydeck_chart(pdk.Deck(map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
                                 initial_view_state=view, layers=layers, tooltip={"text": "{name}"}))

        # Exports
        df_export = df.drop(columns=["lat_o","lon_o","lat_d","lon_d","route_coords"]).copy()
        dossier_val = st.session_state.get("dossier_transport","")
        df_export.insert(0, "N° dossier Transport", dossier_val)
        csv = df_export.to_csv(index=False).encode("utf-8")
        safe_suffix = "".join(c if (c.isalnum() or c in "-_") else "_" for c in dossier_val.strip())
        safe_suffix = f"_{safe_suffix}" if safe_suffix else ""
        filename_csv = f"resultats_co2_multimodal{safe_suffix}.csv"
        filename_pdf = f"rapport_co2_multimodal{safe_suffix}.pdf"

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Télécharger le détail (CSV)", data=csv, file_name=filename_csv, mime="text/csv")
        with col2:
            try:
                with st.spinner("Génération du PDF en cours..."):
                    pdf_buffer = generate_pdf_report(
                        df=df, dossier_val=dossier_val, total_distance=total_distance, total_emissions=total_emissions,
                        unit=unit, rows=rows, pdf_basemap_mode='auto', ne_scale=NE_SCALE_DEFAULT, pdf_theme=PDF_THEME_DEFAULT,
                        pdf_icon_size_px=28
                    )
                st.download_button("Télécharger le rapport PDF", data=pdf_buffer, file_name=filename_pdf, mime="application/pdf")
            except Exception as e:
                st.error(f"Erreur lors de la génération du PDF : {e}")
                import traceback; st.code(traceback.format_exc())
    else:
        st.info("Aucun segment valide n'a été calculé. Vérifiez les entrées ou les sélections.")
