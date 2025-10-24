# co2_calculator_app.py
# ------------------------------------------------------------
# Calculateur CO2 multimodal - NILEY EXPERTS
# - 🔐 Authentification par mot de passe
# - Géocodage OpenCage
# - Distance routière via OSRM + polyline sur la carte
# - Facteurs d'émission éditables, poids global ou par segment
# - Carte PyDeck (PathLayer pour routes OSRM, LineLayer en fallback)
# - Reset via st.rerun() + reset_form()
# - UI segments: boutons d’ajout/suppression
# - Carte: auto-zoom/centrage + balises (points + étiquettes)
# - Export CSV + PDF (1 page paysage) avec logo, n° de dossier, et capture de la carte (PNG)
# - Fond de page Streamlit #DFEDF5 • Fond carte PDF robuste (GeoPandas/pyogrio) + fallback
# ------------------------------------------------------------

import os
import io
import re
import time
import math
import requests
import datetime
import pandas as pd
import streamlit as st
import pydeck as pdk
from opencage.geocoder import OpenCageGeocode
from geopy.distance import great_circle

# --- Image/figures & PDF ---
import matplotlib
matplotlib.use("Agg")  # backend non interactif
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.gridspec import GridSpec
from PIL import Image

# --- Cartographie statique ---
import geopandas as gpd
from shapely.geometry import LineString

# =========================
# 🎯 Paramètres par défaut
# =========================
DEFAULT_EMISSION_FACTORS = {
    "🚛 Routier 🚛": 0.100,   # kg CO2e / t.km
    "✈️ Aérien ✈️": 0.500,
    "🚢 Maritime 🚢": 0.015,
    "🚂 Ferroviaire 🚂": 0.030,
}
MAX_SEGMENTS = 10  # limite haute

# Logo NILEY EXPERTS (URL raw GitHub)
LOGO_URL = "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/NILEY-EXPERTS-logo-removebg-preview.png"

st.set_page_config(
    page_title="Calculateur CO₂ multimodal - NILEY EXPERTS",
    page_icon="🌍",
    layout="centered"
)

# =========================
# 🔐 Authentification simple par mot de passe
# =========================
APP_PASSWORD = "Niley2019!"  # ⚠️ En prod: préférez st.secrets / variable d'environnement

def check_password():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if st.session_state.auth_ok:
        return True

    st.title("🔐 Accès protégé")
    st.write("Veuillez saisir le mot de passe pour accéder à l’application.")
    with st.form("password_form", clear_on_submit=False):
        pwd = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")

    if submitted:
        if pwd == APP_PASSWORD:
            st.session_state.auth_ok = True
            st.success("Connexion réussie. Chargement de l’application…")
            st.rerun()
        else:
            st.error("Mot de passe incorrect. Réessayez.")
    return False

# Bloque l’app tant que non authentifié
if not check_password():
    st.stop()

# =========================
# 🎨 Styles (fond de page personnalisé #DFEDF5)
# =========================
st.markdown(
    """
    <style>
    /* Fond global de l'application */
    .stApp { background-color: #DFEDF5 !important; }
    /* Zone principale */
    .block-container {
        background-color: #DFEDF5 !important;
        padding-top: 1.2rem; padding-bottom: 2rem; border-radius: 8px;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #DFEDF5 !important; }
    /* Expanders : fond blanc pour lisibilité */
    div[data-testid="stExpander"] > details {
        background-color: #FFFFFF !important;
        border-radius: 8px; border: 1px solid #E6EEF3;
    }
    .stButton>button { border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 🧠 Utilitaires
# =========================
def read_secret(key: str, default: str = "") -> str:
    if "secrets" in dir(st) and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

def safe_filename_component(s: str) -> str:
    """Conserve lettres/chiffres/._- et remplace le reste par _"""
    if not s:
        return ""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

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
    """
    lon1, lat1 = coord1[1], coord1[0]
    lon2, lat2 = coord2[1], coord2[0]
    url = f"{base_url.rstrip('/')}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {
        "overview": overview,
        "alternatives": "false",
        "annotations": "false",
        "geometries": "geojson",
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

    for k in ["segments", "osrm_base_url", "weight_0", "case_ref"]:
        if k in st.session_state:
            del st.session_state[k]

    st.cache_data.clear()
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
# 🧩 En-tête avec logo en haut à gauche
# =========================
with st.container():
    col_logo, col_title = st.columns([1, 6], vertical_alignment="center")
    with col_logo:
        st.image(LOGO_URL, caption=None, use_column_width=False, width=120)
    with col_title:
        st.markdown("## Calculateur d'empreinte carbone multimodal - NILEY EXPERTS")
        st.markdown("Ajoutez plusieurs segments (origine → destination), choisissez le mode et le poids. "
                    "Le mode **Routier** utilise OSRM (distance réelle + tracé).")

# =========================
# 🗂️ Dossier Transport
# =========================
st.markdown("### Dossier Transport")
case_ref = st.text_input("Dossier Transport N°", value=st.session_state.get("case_ref", ""),
                         help="Identifiant interne de votre expédition / dossier.")
st.session_state["case_ref"] = case_ref

# =========================
# 🔄 Reset (utilise reset_form)
# =========================
col_r, _ = st.columns([1, 4])
with col_r:
    if st.button("🔄 Réinitialiser le formulaire"):
        reset_form()

# =========================
# ⚙️ Paramètres
# =========================
with st.expander("⚙️ Paramètres, facteurs d'émission & OSRM"):
    default_mode_label = "Envoi unique (même poids sur tous les segments)"
    weight_mode = st.radio(
        "Mode de gestion du poids :", [default_mode_label, "Poids par segment"], horizontal=False
    )

    factors = {}
    for mode_name, val in DEFAULT_EMISSION_FACTORS.items():
        factors[mode_name] = st.number_input(
            f"Facteur {mode_name} (kg CO₂e / tonne.km)",
            min_value=0.0, value=float(val), step=0.001, format="%.3f",
            key=f"factor_{mode_name}"
        )

    unit = st.radio("Unité de saisie du poids", ["kg", "tonnes"], index=0, horizontal=True)

    osrm_help = (
        "**OSRM** – pour test : `https://router.project-osrm.org` (serveur démo, non garanti). "
        "En production, utilisez un serveur auto‑hébergé ou un provider."
    )
    st.markdown(osrm_help)

    osrm_default = st.session_state.get("osrm_base_url", "https://router.project-osrm.org")
    osrm_base_url = st.text_input(
        "Endpoint OSRM", value=osrm_default,
        help="Ex: https://router.project-osrm.org ou votre propre serveur OSRM"
    )
    st.session_state["osrm_base_url"] = osrm_base_url

# =========================
# 🧩 Saisie des segments (avec boutons d'ajout/suppression)
# =========================
def _default_segment(origin_raw="", origin_sel="", dest_raw="", dest_sel="", mode="🚛 Routier 🚛", weight=1000.0):
    return {"origin_raw": origin_raw, "origin_sel": origin_sel,
            "dest_raw": dest_raw,   "dest_sel": dest_sel,
            "mode": mode,           "weight": weight}

# État initial : au moins 1 segment
if "segments" not in st.session_state or not st.session_state.segments:
    st.session_state.segments = [_default_segment()]

# Chaînage auto : si dest[i-1] défini et origin[i] vide → on propage
for i in range(1, len(st.session_state.segments)):
    prev = st.session_state.segments[i - 1]
    cur = st.session_state.segments[i]
    if prev.get("dest_sel") and not cur.get("origin_raw") and not cur.get("origin_sel"):
        cur["origin_raw"] = prev["dest_sel"]
        cur["origin_sel"] = prev["dest_sel"]

segments_out = []

# Rendu de chaque segment
for i in range(len(st.session_state.segments)):
    st.markdown(f"##### Segment {i+1}")

    c1, c2 = st.columns(2)
    with c1:
        origin_raw = st.text_input(
            f"Origine du segment {i+1}",
            value=st.session_state.segments[i]["origin_raw"],
            key=f"origin_input_{i}"
        )
        origin_suggestions = geocode_cached(origin_raw, limit=5) if origin_raw else []
        origin_options = [r['formatted'] for r in origin_suggestions] if origin_suggestions else []
        origin_sel = st.selectbox("Suggestions pour l'origine", origin_options or ["—"], index=0, key=f"origin_select_{i}")
        if origin_sel == "—":
            origin_sel = ""

    with c2:
        dest_raw = st.text_input(
            f"Destination du segment {i+1}",
            value=st.session_state.segments[i]["dest_raw"],
            key=f"dest_input_{i}"
        )
        dest_suggestions = geocode_cached(dest_raw, limit=5) if dest_raw else []
        dest_options = [r['formatted'] for r in dest_suggestions] if dest_suggestions else []
        dest_sel = st.selectbox("Suggestions pour la destination", dest_options or ["—"], index=0, key=f"dest_select_{i}")
        if dest_sel == "—":
            dest_sel = ""

    mode = st.selectbox(
        f"Mode de transport du segment {i+1}",
        list(factors.keys()),
        index=list(factors.keys()).index(st.session_state.segments[i]["mode"])
              if st.session_state.segments[i]["mode"] in factors else 0,
        key=f"mode_{i}"
    )

    # Gestion du poids
    if weight_mode == "Poids par segment":
        default_weight = st.session_state.segments[i]["weight"]
        weight_val = st.number_input(
            f"Poids transporté pour le segment {i+1}",
            min_value=0.001, value=float(default_weight),
            step=100.0 if unit == "kg" else 0.1, key=f"weight_{i}"
        )
    else:
        default_weight = st.session_state.segments[0]["weight"]
        if i == 0:
            weight_val = st.number_input(
                f"Poids transporté (appliqué à tous les segments)",
                min_value=0.001, value=float(default_weight),
                step=100.0 if unit == "kg" else 0.1, key=f"weight_{i}"
            )
        else:
            weight_val = st.session_state.get("weight_0", default_weight)

    st.session_state.segments[i] = {
        "origin_raw": origin_raw, "origin_sel": origin_sel,
        "dest_raw": dest_raw,     "dest_sel": dest_sel,
        "mode": mode,             "weight": weight_val
    }

    segments_out.append({
        "origin": origin_sel or origin_raw or "",
        "destination": dest_sel or dest_raw or "",
        "mode": mode, "weight": weight_val
    })

# Boutons globaux
bc1, bc2, _ = st.columns([2, 2, 6])
with bc1:
    can_add = len(st.session_state.segments) < MAX_SEGMENTS
    if st.button("➕ Ajouter un segment à la fin", key="add_at_end", disabled=not can_add):
        last = st.session_state.segments[-1]
        new_seg = _default_segment(
            origin_raw=last.get("dest_sel") or last.get("dest_raw") or "",
            origin_sel=last.get("dest_sel") or "",
            mode=last.get("mode", "🚛 Routier 🚛"),
            weight=last.get("weight", 1000.0)
        )
        st.session_state.segments.append(new_seg)
        st.rerun()

with bc2:
    if st.button("🗑️ Supprimer le dernier segment", key="del_last", disabled=len(st.session_state.segments) <= 1):
        st.session_state.segments.pop(-1)
        st.rerun()

# =========================
# 🗺️ Fond monde robuste + carte PNG
# =========================
@st.cache_data(show_spinner=False, ttl=6*60*60)
def _load_world():
    """
    Charge le fond 'naturalearth_lowres' (WGS84).
    Essaie pyogrio (si installé), sinon GeoPandas standard. Retourne None si indispo.
    """
    try:
        path = gpd.datasets.get_path('naturalearth_lowres')
    except Exception:
        return None
    # 1) pyogrio
    try:
        import pyogrio  # noqa: F401
        world = gpd.read_file(path, engine="pyogrio")
        return world.to_crs(epsg=4326)
    except Exception:
        pass
    # 2) fallback standard
    try:
        world = gpd.read_file(path)
        return world.to_crs(epsg=4326)
    except Exception:
        return None

def build_map_image(rows: list, figsize_px=(1400, 900)) -> bytes | None:
    """Produit une image PNG de la carte (fond Natural Earth ou fallback océan)."""
    if not rows:
        return None

    world = _load_world()
    all_lats, all_lons = [], []
    for r in rows:
        all_lats.extend([r["lat_o"], r["lat_d"]])
        all_lons.extend([r["lon_o"], r["lon_d"]])
        if r.get("route_coords"):
            for lon, lat in r["route_coords"]:
                all_lats.append(lat); all_lons.append(lon)

    if not all_lats or not all_lons:
        return None

    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)

    lat_pad = (max_lat - min_lat) * 0.10 or 2.0
    lon_pad = (max_lon - min_lon) * 0.10 or 2.0
    x_min, x_max = min_lon - lon_pad, max_lon + lon_pad
    y_min, y_max = min_lat - lat_pad, max_lat + lat_pad

    dpi = 150
    fig_w = figsize_px[0] / dpi
    fig_h = figsize_px[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    if world is not None and len(world) > 0:
        ax.set_facecolor("#eaf4ff")
        world.plot(ax=ax, color="#f7f7f7", edgecolor="#bdbdbd", linewidth=0.5, zorder=0)
    else:
        ax.set_facecolor("#eaf4ff")

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    # Routes OSRM
    for r in rows:
        if r.get("route_coords"):
            try:
                ls = LineString([(lon, lat) for lon, lat in r["route_coords"]])
                xs, ys = ls.xy
                ax.plot(xs, ys, color="#BB9357", linewidth=2.5, alpha=0.95, zorder=3)
            except Exception:
                pass

    # Segments droits
    for r in rows:
        if not r.get("route_coords"):
            ax.plot([r["lon_o"], r["lon_d"]], [r["lat_o"], r["lat_d"]],
                    color="#888888", linewidth=1.8, alpha=0.9, linestyle="-.", zorder=2)

    # Points + labels
    for r in rows:
        ax.scatter(r["lon_o"], r["lat_o"], c="#007AFF", s=40, edgecolor="white", linewidth=0.8, zorder=4)
        ax.scatter(r["lon_d"], r["lat_d"], c="#DC4242", s=40, edgecolor="white", linewidth=0.8, zorder=4)
        ax.text(r["lon_o"], r["lat_o"], f" S{r['Segment']} O", fontsize=9, color="#0a0a0a",
                va="bottom", ha="left", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
        ax.text(r["lon_d"], r["lat_d"], f" S{r['Segment']} D", fontsize=9, color="#0a0a0a",
                va="bottom", ha="left", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Carte des segments (aperçu statique)", fontsize=12, pad=8)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# =========================
# 🔎 Vue auto pour PyDeck
# =========================
def _compute_auto_view(all_lats, all_lons, viewport_px=(900, 600), padding_px=80):
    """Calcule un ViewState (centre + zoom) à partir d'une liste lat/lon."""
    if not all_lats or not all_lons:
        return pdk.ViewState(latitude=48.8534, longitude=2.3488, zoom=3)  # Paris ~ Europe
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    mid_lat = (min_lat + max_lat) / 2.0
    mid_lon = (min_lon + max_lon) / 2.0
    span_lat = max(1e-6, max_lat - min_lat)
    span_lon = max(1e-6, max_lon - min_lon)
    span_lon_equiv = span_lon * max(0.1, math.cos(math.radians(mid_lat)))
    world_deg_width = 360.0
    zoom_x = math.log2(world_deg_width / max(1e-6, span_lon_equiv))
    zoom_y = math.log2(180.0 / max(1e-6, span_lat))
    zoom = max(1.0, min(15.0, min(zoom_x, zoom_y)))
    return pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=float(zoom), bearing=0, pitch=0)

# =========================
# 🧾 Génération du PDF (1 page paysage)
# =========================
try:
    RESAMPLE = Image.LANCZOS
except AttributeError:
    from PIL import Image as _Image
    RESAMPLE = _Image.Resampling.LANCZOS

def build_pdf_report(df: pd.DataFrame,
                     total_distance_km: float,
                     total_emissions_kg: float,
                     unit_label: str,
                     company: str = "NILEY EXPERTS",
                     logo_url: str | None = None,
                     map_png_bytes: bytes | None = None,
                     case_ref: str | None = None) -> bytes:
    """
    Génère un PDF **sur une seule page A4 paysage** (Matplotlib + PdfPages):
      - Entête (logo + titre + métadonnées + n° dossier)
      - Carte (si fournie)
      - Tableau compacté (troncature avec …)
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Mise en page A4 paysage
    dpi = 150
    fig_w, fig_h = (11.69, 8.27)  # pouces

    # Logo (optionnel)
    logo_img = None
    if logo_url:
        try:
            resp = requests.get(logo_url, timeout=10)
            resp.raise_for_status()
            logo_img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        except Exception:
            logo_img = None

    # Figure + grille
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = GridSpec(100, 100, figure=fig)
    fig.patch.set_facecolor("white")

    # Entête
    ax_header = fig.add_subplot(gs[0:20, 0:100]); ax_header.axis("off")
    y_cursor = 0.92
    if logo_img is not None:
        target_px = 150
        w, h = logo_img.size
        scale = target_px / float(w)
        new_w, new_h = int(w * scale), int(h * scale)
        logo_disp = logo_img.resize((new_w, new_h), RESAMPLE)
        imagebox = OffsetImage(logo_disp, zoom=1.0)
        ab = AnnotationBbox(imagebox, (0.06, y_cursor), frameon=False, xycoords='axes fraction')
        ax_header.add_artist(ab)

    ax_header.text(0.18, y_cursor, "Rapport d'empreinte carbone multimodal",
                   ha="left", va="center", fontsize=16, fontweight="bold", transform=ax_header.transAxes)
    y_cursor -= 0.28

    meta_lines = [f"Généré le : {now_str}", f"Éditeur : {company}"]
    if case_ref:
        meta_lines.append(f"Dossier Transport N° : {case_ref}")
    ax_header.text(0.18, y_cursor, "\n".join(meta_lines),
                   ha="left", va="top", fontsize=10, transform=ax_header.transAxes)

    resume_txt = f"Distance totale : {total_distance_km:.1f} km\nÉmissions totales : {total_emissions_kg:.2f} kg CO₂e"
    ax_header.text(0.60, 0.30, resume_txt, ha="left", va="center", fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.3", fc="#f6f6f6", ec="#dddddd"), transform=ax_header.transAxes)

    ax_header.plot([0.02, 0.98], [0.05, 0.05], color="#444444", linewidth=0.8, transform=ax_header.transAxes)

    # Carte (si dispo)
    if map_png_bytes:
        try:
            ax_map = fig.add_subplot(gs[22:62, 0:48]); ax_map.axis("off")
            map_img = Image.open(io.BytesIO(map_png_bytes)).convert("RGB")
            ax_map.imshow(map_img)
            ax_map.set_title("Carte des segments (aperçu statique)", fontsize=10, pad=4)
        except Exception:
            ax_map = None
    else:
        ax_map = None

    # Tableau compacté
    columns = [
        "Segment", "Origine", "Destination", "Mode",
        "Distance (km)", f"Poids ({unit_label})",
        "Facteur (kg CO₂e/t.km)", "Émissions (kg CO₂e)"
    ]
    table_df = df[columns].copy()

    def trunc(s, maxlen):
        s = "" if pd.isna(s) else str(s)
        return s if len(s) <= maxlen else s[: max(0, maxlen - 1)] + "…"

    maxlens = {
        "Segment": 4, "Origine": 36, "Destination": 36, "Mode": 10,
        "Distance (km)": 8, f"Poids ({unit_label})": 8,
        "Facteur (kg CO₂e/t.km)": 10, "Émissions (kg CO₂e)": 10
    }
    for col, ml in maxlens.items():
        table_df[col] = table_df[col].map(lambda x: trunc(x, ml))

    table_fontsize = 7.5
    ax_tbl = fig.add_subplot(gs[22:98, 50:100]) if map_png_bytes else fig.add_subplot(gs[22:98, 0:100])
    ax_tbl.axis("off")
    ax_tbl.text(0.0, 1.02, "Détail des segments", ha="left", va="bottom", fontsize=11, fontweight="bold",
                transform=ax_tbl.transAxes)

    tbl = ax_tbl.table(
        cellText=table_df.values, colLabels=table_df.columns.tolist(),
        cellLoc="left", colLoc="left", loc="upper left"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(table_fontsize)
    tbl.scale(1.0, 0.9)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#f0f0f0"); cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#dddddd")

    out = io.BytesIO()
    with PdfPages(out) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out.getvalue()

# =========================
# ▶️ Bouton de calcul
# =========================
if st.button("Calculer l'empreinte carbone totale"):
    rows = []
    total_emissions = 0.0
    total_distance = 0.0

    with st.spinner("Calcul en cours…"):
        for idx, seg in enumerate(segments_out, start=1):
            if not seg["origin"] or not seg["destination"]:
                st.warning(f"Segment {idx} : origine/destination manquante(s).")
                continue
            coord1 = coords_from_formatted(seg["origin"])
            coord2 = coords_from_formatted(seg["destination"])
            if not coord1 or not coord2:
                st.error(f"Segment {idx} : lieu introuvable ou ambigu.")
                continue

            # Distance: OSRM pour Routier, sinon grand-cercle
            route_coords = None
            if seg["mode"].startswith("Routier") or "Routier" in seg["mode"]:
                try:
                    r = osrm_route(coord1, coord2, st.session_state["osrm_base_url"], overview="full")
                    distance_km = r["distance_km"]; route_coords = r["coords"]
                except Exception as e:
                    st.warning(f"Segment {idx}: OSRM indisponible ({e}). Distance à vol d’oiseau utilisée.")
                    distance_km = compute_distance_km(coord1, coord2)
            else:
                distance_km = compute_distance_km(coord1, coord2)

            weight_tonnes = seg["weight"] if unit == "tonnes" else seg["weight"]/1000.0
            factor = float(factors.get(seg["mode"], 0.0))
            emissions = compute_emissions(distance_km, weight_tonnes, factor)

            total_distance += distance_km
            total_emissions += emissions

            rows.append({
                "Segment": idx, "Origine": seg["origin"], "Destination": seg["destination"], "Mode": seg["mode"],
                "Distance (km)": round(distance_km, 1),
                f"Poids ({unit})": round(seg["weight"], 3 if unit=="tonnes" else 1),
                "Facteur (kg CO₂e/t.km)": factor,
                "Émissions (kg CO₂e)": round(emissions, 2),
                "lat_o": coord1[0], "lon_o": coord1[1], "lat_d": coord2[0], "lon_d": coord2[1],
                "route_coords": route_coords,
            })

    # ---- Résultats
    if rows:
        df = pd.DataFrame(rows)
        st.success(
            f"✅ {len(rows)} segment(s) calculé(s) • Distance totale : **{total_distance:.1f} km** • "
            f"Émissions totales : **{total_emissions:.2f} kg CO₂e**"
        )

        # Stock session pour CSV/PDF
        st.session_state["last_result_df"] = df
        st.session_state["last_total_distance"] = total_distance
        st.session_state["last_total_emissions"] = total_emissions
        st.session_state["last_unit"] = unit

        # 🗺️ Carte interactive (pydeck)
        st.subheader("🗺️ Carte des segments")

        route_paths = []
        for r in rows:
            if (r["Mode"].startswith("Routier") or "Routier" in r["Mode"]) and r.get("route_coords"):
                route_paths.append({"path": r["route_coords"], "name": f"Segment {r['Segment']} - {r['Mode']}"})

        layers = []
        if route_paths:
            layers.append(pdk.Layer(
                "PathLayer", data=route_paths, get_path="path",
                get_color=[187, 147, 87, 220], width_scale=1, width_min_pixels=4, pickable=True,
            ))

        straight_lines = []
        for r in rows:
            has_osrm = (r["Mode"].startswith("Routier") or "Routier" in r["Mode"]) and r.get("route_coords")
            if not has_osrm:
                straight_lines.append({
                    "from": [r["lon_o"], r["lat_o"]], "to": [r["lon_d"], r["lat_d"]],
                    "name": f"Segment {r['Segment']} - {r['Mode']}",
                })
        if straight_lines:
            layers.append(pdk.Layer(
                "LineLayer", data=straight_lines,
                get_source_position="from", get_target_position="to",
                get_width=3, get_color=[120, 120, 120, 160], pickable=True,
            ))

        points, labels = [], []
        for r in rows:
            points.append({"position": [r["lon_o"], r["lat_o"]], "name": f"S{r['Segment']} • Origine", "color": [0, 122, 255, 220]})
            labels.append({"position": [r["lon_o"], r["lat_o"]], "text": f"S{r['Segment']} O", "color": [0, 122, 255, 255]})
            points.append({"position": [r["lon_d"], r["lat_d"]], "name": f"S{r['Segment']} • Destination", "color": [220, 66, 66, 220]})
            labels.append({"position": [r["lon_d"], r["lat_d"]], "text": f"S{r['Segment']} D", "color": [220, 66, 66, 255]})

        if points:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=points,
                get_position="position", get_fill_color="color",
                get_radius=20000, radius_min_pixels=4, radius_max_pixels=12,
                pickable=True, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1,
            ))
        if labels:
            layers.append(pdk.Layer(
                "TextLayer", data=labels,
                get_position="position", get_text="text", get_color="color",
                get_size=16, size_units="pixels",
                get_text_anchor='"start"', get_alignment_baseline='"top"',
                background=True, get_background_color=[255, 255, 255, 160],
            ))

        all_lats, all_lons = [], []
        if route_paths and any(d["path"] for d in route_paths):
            all_lats.extend([pt[1] for d in route_paths for pt in d["path"]])
            all_lons.extend([pt[0] for d in route_paths for pt in d["path"]])
        all_lats.extend([r["lat_o"] for r in rows] + [r["lat_d"] for r in rows])
        all_lons.extend([r["lon_o"] for r in rows] + [r["lon_d"] for r in rows])

        view = _compute_auto_view(all_lats, all_lons)
        st.pydeck_chart(pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            initial_view_state=view, layers=layers, tooltip={"text": "{name}"}
        ))

        # ======= Image statique de la carte (PNG) pour PDF =======
        map_png_bytes = build_map_image(rows, figsize_px=(1400, 900))
        st.session_state["last_map_png"] = map_png_bytes
        if _load_world() is None:
            st.info("ℹ️ Fond Natural Earth indisponible : un fond simplifié a été utilisé pour la carte du PDF.")

        # Rappel dossier (UI)
        if st.session_state.get("case_ref"):
            st.info(f"**Dossier Transport N° :** {st.session_state['case_ref']}")

        # ✅ Tableau (aperçu)
        st.dataframe(
            df[[
                "Segment", "Origine", "Destination", "Mode",
                "Distance (km)", f"Poids ({unit})",
                "Facteur (kg CO₂e/t.km)", "Émissions (kg CO₂e)"
            ]],
            use_container_width=True
        )

        # Noms de fichiers avec suffixe dossier
        suffix = f"_{safe_filename_component(st.session_state['case_ref'])}" if st.session_state.get("case_ref") else ""
        csv_name = f"resultats_co2_multimodal{suffix}.csv"
        pdf_name = f"rapport_co2_multimodal{suffix}.pdf"

        # Export CSV
        csv = df.drop(columns=["lat_o","lon_o","lat_d","lon_d","route_coords"]).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger le détail (CSV)", data=csv, file_name=csv_name, mime="text/csv")

        # Export PDF (1 page paysage, avec logo + carte + n° dossier)
        try:
            pdf_bytes = build_pdf_report(
                df=st.session_state["last_result_df"],
                total_distance_km=st.session_state["last_total_distance"],
                total_emissions_kg=st.session_state["last_total_emissions"],
                unit_label=st.session_state["last_unit"],
                company="NILEY EXPERTS",
                logo_url=LOGO_URL,
                map_png_bytes=st.session_state.get("last_map_png"),
                case_ref=st.session_state.get("case_ref")
            )
            st.download_button("🧾 Exporter le rapport (PDF)", data=pdf_bytes, file_name=pdf_name, mime="application/pdf")
        except Exception as e:
            st.warning(f"Impossible de générer le PDF : {e}")

        # (Optionnel) PNG de la carte
        if map_png_bytes:
            st.download_button("🖼️ Télécharger l'image de la carte (PNG)",
                               data=map_png_bytes, file_name=f"carte_co2_multimodal{suffix}.png", mime="image/png")

    else:
        st.info("Aucun segment valide n’a été calculé. Vérifiez les entrées ou les sélections.")
``
