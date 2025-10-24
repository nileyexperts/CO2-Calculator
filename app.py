# co2_calculator_app.py
# ------------------------------------------------------------
# Calculateur CO2 multimodal - NILEY EXPERTS
# - Géocodage OpenCage
# - Distance routière via OSRM + polyline sur la carte
# - Facteurs d'émission éditables, poids global ou par segment
# - Carte PyDeck (PathLayer pour routes OSRM, LineLayer en fallback)
# - Reset via st.rerun() + reset_form()
# - UI segments: boutons d’ajout/suppression
# - Carte: auto-zoom/centrage + balises (points + étiquettes)
# - Export CSV + PDF avec logo et capture de la carte (PNG)
# ------------------------------------------------------------

import os
import io
import time
import math
import requests
import datetime
import pandas as pd
import streamlit as st
import pydeck as pdk
from opencage.geocoder import OpenCageGeocode
from geopy.distance import great_circle

# --- Pour la génération d'image de la carte (statique) et du PDF
import matplotlib
matplotlib.use("Agg")  # backend non interactif
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
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
# 🎨 Styles (placeholder)
# =========================
st.markdown("""""", unsafe_allow_html=True)

# =========================
# 🧠 Utilitaires
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
    Utilise: overview=full, geometries=geojson, alternatives=false, annotations=false
    """
    # OSRM attend lon,lat
    lon1, lat1 = coord1[1], coord1[0]
    lon2, lat2 = coord2[1], coord2[0]
    url = f"{base_url.rstrip('/')}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {
        "overview": overview,           # 'simplified' ou 'full'
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
    # Supprimer les clés de widgets potentielles
    widget_keys = []
    for i in range(max_segments):
        widget_keys.extend([
            f"origin_input_{i}",
            f"origin_select_{i}",
            f"dest_input_{i}",
            f"dest_select_{i}",
            f"mode_{i}",
            f"weight_{i}",
        ])
    for k in widget_keys:
        if k in st.session_state:
            del st.session_state[k]

    # Réinitialiser les données d'app
    for k in ["segments", "osrm_base_url", "weight_0"]:
        if k in st.session_state:
            del st.session_state[k]

    # Purger le cache (géocodage, routes, etc.)
    st.cache_data.clear()

    # Relancer l'app
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
# 🏷️ En-tête & Texte explicatif
# =========================
st.markdown("""
## Calculateur d'empreinte carbone multimodal - NILEY EXPERTS
""", unsafe_allow_html=True)
st.markdown("""
Ajoutez plusieurs segments (origine → destination), choisissez le mode et le poids. Le mode **Routier** utilise OSRM (distance réelle + tracé).
""", unsafe_allow_html=True)

# =========================
# 🔄 Reset (utilise reset_form)
# =========================
col_r, col_dummy = st.columns([1, 4])
with col_r:
    if st.button("🔄 Réinitialiser le formulaire"):
        reset_form()

# =========================
# ⚙️ Paramètres
# =========================
with st.expander("⚙️ Paramètres, facteurs d'émission & OSRM"):
    default_mode_label = "Envoi unique (même poids sur tous les segments)"
    weight_mode = st.radio(
        "Mode de gestion du poids :",
        [default_mode_label, "Poids par segment"],
        horizontal=False
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
        "Endpoint OSRM",
        value=osrm_default,
        help="Ex: https://router.project-osrm.org ou votre propre serveur OSRM"
    )
    st.session_state["osrm_base_url"] = osrm_base_url

# =========================
# 🧩 Saisie des segments (avec boutons d'ajout/suppression)
# =========================
def _default_segment(origin_raw="", origin_sel="", dest_raw="", dest_sel="", mode="🚛 Routier 🚛", weight=1000.0):
    return {
        "origin_raw": origin_raw,
        "origin_sel": origin_sel,
        "dest_raw": dest_raw,
        "dest_sel": dest_sel,
        "mode": mode,
        "weight": weight
    }

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
    st.markdown(f"##### Segment {i+1}", unsafe_allow_html=True)

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

    # Gestion du poids: "Envoi unique" vs "Poids par segment"
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
                f"Poids transporté (appliqué à tous les segments)",
                min_value=0.001,
                value=float(default_weight),
                step=100.0 if unit == "kg" else 0.1,
                key=f"weight_{i}"
            )
        else:
            # Pas d'input pour les suivants : on réutilise la valeur du segment 0 si présente
            weight_val = st.session_state.get("weight_0", default_weight)

    st.markdown("\n", unsafe_allow_html=True)

    # Mise à jour de l'état pour le segment i
    st.session_state.segments[i] = {
        "origin_raw": origin_raw,
        "origin_sel": origin_sel,
        "dest_raw": dest_raw,
        "dest_sel": dest_sel,
        "mode": mode,
        "weight": weight_val
    }

    segments_out.append({
        "origin": origin_sel or origin_raw or "",
        "destination": dest_sel or dest_raw or "",
        "mode": mode,
        "weight": weight_val
    })

# --- Boutons globaux d'ajout/suppression ---
bc1, bc2, bc3 = st.columns([2, 2, 6])

with bc1:
    # ➕ Ajouter un segment à la fin (pratique)
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
    # 🗑️ Supprimer le dernier segment si > 1
    if st.button("🗑️ Supprimer le dernier segment", key="del_last", disabled=len(st.session_state.segments) <= 1):
        st.session_state.segments.pop(-1)
        st.rerun()

# =========================
# 🗺️ Génération d'une image statique de la carte (PNG)
# =========================
@st.cache_data(show_spinner=False, ttl=6*60*60)
def _load_world():
    # GeoPandas embarque Natural Earth lowres
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        return world.to_crs(epsg=4326)
    except Exception:
        return None

def build_map_image(rows: list, figsize_px=(1400, 900)) -> bytes | None:
    """
    Construit une image PNG (bytes) de la carte :
      - fond monde (Natural Earth)
      - routes OSRM en orange, segments droits en gris
      - points O/D avec labels
      - zoom auto sur l'étendue des données
    """
    if not rows:
        return None

    world = _load_world()
    # Collecte des points pour l'étendue
    all_lats, all_lons = [], []
    for r in rows:
        all_lats.extend([r["lat_o"], r["lat_d"]])
        all_lons.extend([r["lon_o"], r["lon_d"]])
        if r.get("route_coords"):
            for lon, lat in r["route_coords"]:
                all_lats.append(lat)
                all_lons.append(lon)

    if not all_lats or not all_lons:
        return None

    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)

    # Marges (~10%)
    lat_pad = (max_lat - min_lat) * 0.10 or 2.0
    lon_pad = (max_lon - min_lon) * 0.10 or 2.0
    x_min, x_max = min_lon - lon_pad, max_lon + lon_pad
    y_min, y_max = min_lat - lat_pad, max_lat + lat_pad

    # Figure en pouces
    dpi = 150
    fig_w = figsize_px[0] / dpi
    fig_h = figsize_px[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # Fond de carte
    if world is not None:
        world.plot(ax=ax, color="#f7f7f7", edgecolor="#cccccc", linewidth=0.5, zorder=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Tracés
    # 1) Routes OSRM (Path)
    for r in rows:
        if r.get("route_coords"):
            ls = LineString([(lon, lat) for lon, lat in r["route_coords"]])
            xs, ys = ls.xy
            ax.plot(xs, ys, color="#BB9357", linewidth=2.5, alpha=0.95, zorder=3)

    # 2) Segments droits (fallback)
    for r in rows:
        if not r.get("route_coords"):
            ax.plot([r["lon_o"], r["lon_d"]], [r["lat_o"], r["lat_d"]],
                    color="#888888", linewidth=1.8, alpha=0.9, linestyle="-.", zorder=2)

    # 3) Points O/D + labels
    for r in rows:
        ax.scatter(r["lon_o"], r["lat_o"], c="#007AFF", s=40, edgecolor="white", linewidth=0.8, zorder=4)
        ax.scatter(r["lon_d"], r["lat_d"], c="#DC4242", s=40, edgecolor="white", linewidth=0.8, zorder=4)
        ax.text(r["lon_o"], r["lat_o"], f" S{r['Segment']} O", fontsize=9, color="#0a0a0a",
                va="bottom", ha="left", zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
        ax.text(r["lon_d"], r["lat_d"], f" S{r['Segment']} D", fontsize=9, color="#0a0a0a",
                va="bottom", ha="left", zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
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
    """
    Calcule un ViewState (centre + zoom) à partir d'une liste de latitudes/longitudes.
    """
    if not all_lats or not all_lons:
        return pdk.ViewState(latitude=48.8534, longitude=2.3488, zoom=3)  # fallback: Paris ~ Europe
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    # Centre
    mid_lat = (min_lat + max_lat) / 2.0
    mid_lon = (min_lon + max_lon) / 2.0
    # Étendue
    span_lat = max(1e-6, max_lat - min_lat)
    span_lon = max(1e-6, max_lon - min_lon)
    # Corrige l'axe Est-Ouest par cos(lat)
    span_lon_equiv = span_lon * max(0.1, math.cos(math.radians(mid_lat)))
    world_deg_width = 360.0
    zoom_x = math.log2(world_deg_width / max(1e-6, span_lon_equiv))
    zoom_y = math.log2(180.0 / max(1e-6, span_lat))
    zoom = max(1.0, min(15.0, min(zoom_x, zoom_y)))
    return pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=float(zoom), bearing=0, pitch=0)

# =========================
# 🧾 Génération du PDF (Matplotlib + PdfPages)
# =========================
# Compatibilité Pillow (filtre de redimensionnement)
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
                     map_png_bytes: bytes | None = None) -> bytes:
    """
    Génère un PDF (bytes) multi-pages avec Matplotlib + PdfPages :
      - Page 1 : logo (optionnel), titre, résumé, carte (optionnelle)
      - Pages suivantes : tableau des segments paginé
    Format : A4 portrait (8.27 x 11.69 in). Retourne les bytes du PDF.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Mise en page A4
    dpi = 150
    a4_in = (8.27, 11.69)  # pouces (A4)
    fig_w, fig_h = a4_in

    # Récupération logo (si fourni)
    logo_img = None
    if logo_url:
        try:
            resp = requests.get(logo_url, timeout=10)
            resp.raise_for_status()
            logo_img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        except Exception:
            logo_img = None
    # Prépare buffer PDF
    out = io.BytesIO()
    with PdfPages(out) as pdf:

        # ---- Page 1 : Titre, résumé, carte
        fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax1.axis("off")

        y_cursor = 0.94  # position verticale relative (0..1 du haut vers le bas)

        # Logo à gauche
        if logo_img is not None:
            # redimensionner le logo à ~ 2.8 cm de large (≈ 110 px @ dpi 150)
            target_px = 110
            w, h = logo_img.size
            scale = target_px / float(w)
            new_w, new_h = int(w * scale), int(h * scale)
            logo_disp = logo_img.resize((new_w, new_h), RESAMPLE)
            imagebox = OffsetImage(logo_disp, zoom=1.0)
            ab = AnnotationBbox(imagebox, (0.12, y_cursor), frameon=False, xycoords='axes fraction')
            ax1.add_artist(ab)

        # Titre à droite du logo
        ax1.text(0.27, y_cursor, "Rapport d'empreinte carbone multimodal",
                 ha="left", va="center", fontsize=16, fontweight="bold", transform=ax1.transAxes)
        y_cursor -= 0.05

        # Métadonnées
        ax1.text(0.12, y_cursor, f"Généré le : {now_str}", ha="left", va="center", fontsize=10, transform=ax1.transAxes)
        y_cursor -= 0.03
        ax1.text(0.12, y_cursor, f"Éditeur : {company}", ha="left", va="center", fontsize=10, transform=ax1.transAxes)
        y_cursor -= 0.025

        # Séparateur
        ax1.plot([0.12, 0.88], [y_cursor, y_cursor], color="#444444", linewidth=0.8, transform=ax1.transAxes)
        y_cursor -= 0.04

        # Résumé
        ax1.text(0.12, y_cursor, "Résumé", ha="left", va="center", fontsize=12, fontweight="bold", transform=ax1.transAxes)
        y_cursor -= 0.04
        ax1.text(0.12, y_cursor, f"Distance totale : {total_distance_km:.1f} km", ha="left", va="center", fontsize=10, transform=ax1.transAxes)
        y_cursor -= 0.03
        ax1.text(0.12, y_cursor, f"Émissions totales : {total_emissions_kg:.2f} kg CO₂e", ha="left", va="center", fontsize=10, transform=ax1.transAxes)
        y_cursor -= 0.02

        # Carte (si disponible)
        if map_png_bytes:
            try:
                map_img = Image.open(io.BytesIO(map_png_bytes)).convert("RGB")
                # Espace dédié à la carte (hauteur ~ 40% de la page)
                card_h_rel = 0.40
                card_w_rel = 0.76
                # Position rectangle
                x0, y0 = 0.12, max(y_cursor - card_h_rel - 0.01, 0.10)
                x1, y1 = x0 + card_w_rel, y0 + card_h_rel
                ax1.add_patch(plt.Rectangle((x0, y0), card_w_rel, card_h_rel,
                                            transform=ax1.transAxes, fill=False, edgecolor="#dddddd", linewidth=1.0))
                # Affiche l'image
                ax1.imshow(map_img, extent=(x0, x1, y0, y1), transform=ax1.transAxes, aspect='auto', zorder=0)
                y_cursor = y0 - 0.03
            except Exception:
                pass

        # Séparateur bas de page
        ax1.plot([0.12, 0.88], [y_cursor, y_cursor], color="#444444", linewidth=0.6, transform=ax1.transAxes)

        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # ---- Pages suivantes : tableau paginé
        # Colonnes affichées (identiques au dataframe montré)
        columns = [
            "Segment", "Origine", "Destination", "Mode",
            "Distance (km)", f"Poids ({unit_label})",
            "Facteur (kg CO₂e/t.km)", "Émissions (kg CO₂e)"
        ]
        table_df = df[columns].copy()

        # Pagination simple : ~ 30 lignes par page (selon taille police)
        rows_per_page = 30
        n = len(table_df)
        pages = max(1, (n + rows_per_page - 1) // rows_per_page)

        for p in range(pages):
            sub = table_df.iloc[p*rows_per_page:(p+1)*rows_per_page]
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.axis("off")
            ax.text(0.12, 0.95, "Détail des segments", ha="left", va="center",
                    fontsize=12, fontweight="bold", transform=ax.transAxes)

            # Table Matplotlib
            tbl = ax.table(cellText=sub.values,
                           colLabels=sub.columns,
                           loc="upper left",
                           colLoc="left",
                           cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1.0, 1.2)

            # Styliser l’en‑tête
            for (row, col), cell in tbl.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#f0f0f0")
                    cell.set_text_props(fontweight="bold")
                cell.set_edgecolor("#dddddd")

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

            # --- Distance: OSRM + géométrie pour Routier, sinon grand-cercle
            route_coords = None  # liste de [lon, lat]
            if seg["mode"].startswith("Routier") or "Routier" in seg["mode"]:
                try:
                    r = osrm_route(coord1, coord2, st.session_state["osrm_base_url"], overview="full")
                    distance_km = r["distance_km"]
                    route_coords = r["coords"]
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
                "Segment": idx,
                "Origine": seg["origin"],
                "Destination": seg["destination"],
                "Mode": seg["mode"],
                "Distance (km)": round(distance_km, 1),
                f"Poids ({unit})": round(seg["weight"], 3 if unit=="tonnes" else 1),
                "Facteur (kg CO₂e/t.km)": factor,
                "Émissions (kg CO₂e)": round(emissions, 2),
                "lat_o": coord1[0],
                "lon_o": coord1[1],
                "lat_d": coord2[0],
                "lon_d": coord2[1],
                "route_coords": route_coords,  # polyline OSRM si dispo
            })

    # ---- Résultats
    if rows:
        df = pd.DataFrame(rows)
        st.success(
            f"✅ {len(rows)} segment(s) calculé(s) • Distance totale : **{total_distance:.1f} km** • "
            f"Émissions totales : **{total_emissions:.2f} kg CO₂e**"
        )

        # Stocker le dernier résultat pour CSV/PDF
        st.session_state["last_result_df"] = df
        st.session_state["last_total_distance"] = total_distance
        st.session_state["last_total_emissions"] = total_emissions
        st.session_state["last_unit"] = unit

        # 🗺️ Carte interactive (pydeck)
        st.subheader("🗺️ Carte des segments")

        # 1) Data pour les lignes (OSRM ou droites)
        route_paths = []
        for r in rows:
            if (r["Mode"].startswith("Routier") or "Routier" in r["Mode"]) and r.get("route_coords"):
                route_paths.append({
                    "path": r["route_coords"],   # [[lon, lat], ...]
                    "name": f"Segment {r['Segment']} - {r['Mode']}",
                })

        layers = []

        # 1a) Polyline routière exacte (OSRM)
        if route_paths:
            layers.append(pdk.Layer(
                "PathLayer",
                data=route_paths,
                get_path="path",
                get_color=[187, 147, 87, 220],  # #BB9357 avec alpha
                width_scale=1,
                width_min_pixels=4,
                pickable=True,
            ))

        # 1b) Lignes droites pour les segments restants
        straight_lines = []
        for r in rows:
            has_osrm_line = (r["Mode"].startswith("Routier") or "Routier" in r["Mode"]) and r.get("route_coords")
            if not has_osrm_line:
                straight_lines.append({
                    "from": [r["lon_o"], r["lat_o"]],
                    "to": [r["lon_d"], r["lat_d"]],
                    "name": f"Segment {r['Segment']} - {r['Mode']}",
                })
        if straight_lines:
            layers.append(pdk.Layer(
                "LineLayer",
                data=straight_lines,
                get_source_position="from",
                get_target_position="to",
                get_width=3,
                get_color=[120, 120, 120, 160],
                pickable=True,
            ))

        # 2) Points + balises (origines & destinations)
        points = []
        labels = []
        for r in rows:
            # Origine
            points.append({"position": [r["lon_o"], r["lat_o"]], "name": f"S{r['Segment']} • Origine", "color": [0, 122, 255, 220]})
            labels.append({"position": [r["lon_o"], r["lat_o"]], "text": f"S{r['Segment']} O", "color": [0, 122, 255, 255]})
            # Destination
            points.append({"position": [r["lon_d"], r["lat_d"]], "name": f"S{r['Segment']} • Destination", "color": [220, 66, 66, 220]})
            labels.append({"position": [r["lon_d"], r["lat_d"]], "text": f"S{r['Segment']} D", "color": [220, 66, 66, 255]})

        if points:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=points,
                get_position="position",
                get_fill_color="color",
                get_radius=20000,               # rayon en mètres
                radius_min_pixels=4,
                radius_max_pixels=12,
                pickable=True,
                stroked=True,
                get_line_color=[255, 255, 255],
                line_width_min_pixels=1,
            ))

        if labels:
            # littéraux JS pour deck.gl (passer la chaîne avec guillemets)
            layers.append(pdk.Layer(
                "TextLayer",
                data=labels,
                get_position="position",
                get_text="text",
                get_color="color",
                get_size=16,
                size_units="pixels",
                get_text_anchor='"start"',
                get_alignment_baseline='"top"',
                background=True,
                get_background_color=[255, 255, 255, 160],
            ))

        # 3) Vue automatiquement adaptée (tous points + polylignes OSRM)
        all_lats, all_lons = [], []
        if route_paths and any(d["path"] for d in route_paths):
            all_lats.extend([pt[1] for d in route_paths for pt in d["path"]])
            all_lons.extend([pt[0] for d in route_paths for pt in d["path"]])
        # Ajoute aussi les extrémités (utile quand aucun OSRM)
        all_lats.extend([r["lat_o"] for r in rows] + [r["lat_d"] for r in rows])
        all_lons.extend([r["lon_o"] for r in rows] + [r["lon_d"] for r in rows])

        view = _compute_auto_view(all_lats, all_lons, viewport_px=(900, 600), padding_px=80)

        st.pydeck_chart(pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            initial_view_state=view,
            layers=layers,
            tooltip={"text": "{name}"}
        ))

        # ======= Image statique de la carte (PNG) pour PDF =======
        map_png_bytes = build_map_image(rows, figsize_px=(1400, 900))
        st.session_state["last_map_png"] = map_png_bytes

        # Tableau (aperçu)
        st.dataframe(
            df[["Segment", "Origine", "Destination", "Mode", "Distance (km)", f"Poids ({unit})", "Facteur (kg CO₂e/t.km)", "Émissions (kg CO₂e)"]],
            use_container_width=True
        )

        # Export CSV
        csv = df.drop(columns=["lat_o","lon_o","lat_d","lon_d","route_coords"]).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger le détail (CSV)", data=csv, file_name="resultats_co2_multimodal.csv", mime="text/csv")

        # Export PDF (avec logo + carte)
        pdf_name = "rapport_co2_multimodal.pdf"
        try:
            pdf_bytes = build_pdf_report(
                df=st.session_state["last_result_df"],
                total_distance_km=st.session_state["last_total_distance"],
                total_emissions_kg=st.session_state["last_total_emissions"],
                unit_label=st.session_state["last_unit"],
                company="NILEY EXPERTS",
                logo_url=LOGO_URL,
                map_png_bytes=st.session_state.get("last_map_png")
            )
            st.download_button(
                "🧾 Exporter le rapport (PDF)",
                data=pdf_bytes,
                file_name=pdf_name,
                mime="application/pdf"
            )
        except Exception as e:
            st.warning(f"Impossible de générer le PDF : {e}")

        # (Optionnel) Télécharger aussi l'image PNG seule
        if map_png_bytes:
            st.download_button(
                "🖼️ Télécharger l'image de la carte (PNG)",
                data=map_png_bytes,
                file_name="carte_co2_multimodal.png",
                mime="image/png"
            )

    else:
        st.info("Aucun segment valide n’a été calculé. Vérifiez les entrées ou les sélections.")
