# -*- coding: utf-8 -*-
# co2_calculator_app.py
# ------------------------------------------------------------
# Calculateur CO2 multimodal - NILEY EXPERTS
# - Géocodage OpenCage
# - Distance routière via OSRM + polyline sur la carte
# - Facteurs d'émission éditables, poids global ou par segment
# - Carte PyDeck (PathLayer pour routes OSRM, LineLayer en fallback)
# - Reset via st.rerun() + reset_form()
# - UI segments: boutons d’ajout/suppression (plus de "Nombre de segments")
# - Carte: auto-zoom/centrage + balises (points + étiquettes)
# - Option: rayon des points dynamique (mètres) ou fixe (pixels)
# - Fond de page rendu transparent via CSS
# - Logo en haut à gauche (PNG transparent)
# - Icônes par mode sur le trait (IconLayer au milieu du segment)
# ------------------------------------------------------------

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

# =========================
# 🎯 Paramètres par défaut
# =========================
DEFAULT_EMISSION_FACTORS = {
    "🚛 Routier 🚛": 0.100,  # kg CO2e / t.km
    "✈️ Aérien ✈️": 0.500,
    "🛢️ Maritime 🛢️": 0.015,
    "🚂 Ferroviaire 🚂": 0.030,
}
MAX_SEGMENTS = 10  # limite haute

st.set_page_config(
    page_title="Calculateur CO₂ multimodal - NILEY EXPERTS",
    page_icon="🌍",
    layout="centered"
)

# =========================
# 🎨 Fond de page (suppression de la couleur de fond)
# =========================
# On force un fond transparent pour l'app, l'en-tête et le conteneur principal.
# Si tu préfères un fond blanc, remplace `transparent` par `#ffffff`.
st.markdown("""
    <style>
        .stApp, .stApp > header, .block-container {
            background: transparent !important;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# 🖼️ Logo + fond
# =========================
LOGO_URL = "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/NILEY-EXPERTS-logo-removebg-preview.png"

st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:10px;margin:4px 0 12px 0">
        <img src="{LOGO_URL}" alt="NILEY EXPERTS" height="48" stylepx;line-height:1.2">
            Calculateur d'empreinte carbone multimodal - NILEY EXPERTS
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

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
    Utilise: overview=full, géometries=geojson, alternatives=false, annotations=false
    """
    # OSRM attend lon,lat
    lon1, lat1 = coord1[1], coord1[0]
    lon2, lat2 = coord2[1], coord2[0]
    url = f"{base_url.rstrip('/')}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {
        "overview": overview,  # 'simplified' ou 'full'
        "alternatives": "false",
        "annotations": "false",
        "geometries": "geojson"
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
    """Vide explicitement tous les champs et l'état et relance l'app."""
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
    for k in ["segments", "osrm_base_url", "weight_0", "dossier_transport"]:
        if k in st.session_state:
            del st.session_state[k]
    st.cache_data.clear()
    st.rerun()

# =========================
# 🔐 API OpenCage
# =========================
API_KEY = read_secret("OPENCAGE_KEY")
if not API_KEY:
    st.error("Clé API OpenCage absente. Ajoutez OPENCAGE_KEY à st.secrets ou à vos variables d'environnement.")
    st.stop()

geocoder = OpenCageGeocode(API_KEY)

# =========================
# 🏷️ En-tête & Texte explicatif (ASCII-safe)
# =========================
st.markdown("## Calculateur d'empreinte carbone multimodal - NILEY EXPERTS")
st.markdown(
    "Ajoutez plusieurs segments (origine -> destination), choisissez le mode et le poids. "
    "Le mode Routier utilise OSRM (distance reelle + trace)."
)

# =========================
# 🔄 N° dossier + Reset
# =========================
col_id, col_reset, _ = st.columns([3, 1, 6])
with col_id:
    # Saisie libre du numéro de dossier (stockée dans la session)
    dossier_transport = st.text_input(
        "N° dossier Transport",
        value=st.session_state.get("dossier_transport", ""),
        placeholder="ex : TR-2025-001"
    )
    st.session_state["dossier_transport"] = dossier_transport
with col_reset:
    # Petit espace vertical pour aligner le bouton avec le champ texte
    st.write("")
    if st.button("🔄 Réinitialiser le formulaire", use_container_width=True):
    
# =========================
# ⚙️ Paramètres
# =========================
with st.expander("⚙️ Paramètres, facteurs d'émission & OSRM"):
    default_mode_label = "Envoi unique (meme poids sur tous les segments)"
    weight_mode = st.radio("Mode de gestion du poids :", [default_mode_label, "Poids par segment"], horizontal=False)

    factors = {}
    for mode_name, val in DEFAULT_EMISSION_FACTORS.items():
        factors[mode_name] = st.number_input(
            f"Facteur {mode_name} (kg CO₂e / tonne.km)",
            min_value=0.0,
            value=float(val),
            step=0.001,
            format="%.3f",
            key=f"factor_{mode_name}"
        )

    unit = st.radio("Unite de saisie du poids", ["kg", "tonnes"], index=0, horizontal=True)

    osrm_help = (
        "**OSRM** - pour test : https://router.project-osrm.org (serveur demo, non garanti). "
        "En production, utilisez un serveur auto-heberge ou un provider."
    )
    st.markdown(osrm_help)

    osrm_default = st.session_state.get("osrm_base_url", "https://router.project-osrm.org")
    osrm_base_url = st.text_input(
        "Endpoint OSRM",
        value=osrm_default,
        help="Ex: https://router.project-osrm.org ou votre propre serveur OSRM"
    )
    st.session_state["osrm_base_url"] = osrm_base_url

with st.expander("🎯 Apparence de la carte (points & logos)"):
    # Option de rayon dynamique (mètres) vs fixe (pixels) pour les points
    dynamic_radius = st.checkbox(
        "Rayon des points dynamique (varie avec le zoom)",
        value=True,
        help="Dynamique: en mètres, varie visuellement au zoom. Fixe: en pixels, constant à l'écran."
    )
    if dynamic_radius:
        radius_m = st.slider("Rayon des points (mètres)", 1000, 100000, 20000, 1000)
        radius_px = None
    else:
        radius_px = st.slider("Rayon des points (pixels)", 2, 30, 8, 1)
        radius_m = None

    # Taille des logos (IconLayer) en pixels
    icon_size_px = st.slider("Taille des logos de segment (pixels)", 16, 64, 28, 2)

# =========================
# 🧭 Saisie des segments (avec boutons d'ajout/suppression)
# =========================
def _default_segment(origin_raw="", origin_sel="", dest_raw="", dest_sel="", mode=None, weight=1000.0):
    if mode is None:
        mode = list(DEFAULT_EMISSION_FACTORS.keys())[0]
    return {
        "origin_raw": origin_raw,
        "origin_sel": origin_sel,
        "dest_raw": dest_raw,
        "dest_sel": dest_sel,
        "mode": mode,
        "weight": weight
    }

if "segments" not in st.session_state or not st.session_state.segments:
    st.session_state.segments = [_default_segment()]

# Chaînage auto
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
        origin_raw = st.text_input(
            f"Origine du segment {i+1}",
            value=st.session_state.segments[i]["origin_raw"],
            key=f"origin_input_{i}"
        )
        origin_suggestions = geocode_cached(origin_raw, limit=5) if origin_raw else []
        origin_options = [r['formatted'] for r in origin_suggestions] if origin_suggestions else []
        origin_sel = st.selectbox("Suggestions pour l'origine", origin_options or ["-"], index=0, key=f"origin_select_{i}")
        if origin_sel == "-":
            origin_sel = ""
    with c2:
        dest_raw = st.text_input(
            f"Destination du segment {i+1}",
            value=st.session_state.segments[i]["dest_raw"],
            key=f"dest_input_{i}"
        )
        dest_suggestions = geocode_cached(dest_raw, limit=5) if dest_raw else []
        dest_options = [r['formatted'] for r in dest_suggestions] if dest_suggestions else []
        dest_sel = st.selectbox("Suggestions pour la destination", dest_options or ["-"], index=0, key=f"dest_select_{i}")
        if dest_sel == "-":
            dest_sel = ""

    mode = st.selectbox(
        f"Mode de transport du segment {i+1}",
        list(factors.keys()),
        index=list(factors.keys()).index(st.session_state.segments[i]["mode"]) if st.session_state.segments[i]["mode"] in factors else 0,
        key=f"mode_{i}"
    )

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
                key=f"weight_{i}"
            )
        else:
            weight_val = st.session_state.get("weight_0", default_weight)

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

    bc1, bc2, _ = st.columns([2, 2, 6])
    with bc1:
        can_add = len(st.session_state.segments) < MAX_SEGMENTS
        if st.button("➕ Ajouter un segment après ce segment", key=f"add_after_{i}", disabled=not can_add):
            new_seg = _default_segment(
                origin_raw=dest_sel or dest_raw or "",
                origin_sel=dest_sel or "",
                mode=st.session_state.segments[i]["mode"],
                weight=st.session_state.segments[i]["weight"]
            )
            st.session_state.segments.insert(i + 1, new_seg)
            st.rerun()
    with bc2:
        if st.button("🗑️ Supprimer ce segment", key=f"del_{i}", disabled=len(st.session_state.segments) <= 1):
            st.session_state.segments.pop(i)
            st.rerun()

st.markdown("")
if st.button("➕ Ajouter un segment à la fin", key="add_at_end", disabled=len(st.session_state.segments) >= MAX_SEGMENTS):
    last = st.session_state.segments[-1]
    new_seg = _default_segment(
        origin_raw=last.get("dest_sel") or last.get("dest_raw") or "",
        origin_sel=last.get("dest_sel") or "",
        mode=last.get("mode", list(DEFAULT_EMISSION_FACTORS.keys())[0]),
        weight=last.get("weight", 1000.0)
    )
    st.session_state.segments.append(new_seg)
    st.rerun()

# =========================
# 🧮 Calcul + Carte (auto-zoom + balises + logos sur trait)
# =========================
def _compute_auto_view(all_lats, all_lons, viewport_px=(900, 600), padding_px=80):
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

# -- Helpers pour logos --
def _normalize_no_diacritics(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()

def mode_to_category(mode_str: str) -> str:
    """Mappe un libellé de mode vers {routier,aerien,maritime,ferroviaire}."""
    s = _normalize_no_diacritics(mode_str)
    if "routier" in s: return "routier"
    if "aerien" in s or "aérien" in s: return "aerien"
    if "maritime" in s or "mer" in s or "bateau" in s: return "maritime"
    if "ferroviaire" in s or "train" in s: return "ferroviaire"
    if "road" in s or "truck" in s: return "routier"
    if "air" in s or "plane" in s: return "aerien"
    if "sea" in s or "ship" in s: return "maritime"
    if "rail" in s: return "ferroviaire"
    return "routier"

# URLs RAW de tes icônes (ajuste si besoin)
ICON_URLS = {
    "routier": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/truck.png",
    "aerien": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/plane.png",
    "maritime": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/ship.png",
    "ferroviaire": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/train.png",
}

def midpoint_on_path(route_coords, lon_o, lat_o, lon_d, lat_d):
    """
    Retourne le point [lon,lat] au milieu du tracé.
    - Si route_coords (OSRM) est dispo : prend le point central de la polyline.
    - Sinon : milieu géométrique du segment (origine/destination).
    """
    if route_coords and isinstance(route_coords, list) and len(route_coords) >= 2:
        idx = len(route_coords) // 2
        pt = route_coords[idx]
        return [float(pt[0]), float(pt[1])]
    return [(lon_o + lon_d) / 2.0, (lat_o + lat_d) / 2.0]

if st.button("Calculer l'empreinte carbone totale"):
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

            route_coords = None  # liste de [lon, lat]
            if "routier" in _normalize_no_diacritics(seg["mode"]):
                try:
                    r = osrm_route(coord1, coord2, st.session_state["osrm_base_url"], overview="full")
                    distance_km = r["distance_km"]
                    route_coords = r["coords"]
                except Exception as e:
                    st.warning(f"Segment {idx}: OSRM indisponible ({e}). Distance à vol d'oiseau utilisée.")
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
                "route_coords": route_coords,
            })

    if rows:
        df = pd.DataFrame(rows)
        st.success(
            f"✅ {len(rows)} segment(s) calculé(s) • Distance totale : {total_distance:.1f} km • "
            f"Émissions totales : {total_emissions:.2f} kg CO₂e"
        )

        # Affiche le N° de dossier si saisi
        if st.session_state.get("dossier_transport"):
            st.info(f"**N° dossier Transport :** {st.session_state['dossier_transport']}")

        st.dataframe(
            df[["Segment", "Origine", "Destination", "Mode", "Distance (km)", f"Poids ({unit})", "Facteur (kg CO₂e/t.km)", "Émissions (kg CO₂e)"]],
            use_container_width=True
        )

        # ---------- 🗺️ Carte : routes + points + étiquettes + icônes de mode ----------
        st.subheader("Carte des segments")

        # 1) Lignes (OSRM ou droites)
        route_paths = []
        for r in rows:
            if "routier" in _normalize_no_diacritics(r["Mode"]) and r.get("route_coords"):
                route_paths.append({
                    "path": r["route_coords"],
                    "name": f"Segment {r['Segment']} - {r['Mode']}",
                })

        layers = []
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

        straight_lines = []
        for r in rows:
            if not ("routier" in _normalize_no_diacritics(r["Mode"]) and r.get("route_coords")):
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

        # 2) Points + étiquettes (origines/destinations)
        points, labels = [], []
        for r in rows:
            # Origine
            points.append({"position": [r["lon_o"], r["lat_o"]], "name": f"S{r['Segment']} • Origine", "color": [0, 122, 255, 220]})
            labels.append({"position": [r["lon_o"], r["lat_o"]], "text": f"S{r['Segment']} O", "color": [0, 122, 255, 255]})
            # Destination
            points.append({"position": [r["lon_d"], r["lat_d"]], "name": f"S{r['Segment']} • Destination", "color": [220, 66, 66, 220]})
            labels.append({"position": [r["lon_d"], r["lat_d"]], "text": f"S{r['Segment']} D", "color": [220, 66, 66, 255]})

        if points:
            if dynamic_radius:
                layers.append(pdk.Layer(
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
                ))
            else:
                layers.append(pdk.Layer(
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
                ))

        if labels:
            # Retire le fond des étiquettes pour un rendu sans "pavé" blanc
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
                background=False  # pas de fond
            ))

        # 3) Icônes de mode au milieu du trait
        icons = []
        for r in rows:
            cat = mode_to_category(r["Mode"])
            url = ICON_URLS.get(cat)
            if not url:
                continue
            # Point milieu du trajet
            lon_mid, lat_mid = midpoint_on_path(
                r.get("route_coords"),
                r["lon_o"], r["lat_o"],
                r["lon_d"], r["lat_d"]
            )
            icons.append({
                "position": [lon_mid, lat_mid],
                "name": f"S{r['Segment']} - {cat.capitalize()}",
                "icon": {
                    "url": url,
                    "width": 64,
                    "height": 64,
                    "anchorY": 64,
                    "anchorX": 32
                },
            })
        if icons:
            layers.append(pdk.Layer(
                "IconLayer",
                data=icons,
                get_icon="icon",
                get_position="position",
                get_size=icon_size_px,
                size_units="pixels",
                pickable=True,
            ))

        # 4) Vue automatiquement adaptée (tous points + polylignes)
        all_lats, all_lons = [], []
        if route_paths and any(d["path"] for d in route_paths):
            all_lats.extend([pt[1] for d in route_paths for pt in d["path"]])
            all_lons.extend([pt[0] for d in route_paths for pt in d["path"]])
        all_lats.extend([r["lat_o"] for r in rows] + [r["lat_d"] for r in rows])
        all_lons.extend([r["lon_o"] for r in rows] + [r["lon_d"] for r in rows])

        view = _compute_auto_view(all_lats, all_lons, viewport_px=(900, 600), padding_px=80)

        st.pydeck_chart(pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            initial_view_state=view,
            layers=layers,
            tooltip={"text": "{name}"}
        ))

        # Export CSV (sans colonnes techniques)
        csv = df.drop(columns=["lat_o","lon_o","lat_d","lon_d","route_coords"]).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger le détail (CSV)", data=csv, file_name="resultats_co2_multimodal.csv", mime="text/csv")
    else:
        st.info("Aucun segment valide n’a été calculé. Vérifiez les entrées ou les sélections.")
