# co2_calculator_app.py
# ------------------------------------------------------------
# Calculateur CO2 multimodal - NILEY EXPERTS
# - Géocodage OpenCage
# - Distance routière via OSRM + polyline sur la carte
# - Fond recentré, couleur d'origine + texte explicatif clair
# - Facteurs d'émission éditables, poids global ou par segment
# - Carte PyDeck (PathLayer pour routes OSRM, LineLayer en fallback)
# - Correctif: utilisation de st.rerun() (plus de st.experimental_rerun())
# - Nouveauté: reset_form() pour vider explicitement tous les champs
# ------------------------------------------------------------

import os
import time
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
    "Routier 🚚": 0.100,     # kg CO2e / t.km
    "Aérien ✈️": 0.500,
    "Maritime 🚢": 0.015,
    "Ferroviaire 🚆": 0.030,
}

BACKGROUND_URL = "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/background.png"
MAX_SEGMENTS = 10  # utilisé pour nettoyer toutes les clés potentielles

st.set_page_config(page_title="Calculateur CO₂ multimodal - NILEY EXPERTS",
                   page_icon="🌍", layout="centered")

# =========================
# 🎨 Styles (fond recentré, couleur d'origine)
# =========================
st.markdown(f"""
<style>
/* 🖼️ Image de fond : centrée, couleur d'origine (pas de voile) */
.stApp {{
  background-image: url("{BACKGROUND_URL}");
  background-size: 1200px auto;     /* largeur fixe confortable sur desktop */
  background-repeat: no-repeat;
  background-position: top center;   /* recentré */
  background-attachment: scroll;     /* moins envahissant au scroll */
}}

/* 🎯 Conteneur de contenu : fond blanc léger pour lisibilité */
.main .block-container {{
  background: rgba(255,255,255,0.95);
  border-radius: 10px;
  padding: 1.2rem 1.2rem 1.6rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  max-width: 960px;
}}

/* 🧩 Cartouches segments + contrôles (couleurs d'origine NILEY) */
.segment-box {{
  background-color: #DFEDF5;
  padding: 15px; border-radius: 10px; margin-bottom: 10px;
  border: 2px solid #BB9357;
}}
.stButton > button {{
  background-color: #BB9357; color: white; font-weight: bold;
  border-radius: 8px; padding: 10px 20px; border: none;
}}
.stTextInput > div > input, .stNumberInput > div > input {{
  background-color: #DFEDF5; border: 2px solid #BB9357; border-radius: 5px;
}}
.stSelectbox > div > div {{
  background-color: #DFEDF5 !important; border: 2px solid #BB9357; border-radius: 5px;
}}

/* 📱 Responsive : fond plus petit sur mobile */
@media (max-width: 768px) {{
  .stApp {{ background-size: 900px auto; }}
}}
</style>
""", unsafe_allow_html=True)

# =========================
# 🧰 Utilitaires
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
    geom = route.get("geometry", {})  # GeoJSON LineString
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
            f"origin_input_{i}", f"origin_select_{i}",
            f"dest_input_{i}",   f"dest_select_{i}",
            f"mode_{i}",         f"weight_{i}",
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
# 🏷️ En-tête & Texte explicatif (clair)
# =========================
st.markdown("""
<div style='background-color:#002E49;padding:20px;border-radius:10px'>
  <h1 style='color:white;text-align:center;margin:0'>
    Calculateur d'empreinte carbone multimodal - NILEY EXPERTS
  </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#f2f7fb; font-weight:500; margin-top:6px;">
  Ajoutez plusieurs segments (origine → destination), choisissez le mode et le poids.
  Le mode <strong>Routier</strong> utilise <strong>OSRM</strong> (distance réelle + tracé).
</div>
""", unsafe_allow_html=True)

# =========================
# 🔄 Reset (utilise reset_form)
# =========================
col_r, col_dummy = st.columns([1,4])
with col_r:
    if st.button("🔄 Réinitialiser le formulaire"):
        reset_form()

# =========================
# ⚙️ Paramètres
# =========================
with st.expander("⚙️ Paramètres, facteurs d'émission & OSRM"):
    default_mode = "Envoi unique (même poids sur tous les segments)"
    weight_mode = st.radio("Mode de gestion du poids :", [default_mode, "Poids par segment"], horizontal=False)
    factors = {}
    for mode, val in DEFAULT_EMISSION_FACTORS.items():
        factors[mode] = st.number_input(
            f"Facteur {mode} (kg CO₂e / tonne.km)",
            min_value=0.0, value=float(val), step=0.001, format="%.3f", key=f"factor_{mode}"
        )
    unit = st.radio("Unité de saisie du poids", ["kg", "tonnes"], index=0, horizontal=True)

    osrm_help = ("**OSRM** – pour test : `https://router.project-osrm.org` (serveur démo, non garanti). "
                 "En production, utilisez un serveur auto‑hébergé ou un provider.")
    st.markdown(osrm_help)

    osrm_default = st.session_state.get("osrm_base_url", "https://router.project-osrm.org")
    osrm_base_url = st.text_input("Endpoint OSRM", value=osrm_default,
                                  help="Ex: https://router.project-osrm.org ou votre propre serveur OSRM")
    st.session_state["osrm_base_url"] = osrm_base_url

# =========================
# 🧩 Saisie des segments
# =========================
if "segments" not in st.session_state:
    st.session_state.segments = []

num_legs = st.number_input(
    "Nombre de segments de transport", min_value=1, max_value=MAX_SEGMENTS,
    value=max(1, len(st.session_state.segments) or 1), step=1
)

# Ajuster la liste au nombre demandé
while len(st.session_state.segments) < num_legs:
    st.session_state.segments.append({
        "origin_raw": "", "origin_sel": "", "dest_raw": "", "dest_sel": "",
        "mode": "Routier 🚚", "weight": 1000.0
    })
while len(st.session_state.segments) > num_legs:
    st.session_state.segments.pop()

# Chaînage auto origine[i] = destination[i-1]
for i in range(1, int(num_legs)):
    prev = st.session_state.segments[i-1]
    cur = st.session_state.segments[i]
    if prev.get("dest_sel") and not cur.get("origin_raw") and not cur.get("origin_sel"):
        cur["origin_raw"] = prev["dest_sel"]
        cur["origin_sel"] = prev["dest_sel"]

segments_out = []
for i in range(int(num_legs)):
    st.markdown(f"<div class='segment-box'><h4>Segment {i+1}</h4>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        origin_raw = st.text_input(
            f"Origine du segment {i+1}",
            value=st.session_state.segments[i]["origin_raw"], key=f"origin_input_{i}"
        )
        origin_suggestions = geocode_cached(origin_raw, limit=5) if origin_raw else []
        origin_options = [r['formatted'] for r in origin_suggestions] if origin_suggestions else []
        origin_sel = st.selectbox("Suggestions pour l'origine", origin_options or ["—"], index=0, key=f"origin_select_{i}")
        if origin_sel == "—":
            origin_sel = ""

    with c2:
        dest_raw = st.text_input(
            f"Destination du segment {i+1}",
            value=st.session_state.segments[i]["dest_raw"], key=f"dest_input_{i}"
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
            # Pas d'input pour les suivants : on réutilise la valeur du segment 0 si présente
            weight_val = st.session_state.get("weight_0", default_weight)

    st.markdown("</div>", unsafe_allow_html=True)

    # Maj state
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

# =========================
# 🧮 Calcul + Carte
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
            if seg["mode"].startswith("Routier"):
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
                "lat_o": coord1[0], "lon_o": coord1[1],
                "lat_d": coord2[0], "lon_d": coord2[1],
                "route_coords": route_coords,   # polyline OSRM si dispo
            })

    # ---- Résultats
    if rows:
        df = pd.DataFrame(rows)
        st.success(
            f"✅ {len(rows)} segment(s) calculé(s) • Distance totale : **{total_distance:.1f} km** • "
            f"Émissions totales : **{total_emissions:.2f} kg CO₂e**"
        )

        # Tableau
        st.dataframe(
            df[["Segment", "Origine", "Destination", "Mode", "Distance (km)",
                f"Poids ({unit})", "Facteur (kg CO₂e/t.km)", "Émissions (kg CO₂e)"]],
            use_container_width=True
        )

        # Carte : PathLayer (OSRM) + LineLayer (fallback)
        st.subheader("🗺️ Carte des segments")

        # Données pour PathLayer (routier avec géométrie OSRM)
        route_paths = []
        for r in rows:
            if r["Mode"].startswith("Routier") and r.get("route_coords"):
                route_paths.append({
                    "path": r["route_coords"],   # [[lon, lat], ...]
                    "name": f"Segment {r['Segment']} - {r['Mode']}",
                })

        layers = []

        # 1) Polyline routière exacte (OSRM)
        if route_paths:
            layers.append(pdk.Layer(
                "PathLayer",
                data=route_paths,
                get_path="path",
                get_color=[187, 147, 87, 220],   # #BB9357 avec alpha
                width_scale=1,
                width_min_pixels=4,
                pickable=True,
            ))

        # 2) Lignes droites pour les segments restants
        straight_lines = []
        for r in rows:
            if not (r["Mode"].startswith("Routier") and r.get("route_coords")):
                straight_lines.append({
                    "from": [r["lon_o"], r["lat_o"]],
                    "to":   [r["lon_d"], r["lat_d"]],
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

        # Vue centrée (moyenne simple des points)
        if route_paths and any(d["path"] for d in route_paths):
            all_lats = [pt[1] for d in route_paths for pt in d["path"]]
            all_lons = [pt[0] for d in route_paths for pt in d["path"]]
        else:
            all_lats = [r["lat_o"] for r in rows] + [r["lat_d"] for r in rows]
            all_lons = [r["lon_o"] for r in rows] + [r["lon_d"] for r in rows]

        mid_lat = sum(all_lats) / len(all_lats)
        mid_lon = sum(all_lons) / len(all_lons)

        view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=3)

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",  # peut nécessiter une clé Mapbox pour le fond de carte
            initial_view_state=view,
            layers=layers,
            tooltip={"text": "{name}"}
        ))

        # Export CSV (sans colonnes techniques)
        csv = df.drop(columns=["lat_o","lon_o","lat_d","lon_d","route_coords"]).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger le détail (CSV)", data=csv,
                           file_name="resultats_co2_multimodal.csv", mime="text/csv")

    else:
        st.info("Aucun segment valide n’a été calculé. Vérifie les entrées ou les sélections.")
