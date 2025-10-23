# co2_calculator_app.py
import os
import io
import time
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

st.set_page_config(page_title="Calculateur CO₂ multimodal - NILEY EXPERTS", page_icon="🌍", layout="centered")

# =========================
# 🎨 Styles
# =========================
st.markdown(f"""
<style>
.stApp {{
  background-image: url("{BACKGROUND_URL}");
  background-size: cover;
  background-repeat: no-repeat;
  background-attachment: fixed;
}}
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
.css-1dp5vir, .stSelectbox > div > div {{
  background-color: #DFEDF5 !important; border: 2px solid #BB9357; border-radius: 5px;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# 🧰 Utils
# =========================
def read_secret(key: str, default: str = "") -> str:
    # Compatible local et cloud
    if "secrets" in dir(st) and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

@st.cache_data(show_spinner=False, ttl=60*60)
def geocode_cached(query: str, limit: int = 5):
    if not query:
        return []
    try:
        # petite temporisation pour éviter de spammer en dev
        time.sleep(0.1)
        return geocoder.geocode(query, no_annotations=1, limit=limit) or []
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=24*60*60)
def coords_from_formatted(formatted: str):
    # Re-géocode le libellé "formatted" sélectionné
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

# =========================
# 🔐 API OpenCage
# =========================
API_KEY = read_secret("OPENCAGE_KEY")
if not API_KEY:
    st.error("Clé API OpenCage absente. Ajoutez `OPENCAGE_KEY` à `st.secrets` ou à vos variables d’environnement.")
    st.stop()
geocoder = OpenCageGeocode(API_KEY)

# =========================
# 🏷️ Header
# =========================
st.markdown("""
<div style='background-color:#002E49;padding:20px;border-radius:10px'>
<h1 style='color:white;text-align:center;margin:0'>Calculateur d'empreinte carbone multimodal - NILEY EXPERTS</h1>
</div>
""", unsafe_allow_html=True)
st.write("Ajoutez plusieurs segments (origine → destination), choisissez le mode et le poids transporté. Les facteurs sont éditables.")

# =========================
# 🔄 Reset
# =========================
col_r, col_dummy = st.columns([1,4])
with col_r:
    if st.button("🔄 Réinitialiser le formulaire"):
        st.cache_data.clear()
        st.session_state.clear()
        st.experimental_rerun()

# =========================
# ⚙️ Paramètres
# =========================
with st.expander("⚙️ Paramètres et facteurs d'émission"):
    default_mode = "Envoi unique (même poids sur tous les segments)"
    weight_mode = st.radio("Mode de gestion du poids :", [default_mode, "Poids par segment"], horizontal=False)
    factors = {}
    for mode, val in DEFAULT_EMISSION_FACTORS.items():
        factors[mode] = st.number_input(f"Facteur {mode} (kg CO₂e / tonne.km)", min_value=0.0, value=float(val), step=0.001, format="%.3f", key=f"factor_{mode}")
    unit = st.radio("Unité de saisie du poids", ["kg", "tonnes"], index=0, horizontal=True)

# =========================
# 🧩 Saisie des segments
# =========================
if "segments" not in st.session_state:
    st.session_state.segments = []

num_legs = st.number_input("Nombre de segments de transport", min_value=1, max_value=10, value=max(1, len(st.session_state.segments) or 1), step=1)

# Ajuste la taille de la liste des segments dans le state
while len(st.session_state.segments) < num_legs:
    st.session_state.segments.append({"origin_raw": "", "origin_sel": "", "dest_raw": "", "dest_sel": "", "mode": "Routier 🚚", "weight": 1000.0})
while len(st.session_state.segments) > num_legs:
    st.session_state.segments.pop()

# Chaînage auto: si un segment (i-1) a une destination sélectionnée, préremplir l'origine i
for i in range(1, num_legs):
    prev = st.session_state.segments[i-1]
    cur = st.session_state.segments[i]
    if prev.get("dest_sel") and not cur.get("origin_raw") and not cur.get("origin_sel"):
        cur["origin_raw"] = prev["dest_sel"]
        cur["origin_sel"] = prev["dest_sel"]

segments_out = []
for i in range(num_legs):
    st.markdown(f"<div class='segment-box'><h4>Segment {i+1}</h4>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        origin_raw = st.text_input(f"Origine du segment {i+1}", value=st.session_state.segments[i]["origin_raw"], key=f"origin_input_{i}")
        origin_suggestions = geocode_cached(origin_raw, limit=5) if origin_raw else []
        origin_options = [r['formatted'] for r in origin_suggestions] if origin_suggestions else []
        origin_sel = st.selectbox("Suggestions pour l'origine", origin_options or ["—"], index=0, key=f"origin_select_{i}")
        if origin_sel == "—":
            origin_sel = ""
    with c2:
        dest_raw = st.text_input(f"Destination du segment {i+1}", value=st.session_state.segments[i]["dest_raw"], key=f"dest_input_{i}")
        dest_suggestions = geocode_cached(dest_raw, limit=5) if dest_raw else []
        dest_options = [r['formatted'] for r in dest_suggestions] if dest_suggestions else []
        dest_sel = st.selectbox("Suggestions pour la destination", dest_options or ["—"], index=0, key=f"dest_select_{i}")
        if dest_sel == "—":
            dest_sel = ""

    mode = st.selectbox(f"Mode de transport du segment {i+1}", list(factors.keys()), index=list(factors.keys()).index(st.session_state.segments[i]["mode"]) if st.session_state.segments[i]["mode"] in factors else 0, key=f"mode_{i}")

    if weight_mode == "Poids par segment":
        default_weight = st.session_state.segments[i]["weight"]
        weight_val = st.number_input(f"Poids transporté pour le segment {i+1}", min_value=0.001, value=float(default_weight), step=100.0 if unit=="kg" else 0.1, key=f"weight_{i}")
    else:
        # envoi unique : on prend le poids du segment 0
        default_weight = st.session_state.segments[0]["weight"]
        if i == 0:
            weight_val = st.number_input(f"Poids transporté (appliqué à tous les segments)", min_value=0.001, value=float(default_weight), step=100.0 if unit=="kg" else 0.1, key=f"weight_{i}")
        else:
            weight_val = st.session_state.get("weight_0", default_weight)

    st.markdown("</div>", unsafe_allow_html=True)

    # maj state
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
# 🧮 Calcul
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
            coord1 = coords_from_formatted(seg["origin"]) or coords_from_formatted(seg["origin"])
            coord2 = coords_from_formatted(seg["destination"]) or coords_from_formatted(seg["destination"])

            if not coord1 or not coord2:
                st.error(f"Segment {idx} : lieu introuvable ou ambigu.")
                continue

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
            })

    # Résultats
    if rows:
        df = pd.DataFrame(rows)
        st.success(f"✅ {len(rows)} segment(s) calculé(s) • Distance totale : **{total_distance:.1f} km** • Émissions totales : **{total_emissions:.2f} kg CO₂e**")

        # Tableau
        st.dataframe(df[["Segment", "Origine", "Destination", "Mode", "Distance (km)", f"Poids ({unit})", "Facteur (kg CO₂e/t.km)", "Émissions (kg CO₂e)"]], use_container_width=True)

        # Carte
        st.subheader("🗺️ Carte des segments")
        layers = []
        for r in rows:
            layers.append(pdk.Layer(
                "LineLayer",
                data=[{"from": [r["lon_o"], r["lat_o"]], "to": [r["lon_d"], r["lat_d"]]}],
                get_source_position="from",
                get_target_position="to",
                get_width=4,
                get_color=[187, 147, 87, 160],  # #BB9357
                pickable=True,
            ))
        midpoint = [sum([r["lat_o"] for r in rows]) / len(rows), sum([r["lon_o"] for r in rows]) / len(rows)]
        r_view = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=2)
        st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=r_view, layers=layers))

        # Export CSV
        csv = df.drop(columns=["lat_o","lon_o","lat_d","lon_d"]).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger le détail (CSV)", data=csv, file_name="resultats_co2_multimodal.csv", mime="text/csv")

    else:
        st.info("Aucun segment valide n’a été calculé. Vérifie les entrées ou les sélections.")
