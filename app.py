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

# Matplotlib pour le rendu de la carte du PDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Cache Cartopy
os.environ.setdefault("CARTOPY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "cartopy_cache"))

# --- Configuration page ---
st.set_page_config(page_title="Calculateur CO2 multimodal - NILEY EXPERTS", page_icon="🌍", layout="centered")
st.markdown(""" """, unsafe_allow_html=True)

LOGO_URL = "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/NILEY-EXPERTS-logo-removebg-preview.png"

# --- Constantes ---
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
    "aerien": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/plane.png",
    "maritime": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/ship.png",
    "ferroviaire": "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/icons/train.png",
}

# --- Helpers segments ---
def _default_segment(mode=None, weight=1000.0):
    if mode is None:
        mode = list(DEFAULT_EMISSION_FACTORS.keys())[0]
    return {
        "origin": {"query":"", "display":"", "iata":"", "coord":None},
        "dest": {"query":"", "display":"", "iata":"", "coord":None},
        "mode": mode,
        "weight": weight,
    }

# (Toutes les fonctions add_segment_end, remove_last_segment, reset_segments, etc. inchangées)

# --- Champ unifié Adresse/Ville/Pays ou IATA ---
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

    # ✅ Désactiver le champ si origine est autofill
    is_autofill = st.session_state.get(f"origin_autofill_{seg_index}", False) if side_key == "origin" else False
    query_val = st.text_input(label, value=st.session_state.get(q_key, ""), key=q_key, disabled=is_autofill)

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
                exact = airports[airports["iata_code"].str.upper() == q_iata]
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
        cur["origin"]["coord"] = prev["dest"]["coord"]
        cur["origin"]["iata"] = prev["dest"]["iata"]
        cur["origin"]["query"] = prev["dest"]["display"]
        st.session_state[f"origin_autofill_{i}"] = True
        st.session_state[f"chain_src_signature_{i}"] = _normalize_signature(
            prev["dest"].get("display"), prev["dest"].get("coord")
        )
        st.session_state[f"origin_user_edited_{i}"] = False

col_title, col_reset = st.columns([8, 2])
with col_title:
    st.subheader("Saisie des segments")
with col_reset:
    st.button("Reinitialiser", type="secondary", use_container_width=True,
              help="Effacer tous les segments", key="btn_reset_segments", on_click=reset_segments)

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
            mode = st.selectbox("Mode de transport", options=mode_options,
                                index=mode_options.index(current_mode), key=f"mode_select_{i}")
            st.session_state.segments[i]["mode"] = mode

        c1, c2 = st.columns(2)
        with c1:
            if st.session_state.get(f"origin_autofill_{i}", False):
                st.markdown("**Origine** <span class='badge-autofill'>(repris du segment precedent)</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Origine**")
            o = unified_location_input("origin", i, "Origine",
                                       show_airports=("aerien" in _normalize_no_diacritics(mode)))
        with c2:
            st.markdown("**Destination**")
            d = unified_location_input("dest", i, "Destination", show_airports=False)

        # Si l'utilisateur modifie l'origine par rapport a la source de chainage, enlever le badge et verrouiller
        if st.session_state.get(f"origin_autofill_{i}", False) and i > 0:
            prev_sig = st.session_state.get(f"chain_src_signature_{i}")
            cur_sig  = _normalize_signature(o["display"], o["coord"])
            if prev_sig and cur_sig != prev_sig:
                st.session_state[f"origin_autofill_{i}"] = False
                st.session_state[f"origin_user_edited_{i}"] = (not _is_location_empty({"display": o["display"], "coord": o["coord"]}))

        if "weight_0" not in st.session_state:
            st.session_state["weight_0"] = st.session_state.segments[0]["weight"]
        if i == 0:
            weight_val = st.number_input("Poids transporte (applique a tous les segments)",
                                         min_value=0.001, value=float(st.session_state["weight_0"]),
                                         step=100.0, key="weight_0")
        else:
            weight_val = st.session_state["weight_0"]
        st.session_state.segments[i]["weight"] = weight_val

        st.session_state.segments[i]["origin"] = {"query": o["query"], "display": o["display"], "iata": o["iata"], "coord": o["coord"]}
        st.session_state.segments[i]["dest"]   = {"query": d["query"], "display": d["display"], "iata": d["iata"], "coord": d["coord"]}

        # Chaine live D(i) -> O(i+1) robuste
        if i + 1 < len(st.session_state.segments):
            next_seg = st.session_state.segments[i + 1]
            dest_disp = d["display"]; dest_coord = d["coord"]; dest_iata = d["iata"] or ""
            dest_valid = bool(dest_disp and dest_coord)

            next_origin = next_seg.get("origin", {})
            next_empty = _is_location_empty(next_origin)
            next_autofill   = bool(st.session_state.get(f"origin_autofill_{i+1}", False))
            next_user_locked = bool(st.session_state.get(f"origin_user_edited_{i+1}", False))
            next_prev_sig = st.session_state.get(f"chain_src_signature_{i+1}")

            new_sig = _normalize_signature(dest_disp, dest_coord)

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
                next_seg["origin"]["query"]   = dest_disp

                j = i + 1
                st.session_state[f"origin_query_{j}"]   = dest_disp
                st.session_state[f"origin_display_{j}"] = dest_disp
                st.session_state[f"origin_coord_{j}"]   = dest_coord
                st.session_state[f"origin_iata_{j}"]    = dest_iata

                st.session_state[f"origin_autofill_{j}"] = True
                st.session_state[f"chain_src_signature_{j}"] = new_sig
                st.session_state[f"origin_user_edited_{j}"] = False

                st.session_state.segments[j] = next_seg

        if _is_location_empty(st.session_state.segments[i]["origin"]):
            st.session_state[f"origin_user_edited_{i}"] = False

        segments_out.append({
            "origin_display": o["display"], "destination_display": d["display"],
            "origin_iata": o["iata"], "dest_iata": d["iata"],
            "mode": mode, "weight": weight_val,
            "coord_o": o["coord"], "coord_d": d["coord"],
            "origin": o["display"], "destination": d["display"],
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
                    st.warning(f"Segment {idx}: OSRM indisponible ({e}). Distance a vol d'oiseau.")
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
        st.success(f"{len(rows)} segment(s) - Distance totale : {total_distance:.1f} km - Emissions : {total_emissions:.2f} kg CO2e")

        st.dataframe(
            df[["Segment","Origine","Destination","Mode","Distance (km)",f"Poids ({unit})","Facteur (kg CO2e/t.km)","Emissions (kg CO2e)"]],
            use_container_width=True
        )

        route_paths = [
            {"path": r["route_coords"], "name": f"Segment {r['Segment']} - {r['Mode']}"}
            for r in rows if ("routier" in _normalize_no_diacritics(r["Mode"]) and r.get("route_coords"))
        ]
        layers = []
        if route_paths:
            layers.append(pdk.Layer(
                "PathLayer", data=route_paths, get_path="path",
                get_color=[187,147,87,220], width_scale=1, width_min_pixels=4, pickable=True
            ))
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
                "LineLayer", data=straight_lines,
                get_source_position="from", get_target_position="to",
                get_width=3, get_color=[120,120,120,160], pickable=True
            ))
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
                "ScatterplotLayer", data=points, get_position="position",
                get_fill_color="color", get_radius=20000, radius_min_pixels=2, radius_max_pixels=60,
                pickable=True, stroked=True, get_line_color=[255,255,255], line_width_min_pixels=1
            ))
        if labels:
            layers.append(pdk.Layer(
                "TextLayer", data=labels, get_position="position", get_text="text",
                get_color="color", get_size=16, size_units="pixels",
                get_text_anchor="start", get_alignment_baseline="top", background=False
            ))
        if show_icons:
            icons = []
            for r in rows:
                cat = mode_to_category(r["Mode"])
                url = ICON_URLS.get(cat)
                if not url: continue
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

        # Exports
        df_export = df.drop(columns=["lat_o","lon_o","lat_d","lon_d","route_coords"]).copy()
        dossier_val = st.session_state.get("dossier_transport","")
        df_export.insert(0, "N° dossier Transport", dossier_val)
        csv = df_export.to_csv(index=False).encode("utf-8")
        safe_suffix = "".join(c if (c.isalnum() or c in "-_") else "_" for c in dossier_val.strip())
        safe_suffix = f"_{safe_suffix}" if safe_suffix else ""
        filename_csv = f"resultats_co2_multimodal{safe_suffix}.csv"
        filename_pdf = f"rapport_co2_multimodal{safe_suffix}.pdf"

        st.subheader("Exporter")
        pdf_base_choice = st.selectbox(
            "Fond de carte du PDF",
            options=PDF_BASEMAP_LABELS, index=0,
            help="Identique a la carte Web utilise le style Carto (Voyager/Positron/Dark Matter)."
        )

        detail_levels = {
            "Standard (leger, rapide)": {"dpi": 180, "max_zoom": 7},
            "Detaille (equilibre)": {"dpi": 220, "max_zoom": 9},
            "Ultra (fin mais plus lent)": {"dpi": 280, "max_zoom": 10},
        }
        quality_label = st.selectbox(
            "Qualite de rendu PDF",
            options=list(detail_levels.keys()),
            index=1,
            help="Ajuste la finesse du fond de carte: DPI et niveau de zoom."
        )
        detail_params = detail_levels[quality_label]

        c1, c2 = st.columns(2)
        with c1:
            # Exports
df_export = df.drop(columns=["lat_o","lon_o","lat_d","lon_d","route_coords"]).copy()
dossier_val = st.session_state.get("dossier_transport","")
df_export.insert(0, "N° dossier Transport", dossier_val)

csv = df_export.to_csv(index=False).encode("utf-8")
safe_suffix = "".join(c if (c.isalnum() or c in "-_") else "_" for c in dossier_val.strip())
safe_suffix = f"_{safe_suffix}" if safe_suffix else ""
filename_csv = f"resultats_co2_multimodal{safe_suffix}.csv"
filename_pdf = f"rapport_co2_multimodal{safe_suffix}.pdf"

st.subheader("Exporter")
pdf_base_choice = st.selectbox(
    "Fond de carte du PDF",
    options=PDF_BASEMAP_LABELS,
    index=0,
    help="Identique à la carte Web utilise le style Carto (Voyager/Positron/Dark Matter)."
)
detail_levels = {
    "Standard (léger, rapide)": {"dpi": 180, "max_zoom": 7},
    "Détaillé (équilibre)": {"dpi": 220, "max_zoom": 9},
    "Ultra (fin mais plus lent)": {"dpi": 280, "max_zoom": 10},
}
quality_label = st.selectbox(
    "Qualité de rendu PDF",
    options=list(detail_levels.keys()),
    index=1,
    help="Ajuste la finesse du fond de carte: DPI et niveau de zoom."
)
detail_params = detail_levels[quality_label]

c1, c2 = st.columns(2)
with c1:
    st.download_button("Télécharger le détail (CSV)", data=csv, file_name=filename_csv, mime="text/csv")

with c2:
    try:
        with st.spinner("Génération du PDF..."):
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

            # ✅ Sauvegarde dans un dossier persistant
            output_dir = "media"
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, filename_pdf)
            with open(file_path, "wb") as f_out:
                f_out.write(pdf_buffer.getbuffer())

            # ✅ Bouton basé sur le fichier sauvegardé
            with open(file_path, "rb") as f_in:
                st.download_button(
                    label="Télécharger le rapport PDF",
                    data=f_in,
                    file_name=filename_pdf,
                    mime="application/pdf"
                )
    except Exception as e:
        st.error(f"Erreur lors de la génération du PDF : {e}")
        import traceback; st.code(traceback.format_exc())
