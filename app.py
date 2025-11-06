# -*- coding: utf-8 -*-
# Calculateur CO2 multimodal — version simplifiée (monofichier)
# NILEY EXPERTS — prêt à coller

import os, io, math, re, tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk
from geopy.distance import great_circle

# PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.utils import ImageReader
from xml.sax.saxutils import escape as xml_escape

# Matplotlib (rendu carte PDF)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Cache Cartopy (utile en déploiement)
os.environ.setdefault("CARTOPY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "cartopy_cache"))

# ---------- CONSTANTES ----------
LOGO_URL = "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/NILEY-EXPERTS-logo-removebg-preview.png"

DEFAULT_EMISSION_FACTORS = {
    "Routier": 0.100,
    "Aerien": 0.500,
    "Maritime": 0.015,
    "Ferroviaire": 0.030,
}

MAP_STYLES = {
    "Carto Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    "Carto Positron (clair)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "Carto Dark Matter (sombre)": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
}

PDF_BASEMAP_LABELS = [
    "Identique a la carte Web (Carto)",
    "Auto (Stamen -> OSM -> NaturalEarth)",
    "Stamen Terrain (internet)",
    "OSM (internet)",
    "Natural Earth (vectoriel, offline)",
]
PDF_BASEMAP_MODES = {
    "Identique a la carte Web (Carto)": "carto_web",
    "Auto (Stamen -> OSM -> NaturalEarth)": "auto",
    "Stamen Terrain (internet)": "stamen",
    "OSM (internet)": "osm",
    "Natural Earth (vectoriel, offline)": "naturalearth",
}

MAX_SEGMENTS = 50


# ---------- MODELES ----------
@dataclass
class Location:
    display: str = ""
    coord: Optional[Tuple[float, float]] = None  # (lat, lon)
    iata: str = ""
    unlocode: str = ""
    query: str = ""
    def is_empty(self) -> bool:
        return not self.display or not self.coord

@dataclass
class Segment:
    origin: Location
    dest: Location
    mode: str = "Routier"
    weight_kg: float = 1000.0

def default_segment(mode: Optional[str] = None, weight: float = 1000.0) -> Segment:
    return Segment(Location(), Location(), mode or "Routier", weight)


# ---------- UTILS ----------
def read_secret(key: str, default: str = "") -> str:
    return (st.secrets.get(key) if "secrets" in dir(st) and key in st.secrets else os.getenv(key, default)) or default

def normalize_nodiac(s: str) -> str:
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()

def mode_category(mode_str: str) -> str:
    s = normalize_nodiac(mode_str)
    if any(k in s for k in ("routier", "road", "truck")): return "routier"
    if any(k in s for k in ("aerien", "air", "plane")):   return "aerien"
    if any(k in s for k in ("maritime", "mer", "bateau", "sea", "ship")): return "maritime"
    if any(k in s for k in ("ferroviaire", "rail", "train")): return "ferroviaire"
    return "routier"

def compute_distance_km(coord1, coord2) -> float:
    return great_circle(coord1, coord2).km

def compute_emissions(distance_km: float, weight_tonnes: float, factor: float) -> float:
    return distance_km * weight_tonnes * factor


# ---------- CACHE / SERVICES ----------
@st.cache_data(show_spinner=False, ttl=60*60)
def geocode(query: str, api_key: str, limit: int = 5):
    if not query: return []
    try:
        from opencage.geocoder import OpenCageGeocode
        return OpenCageGeocode(api_key).geocode(query, no_annotations=1, limit=limit) or []
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=24*60*60)
def coords_from_formatted(formatted: str, api_key: str) -> Optional[Tuple[float, float]]:
    try:
        from opencage.geocoder import OpenCageGeocode
        r = OpenCageGeocode(api_key).geocode(formatted, no_annotations=1, limit=1)
        if r:
            g = r[0]["geometry"]
            return (g["lat"], g["lng"])
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False, ttl=6*60*60)
def osrm_route(coord1, coord2, base_url: str = "https://router.project-osrm.org") -> Dict[str, Any]:
    lon1, lat1 = coord1[1], coord1[0]
    lon2, lat2 = coord2[1], coord2[0]
    url = f"{base_url.rstrip('/')}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    r = requests.get(url, params={"overview": "full", "alternatives": "false", "annotations": "false", "geometries": "geojson"}, timeout=12)
    r.raise_for_status()
    routes = r.json().get("routes", [])
    if not routes: raise ValueError("Aucune route retournee par OSRM")
    route = routes[0]
    return {"distance_km": float(route.get("distance", 0.0)) / 1000.0, "coords": route.get("geometry", {}).get("coordinates", [])}

# --- Données IATA / Ports ---
@st.cache_data(show_spinner=False, ttl=7*24*60*60)
def load_airports_iata(path: str = "airport-codes.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.warning(f"Impossible de charger '{path}': {e}")
        return pd.DataFrame(columns=["iata_code","name","municipality","iso_country","lat","lon","label"])
    required = {"iata_code","name","coordinates"}
    if not required.issubset(df.columns):
        st.error("Colonnes manquantes dans airport-codes.csv (requis: iata_code, name, coordinates)")
        return pd.DataFrame(columns=["iata_code","name","municipality","iso_country","lat","lon","label"])
    df = df[df["iata_code"].astype(str).str.len()==3].copy()
    df["iata_code"] = df["iata_code"].astype(str).str.upper()
    coords = df["coordinates"].astype(str).str.replace('"','').str.strip()
    parts = coords.str.split(",", n=1, expand=True)
    if parts.shape[1] < 2:
        parts = pd.DataFrame({0:coords, 1:None})
    df["lat"] = pd.to_numeric(parts[0].astype(str).str.strip(), errors="coerce")
    df["lon"] = pd.to_numeric(parts[1].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["lat","lon"]).copy()
    for col in ["municipality","iso_country","name"]:
        if col not in df.columns: df[col] = ""
        df[col] = df[col].astype(str).replace({"nan":""}).fillna("").str.strip()
    def _label(r):
        base = f"{(r['iata_code'] or '').strip()} - {(r['name'] or 'Sans nom').strip()}"
        extra = " · ".join([p for p in [r["municipality"], r["iso_country"]] if p])
        return f"{base} {extra}" if extra else base
    df["label"] = df.apply(_label, axis=1)
    return df[["iata_code","name","municipality","iso_country","lat","lon","label"]]

@st.cache_data(show_spinner=False, ttl=24*60*60)
def search_airports(query: str, limit: int = 20) -> pd.DataFrame:
    df = load_airports_iata()
    q = (query or "").strip()
    if df.empty: return df
    if not q: return df.head(limit)
    if len(q) <= 3:
        res = df[df["iata_code"].str.startswith(q.upper())]
        if not res.empty: return res.head(limit)
    ql = q.lower()
    res = df[
        df["name"].str.lower().str.contains(ql, na=False)
        | df["municipality"].astype(str).str.lower().str.contains(ql, na=False)
    ]
    return res.head(limit)

@st.cache_data(show_spinner=False, ttl=7*24*60*60)
def load_ports_csv(path: str = "ports.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.warning(f"Impossible de charger '{path}': {e}")
        return pd.DataFrame(columns=["unlocode","name","country","lat","lon","label"])
    cols = {c.lower(): c for c in df.columns}
    def pick(*cand):
        for c in cand:
            if c in cols: return cols[c]
        return None
    col_unlo = pick("unlocode","locode")
    col_name = pick("name","port_name")
    col_ctry = pick("country","iso_country","country_code")
    col_lat  = pick("lat","latitude")
    col_lon  = pick("lon","lng","long","longitude")
    if not col_lat or not col_lon:
        st.error("Colonnes lat/lon manquantes dans ports.csv")
        return pd.DataFrame(columns=["unlocode","name","country","lat","lon","label"])
    out = pd.DataFrame()
    out["unlocode"] = df[col_unlo] if col_unlo else ""
    out["name"] = df[col_name] if col_name else ""
    out["country"] = df[col_ctry] if col_ctry else ""
    out["lat"] = pd.to_numeric(df[col_lat], errors="coerce")
    out["lon"] = pd.to_numeric(df[col_lon], errors="coerce")
    out = out.dropna(subset=["lat","lon"]).copy()
    out["unlocode"] = out["unlocode"].astype(str).str.upper().str.strip()
    out["name"] = out["name"].astype(str).replace({"nan":""}).fillna("").str.strip()
    out["country"] = out["country"].astype(str).replace({"nan":""}).fillna("").str.strip()
    def _label(r):
        base = (r["name"] or "Port sans nom").strip()
        extras = " · ".join([p for p in [r["unlocode"], r["country"]] if p])
        return f"{base} {extras}" if extras else base
    out["label"] = out.apply(_label, axis=1)
    return out[["unlocode","name","country","lat","lon","label"]]

@st.cache_data(show_spinner=False, ttl=24*60*60)
def search_ports(query: str, limit: int = 12) -> pd.DataFrame:
    df = load_ports_csv()
    q = (query or "").strip()
    if df.empty: return df
    if not q: return df.head(limit)
    # UN/LOCODE direct
    if len(q) in (5,6):
        res = df[df["unlocode"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
                 .str.startswith(re.sub(r"[^A-Za-z0-9]", "", q.upper()))]
        if not res.empty: return res.head(limit)
    ql = q.lower()
    res = df[
        df["name"].str.lower().str.contains(ql, na=False)
        | df["country"].str.lower().str.contains(ql, na=False)
        | df["unlocode"].str.lower().str.contains(ql, na=False)
    ]
    return res.head(limit)


# ---------- SESSION STATE ----------
def ss_init():
    if "segments" not in st.session_state or not st.session_state.segments:
        st.session_state.segments = [default_segment()]

def ss_get_segment(i: int) -> Segment:
    return st.session_state.segments[i]

def ss_set_segment(i: int, seg: Segment):
    st.session_state.segments[i] = seg

def add_segment():
    if len(st.session_state.segments) >= MAX_SEGMENTS:
        st.warning("Nombre maximum de segments atteint."); return
    prev = st.session_state.segments[-1]
    nxt = default_segment(prev.mode, prev.weight_kg)
    if not prev.dest.is_empty():
        nxt.origin = Location(display=prev.dest.display, coord=prev.dest.coord, iata=prev.dest.iata, query=prev.dest.display)
    st.session_state.segments.append(nxt)

def remove_last_segment():
    if len(st.session_state.segments) <= 1:
        st.info("Au moins un segment doit rester. Réinitialisation.")
        st.session_state.segments = [default_segment()]
    else:
        st.session_state.segments.pop()

def reset_segments():
    st.session_state.segments = [default_segment()]
    st.session_state.pop("dossier_transport", None)
    st.rerun()


# ---------- UI WIDGETS ----------
def location_input(label_prefix: str, loc: Location, api_key: str, show_airports: bool, show_ports_opt: bool = True) -> Location:
    q = st.text_input(f"{label_prefix} — Adresse / Ville / Pays ou IATA", value=loc.query, key=f"in_{label_prefix}_{id(loc)}")
    loc.query = q
    q_raw = (q or "").strip()
    is_iata = bool(re.fullmatch(r"[A-Za-z]{3}", q_raw))
    q_iata = q_raw.upper() if is_iata else None

    airports = pd.DataFrame()
    ports = pd.DataFrame()
    oc_opts = []

    if q_raw:
        if show_airports or is_iata:
            airports = search_airports(q_raw, limit=20)
            if is_iata and not airports.empty:
                exact = airports[airports["iata_code"].str.upper() == q_iata]
                others = airports[airports["iata_code"].str.upper() != q_iata]
                airports = pd.concat([exact, others], ignore_index=True)
        if show_ports_opt and not is_iata:
            ports = search_ports(q_raw, limit=12)
        if not is_iata:
            oc = geocode(q_raw, api_key=api_key, limit=5)
            oc_opts = [r["formatted"] for r in oc] if oc else []

    options, airport_rows, port_rows = [], [], []
    if (show_airports or is_iata) and not airports.empty:
        for _, r in airports.iterrows():
            options.append(f"✈️ {r['label']} (IATA {r['iata_code']})")
            airport_rows.append(r)
    if show_ports_opt and not is_iata and not ports.empty:
        for _, r in ports.iterrows():
            suffix = f" (UN/LOCODE {r['unlocode']})" if r.get("unlocode") else ""
            options.append(f"⚓ {r['label']}{suffix}")
            port_rows.append(r)
    if not is_iata and oc_opts:
        options += [f"📍 {o}" for o in oc_opts]
    if not options: options = ["— Aucun résultat —"]

    sel = st.selectbox("Résultats", options, key=f"sel_{label_prefix}_{id(loc)}")
    if sel != "— Aucun résultat —":
        if sel.startswith("✈️"):
            idx_global = options.index(sel)
            r = airport_rows[idx_global] if 0 <= idx_global < len(airport_rows) else airports.iloc[0]
            loc.coord = (float(r["lat"]), float(r["lon"]))
            loc.display = str(r["label"])
            loc.iata, loc.unlocode = str(r["iata_code"]), ""
        elif sel.startswith("⚓"):
            idx_global = options.index(sel) - len(airport_rows)
            r = port_rows[idx_global] if 0 <= idx_global < len(port_rows) else ports.iloc[0]
            loc.coord = (float(r["lat"]), float(r["lon"]))
            loc.display = str(r["label"])
            loc.unlocode, loc.iata = str(r.get("unlocode") or ""), ""
        else:
            formatted = sel[2:].strip() if sel.startswith("📍") else sel
            loc.coord = coords_from_formatted(formatted, api_key=api_key)
            loc.display = formatted
            loc.iata = loc.unlocode = ""
    return loc

def segment_block(i: int, api_key: str):
    seg = ss_get_segment(i)
    st.markdown(f"#### Segment {i + 1}")
    c1, c2 = st.columns(2)
    with c1:
        seg.origin = location_input("Origine", seg.origin, api_key, show_airports=("aerien" in normalize_nodiac(seg.mode)))
    with c2:
        seg.dest = location_input("Destination", seg.dest, api_key, show_airports=False)

    modes = ["Routier", "Maritime", "Ferroviaire", "Aerien"]
    seg.mode = st.selectbox("Mode de transport", modes, index=modes.index(seg.mode) if seg.mode in modes else 0, key=f"mode_{i}")

    if i == 0:
        seg.weight_kg = st.number_input("Poids transporté (kg, appliqué à tous les segments)", min_value=0.001, value=float(seg.weight_kg), step=100.0, key=f"w_{i}")
    else:
        seg.weight_kg = ss_get_segment(0).weight_kg
    ss_set_segment(i, seg)


# ---------- CALCUL ----------
def compute_rows(segments: List[Segment]) -> Tuple[pd.DataFrame, List[Dict[str, Any]], float, float]:
    rows, total_d, total_e = [], 0.0, 0.0
    factors = DEFAULT_EMISSION_FACTORS
    for idx, seg in enumerate(segments, start=1):
        if seg.origin.is_empty() or seg.dest.is_empty(): continue
        c1, c2 = seg.origin.coord, seg.dest.coord
        route_coords = None
        try:
            if "routier" in normalize_nodiac(seg.mode):
                r = osrm_route(c1, c2)
                dkm, route_coords = r["distance_km"], r["coords"]
            else:
                dkm = compute_distance_km(c1, c2)
        except Exception:
            dkm = compute_distance_km(c1, c2)
        factor = float(factors.get(seg.mode, 0.0))
        emis = compute_emissions(dkm, seg.weight_kg/1000.0, factor)
        total_d += dkm; total_e += emis
        rows.append({
            "Segment": idx, "Origine": seg.origin.display, "Destination": seg.dest.display, "Mode": seg.mode,
            "Distance (km)": round(dkm,1), "Poids (kg)": round(seg.weight_kg,1), "Facteur (kg CO2e/t.km)": factor,
            "Emissions (kg CO2e)": round(emis,2),
            "lat_o": c1[0], "lon_o": c1[1], "lat_d": c2[0], "lon_d": c2[1], "route_coords": route_coords
        })
    df = pd.DataFrame(rows)
    return df, rows, total_d, total_e


# ---------- PDF : fond de carte Cartopy + tracés ----------
def _compute_extent(all_lats, all_lons, margin_ratio=0.12, min_span_deg=1e-3):
    if not all_lats or not all_lons: return (-10, 30, 30, 60)
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    span_lat = max(max_lat - min_lat, min_span_deg)
    span_lon = max(max_lon - min_lon, min_span_deg)
    min_lat -= span_lat * margin_ratio; max_lat += span_lat * margin_ratio
    min_lon -= span_lon * margin_ratio; max_lon += span_lon * margin_ratio
    return (min_lon, max_lon, min_lat, max_lat)

def _fit_aspect(min_lon, max_lon, min_lat, max_lat, aspect_w_over_h):
    span_lon = max(1e-6, max_lon - min_lon)
    span_lat = max(1e-6, max_lat - min_lat)
    mid_lat = (min_lat + max_lat)/2.0
    cos_mid = max(0.05, math.cos(math.radians(mid_lat)))
    aspect_geo = (span_lon * cos_mid) / span_lat
    target = max(1e-6, float(aspect_w_over_h))
    if aspect_geo < target:
        needed_lon = (target * span_lat) / cos_mid
        extra = (needed_lon - span_lon)/2.0
        min_lon -= extra; max_lon += extra
    else:
        needed_lat = (span_lon * cos_mid) / target
        extra = (needed_lat - span_lat)/2.0
        min_lat -= extra; max_lat += extra
    min_lon = max(-180.0, min_lon); max_lon = min(180.0, max_lon)
    min_lat = max(-90.0, min_lat);  max_lat = min(90.0,  max_lat)
    return (min_lon, max_lon, min_lat, max_lat)

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
    subdomains = ["a","b","c","d"]
    template = url_by_label.get(web_style_label, url_by_label["Carto Positron (clair)"])
    class CartoTiles(cimgt.GoogleTiles):
        def _image_url(self, tile):
            z, x, y = tile
            s = subdomains[(x + y) % len(subdomains)]
            return template.format(s=s, z=z, x=x, y=y)
    return CartoTiles()

def _choose_zoom(min_lon, max_lon, min_lat, max_lat, max_zoom=9):
    span_lon = max_lon - min_lon
    span_lat = max_lat - min_lat
    span = max(span_lon, span_lat)
    if   span <= 0.5: z=9
    elif span <= 1.0: z=8
    elif span <= 2.0: z=7
    elif span <= 5.0: z=6
    elif span <=12.0: z=5
    elif span <=24.0: z=4
    else:             z=3
    return min(z, max_zoom)

def render_map_png(rows: List[Dict[str,Any]], width_px: int, height_px: int, dpi: int,
                   pdf_basemap_label: str, web_map_style_label: Optional[str], ne_scale: str="50m") -> Optional[io.BytesIO]:
    if not rows: return None
    all_lats = [r["lat_o"] for r in rows] + [r["lat_d"] for r in rows]
    all_lons = [r["lon_o"] for r in rows] + [r["lon_d"] for r in rows]
    min_lon, max_lon, min_lat, max_lat = _compute_extent(all_lats, all_lons)
    min_lon, max_lon, min_lat, max_lat = _fit_aspect(min_lon, max_lon, min_lat, max_lat, aspect_w_over_h=(max(1, width_px)/max(1, height_px)))

    mode = PDF_BASEMAP_MODES.get(pdf_basemap_label, "auto")
    use_cartopy = True
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.io.img_tiles import Stamen, OSM
    except Exception:
        use_cartopy = False

    fig_w_in, fig_h_in = width_px/float(dpi), height_px/float(dpi)
    try:
        if use_cartopy:
            fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
            ax = None; raster_ok = False
            try:
                if mode == "carto_web" and web_map_style_label:
                    tiler = _carto_tiler_from_web_style(web_map_style_label)
                    if tiler is not None:
                        ax = plt.axes(projection=tiler.crs)
                        ax.set_extent((min_lon,max_lon,min_lat,max_lat), crs=ccrs.PlateCarree())
                        ax.add_image(tiler, _choose_zoom(min_lon,max_lon,min_lat,max_lat, max_zoom=9))
                        raster_ok = True
                if not raster_ok and mode in ("auto","stamen"):
                    tiler = Stamen('terrain-background')
                    ax = plt.axes(projection=tiler.crs)
                    ax.set_extent((min_lon,max_lon,min_lat,max_lat), crs=ccrs.PlateCarree())
                    ax.add_image(tiler, _choose_zoom(min_lon,max_lon,min_lat,max_lat, max_zoom=9))
                    raster_ok = True
                if not raster_ok and mode in ("auto","osm"):
                    tiler = OSM()
                    ax = plt.axes(projection=tiler.crs)
                    ax.set_extent((min_lon,max_lon,min_lat,max_lat), crs=ccrs.PlateCarree())
                    ax.add_image(tiler, _choose_zoom(min_lon,max_lon,min_lat,max_lat, max_zoom=9))
                    raster_ok = True
            except Exception:
                raster_ok = False

            if not raster_ok:
                ax = plt.axes(projection=ccrs.PlateCarree())
                # Natural Earth vector (offline)
                colors_cfg = {'ocean':'#EAF4FF','land':'#F7F5F2','lakes_fc':'#EAF4FF','lakes_ec':'#B3D4F5','coast':'#818892','borders0':'#8F98A3'}
                try: ax.add_feature(cfeature.OCEAN.with_scale(ne_scale), facecolor=colors_cfg['ocean'], edgecolor='none', zorder=0)
                except Exception: pass
                try: ax.add_feature(cfeature.LAND.with_scale(ne_scale), facecolor=colors_cfg['land'], edgecolor='none', zorder=0)
                except Exception: pass
                try: ax.add_feature(cfeature.LAKES.with_scale(ne_scale), facecolor=colors_cfg['lakes_fc'], edgecolor=colors_cfg['lakes_ec'], linewidth=0.3, zorder=1)
                except Exception: pass
                try: ax.add_feature(cfeature.COASTLINE.with_scale(ne_scale), edgecolor=colors_cfg['coast'], linewidth=0.4, zorder=2)
                except Exception: pass
                try: ax.add_feature(cfeature.BORDERS.with_scale(ne_scale), edgecolor=colors_cfg['borders0'], linewidth=0.5, zorder=2)
                except Exception: pass
                ax.set_extent((min_lon,max_lon,min_lat,max_lat), crs=ccrs.PlateCarree())

            # Tracés
            mode_colors = {"routier":"#0066CC","aerien":"#CC0000","maritime":"#009900","ferroviaire":"#9900CC"}
            for r in rows:
                cat = mode_category(r["Mode"])
                color = mode_colors.get(cat, "#666666")
                ax.plot([r["lon_o"], r["lon_d"]], [r["lat_o"], r["lat_d"]], color=color, linewidth=2.0, alpha=0.9, transform=ccrs.PlateCarree(), zorder=3)
                ax.scatter([r["lon_o"]],[r["lat_o"]], s=22, c="#0A84FF", edgecolors='white', linewidths=0.8, transform=ccrs.PlateCarree(), zorder=4)
                ax.scatter([r["lon_d"]],[r["lat_d"]], s=22, c="#FF3B30", edgecolors='white', linewidths=0.8, transform=ccrs.PlateCarree(), zorder=4)

            out = io.BytesIO()
            plt.tight_layout()
            plt.savefig(out, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            out.seek(0)
            return out
        else:
            # Fallback sans cartopy : quadrillage simple
            fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
            ax.set_facecolor('#F7F8FA')
            ax.set_xlim(min_lon, max_lon); ax.set_ylim(min_lat, max_lat)
            lat_step = (max_lat-min_lat)/6.0 if (max_lat-min_lat)>0 else 1
            lon_step = (max_lon-min_lon)/8.0 if (max_lon-min_lon)>0 else 1
            for yy in np.arange(min_lat, max_lat+1e-9, lat_step): ax.plot([min_lon, max_lon],[yy,yy], color='#E6E9EF', lw=0.6)
            for xx in np.arange(min_lon, max_lon+1e-9, lon_step): ax.plot([xx,xx],[min_lat,max_lat], color='#E6E9EF', lw=0.6)
            mode_colors = {"routier":"#0066CC","aerien":"#CC0000","maritime":"#009900","ferroviaire":"#9900CC"}
            for r in rows:
                cat = mode_category(r["Mode"]); color = mode_colors.get(cat, "#666666")
                ax.plot([r["lon_o"], r["lon_d"]], [r["lat_o"], r["lat_d"]], color=color, lw=2.0, alpha=0.9)
                ax.scatter([r["lon_o"]],[r["lat_o"]], s=22, c="#0A84FF", edgecolor='white', lw=0.8)
                ax.scatter([r["lon_d"]],[r["lat_d"]], s=22, c="#FF3B30", edgecolor='white', lw=0.8)
            out = io.BytesIO()
            plt.tight_layout()
            plt.savefig(out, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            out.seek(0)
            return out
    except Exception:
        return None

def generate_pdf_report(df: pd.DataFrame, dossier_val: str, total_distance: float, total_emissions: float, unit: str,
                        rows: List[Dict[str,Any]], pdf_basemap_choice_label: str, web_map_style_label: Optional[str],
                        detail_params: Dict[str,int]):
    """Résumé + carte (Cartopy si dispo) + tableau. Retourne BytesIO PDF."""
    from reportlab.pdfgen import canvas as pdfcanvas
    PAGE_W, PAGE_H = landscape(A4)
    M = 1.0 * cm
    AVAIL_W, AVAIL_H = PAGE_W - 2*M, PAGE_H - 2*M
    buf = io.BytesIO()
    c = pdfcanvas.Canvas(buf, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=14, textColor=colors.HexColor('#1f4788'), alignment=1)
    normal = styles['Normal']

    # Titre + logo
    y = PAGE_H - M
    try:
        content = requests.get(LOGO_URL, timeout=6).content
        img = ImageReader(io.BytesIO(content))
        c.drawImage(img, M, y-1.5*cm, width=3*cm, height=1.5*cm, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass
    title_para = Paragraph("RAPPORT D'EMPREINTE Co2 TRANSPORT", title_style)
    tw, th = title_para.wrap(AVAIL_W - 3.5*cm, AVAIL_H)
    title_para.drawOn(c, M + 3.5*cm, y - 1.2*cm)
    y -= 2.0*cm

    # Résumé
    info = [
        ["N° dossier Transport:", dossier_val, "Distance totale:", f"{total_distance:.1f} km"],
        ["Date du rapport:", pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"), "Emissions totales:", f"{total_emissions:.2f} kg CO2e"],
        ["Nombre de segments:", str(len(rows)), "Emissions moyennes:", f"{(total_emissions/total_distance):.3f} kg CO2e/km" if total_distance>0 else "N/A"],
    ]
    t = Table(info, colWidths=[4.5*cm, 5.5*cm, 4.5*cm, 5.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(0,-1), colors.HexColor('#e8f0f7')),
        ('BACKGROUND', (2,0),(2,-1), colors.HexColor('#fff4e6')),
        ('GRID', (0,0),(-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0),(-1,-1), 8),
        ('VALIGN', (0,0),(-1,-1), 'MIDDLE'),
    ]))
    tw, th = t.wrap(AVAIL_W, AVAIL_H); t.drawOn(c, M, y-th); y -= th + 0.25*cm

    # Carte
    footer_h = 0.7*cm
    min_table_h = 6.0*cm
    max_map_h = 7.5*cm
    remaining_h = (y - M) - footer_h
    map_h = max(4.0*cm, min(max_map_h, remaining_h * 0.45))
    table_h_avail = remaining_h - map_h - 0.25*cm
    if table_h_avail < min_table_h:
        delta = (min_table_h - table_h_avail)
        map_h = max(4.0*cm, map_h - delta)
        table_h_avail = (y - M) - footer_h - map_h - 0.25*cm

    dpi = int(detail_params.get("dpi", 220))
    map_png = render_map_png(rows, width_px=int((AVAIL_W/72.0)*dpi), height_px=int((map_h/72.0)*dpi), dpi=dpi,
                             pdf_basemap_label=pdf_basemap_choice_label, web_map_style_label=web_map_style_label, ne_scale="50m")
    if map_png:
        img = ImageReader(map_png)
        c.drawImage(img, M, y - map_h, width=AVAIL_W, height=map_h, preserveAspectRatio=True, mask='auto')
        y -= map_h + 0.25*cm

    # Tableau segments
    headers = ["Seg.", "Origine", "Destination", "Mode", "Dist.\n(km)", f"Poids\n({unit})", "Facteur\n(kg CO2e/t.km)", "Emissions\n(kg CO2e)"]
    data_rows = []
    for _, r in df.iterrows():
        data_rows.append([
            str(r["Segment"]),
            Paragraph(xml_escape(str(r["Origine"] or "")), normal),
            Paragraph(xml_escape(str(r["Destination"] or "")), normal),
            str(r["Mode"]),
            f"{r['Distance (km)']:.1f}",
            f"{r['Poids (kg)']:.0f}",
            f"{r['Facteur (kg CO2e/t.km)']:.3f}",
            f"{r['Emissions (kg CO2e)']:.2f}",
        ])
    total_row = ["TOTAL","","","", f"{total_distance:.1f}","","", f"{total_emissions:.2f}"]

    col_widths = [1.2*cm, 4.8*cm, 4.8*cm, 3.0*cm, 1.8*cm, 1.8*cm, 2.2*cm, 2.2*cm]
    tbl = Table([headers] + data_rows + [total_row], colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0,0),(-1,0), colors.whitesmoke),
        ('GRID', (0,0),(-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0),(-1,-1), 8),
        ('VALIGN', (0,0),(-1,-1), 'TOP'),
        ('BACKGROUND', (0,-1),(-1,-1), colors.HexColor('#fff4e6')),
    ]))
    tw, th = tbl.wrap(AVAIL_W, table_h_avail)
    tbl.drawOn(c, M, y - th); y -= th

    # Footer
    footer = Paragraph(f"Document généré le {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')} — Calculateur CO2 multimodal",
                       ParagraphStyle('F', parent=normal, fontSize=7, textColor=colors.grey, alignment=1))
    fw, fh = footer.wrap(AVAIL_W, 1*cm)
    footer.drawOn(c, M + (AVAIL_W - fw)/2.0, M - 0.1*cm)

    c.showPage(); c.save(); buf.seek(0)
    return buf


# ---------- APP ----------
st.set_page_config(page_title="Calculateur CO2 multimodal", page_icon="🌍", layout="centered")

# Auth
if "APP_PASSWORD" not in st.secrets:
    st.error("Mot de passe non configuré (APP_PASSWORD)."); st.stop()
if "auth_ok" not in st.session_state: st.session_state.auth_ok = False
if not st.session_state.auth_ok:
    st.header("Accès sécurisé")
    with st.form("login", clear_on_submit=True):
        pwd = st.text_input("Mot de passe", type="password")
        if st.form_submit_button("Valider"):
            if pwd == st.secrets["APP_PASSWORD"]:
                st.session_state.auth_ok = True; st.rerun()
            else:
                st.error("Mot de passe incorrect.")
    st.stop()

# API key
API_KEY = read_secret("OPENCAGE_KEY")
if not API_KEY:
    st.error("Clé API OpenCage absente (OPENCAGE_KEY)."); st.stop()

# UI
st.image(LOGO_URL, width=520)
st.subheader("Calculateur d'empreinte CO2 multimodal")
ss_init()

col_title, col_reset = st.columns([8,2])
with col_title: st.subheader("Saisie des segments")
with col_reset: st.button("Réinitialiser", type="secondary", use_container_width=True, on_click=reset_segments)

# Segments
for i in range(len(st.session_state.segments)):
    with st.container(border=True):
        segment_block(i, API_KEY)

# Add / Remove
c_add, c_del = st.columns(2)
with c_add: st.button("Ajouter un segment", use_container_width=True, on_click=add_segment)
with c_del: st.button("Supprimer le dernier segment", use_container_width=True, on_click=remove_last_segment)

# Carte web
st.subheader("Carte interactive")
c1, c2, c3 = st.columns([3,2,3])
with c1:
    map_style_label = st.selectbox("Fond de carte (Web)", list(MAP_STYLES), index=0)
with c2:
    focus_choices = ["— Tous —"] + [f"Segment {i+1}" for i in range(len(st.session_state.segments))]
    focus_sel = st.selectbox("Focus segment", focus_choices, index=0)
with c3:
    show_icons = st.checkbox("Afficher les icônes mode", value=True)  # placeholder (icônes non utilisées dans ce monofichier)

# Calcul
st.subheader("Calcul")
dossier_transport = st.text_input("N° dossier Transport (obligatoire) *", value=st.session_state.get("dossier_transport",""), placeholder="ex: TR-2025-001")
st.session_state["dossier_transport"] = dossier_transport.strip()

if st.button("Calculer l'empreinte carbone totale", disabled=not bool(dossier_transport.strip())):
    segs: List[Segment] = st.session_state.segments
    df, rows, tot_d, tot_e = compute_rows(segs)
    if df.empty:
        st.info("Aucun segment valide. Vérifiez les entrées.")
    else:
        st.success(f"{len(df)} segment(s) — Distance totale: {tot_d:.1f} km — Emissions: {tot_e:.2f} kg CO2e")
        st.dataframe(df[["Segment","Origine","Destination","Mode","Distance (km)","Poids (kg)","Facteur (kg CO2e/t.km)","Emissions (kg CO2e)"]], use_container_width=True)

        # Deck.gl (routes OSRM + lignes droites + points)
        layers = []
        route_paths = [{"path": r["route_coords"], "name": f"Segment {r['Segment']} - {r['Mode']}"} for r in rows if r.get("route_coords")]
        if route_paths:
            layers.append(pdk.Layer("PathLayer", data=route_paths, get_path="path", get_color=[187,147,87,220], width_scale=1, width_min_pixels=4, pickable=True))
        straight = [{"from":[r["lon_o"],r["lat_o"]], "to":[r["lon_d"],r["lat_d"]], "name": f"Segment {r['Segment']} - {r['Mode']}"} for r in rows if not r.get("route_coords")]
        if straight:
            layers.append(pdk.Layer("LineLayer", data=straight, get_source_position="from", get_target_position="to", get_width=3, get_color=[120,120,120,160], pickable=True))
        pts = []
        for r in rows:
            pts += [
                {"position":[r["lon_o"], r["lat_o"]], "name":f"S{r['Segment']} - Origine", "color":[0,122,255,220]},
                {"position":[r["lon_d"], r["lat_d"]], "name":f"S{r['Segment']} - Destination", "color":[220,66,66,220]},
            ]
        if pts:
            layers.append(pdk.Layer("ScatterplotLayer", data=pts, get_position="position", get_fill_color="color",
                                    get_radius=20000, radius_min_pixels=2, radius_max_pixels=60, pickable=True,
                                    stroked=True, get_line_color=[255,255,255], line_width_min_pixels=1))
        # Vue auto
        def view_for_all(rrs):
            if not rrs: return pdk.ViewState(latitude=48.85, longitude=2.35, zoom=3)
            lats = [v for r in rrs for v in [r["lat_o"], r["lat_d"]]]
            lons = [v for r in rrs for v in [r["lon_o"], r["lon_d"]]]
            mid_lat, mid_lon = (min(lats)+max(lats))/2, (min(lons)+max(lons))/2
            span_lat = max(1e-6, max(lats)-min(lats))
            span_lon = max(1e-6, max(lons)-min(lons)) * max(0.1, math.cos(math.radians(mid_lat)))
            zoom = max(1.0, min(15.0, min(math.log2(360.0/span_lon), math.log2(180.0/span_lat))))
            return pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=float(zoom))
        view = view_for_all(rows)
        st.pydeck_chart(pdk.Deck(map_style=MAP_STYLES[map_style_label], initial_view_state=view, layers=layers, tooltip={"text":"{name}"}))

        # Exports
        df_export = df.drop(columns=["lat_o","lon_o","lat_d","lon_d","route_coords"]).copy()
        df_export.insert(0, "N° dossier Transport", dossier_transport)
        csv = df_export.to_csv(index=False).encode("utf-8")
        safe_suffix = "".join(c if (c.isalnum() or c in "-_") else "_" for c in dossier_transport.strip())
        fn_csv = f"resultats_co2_multimodal_{safe_suffix}.csv" if safe_suffix else "resultats_co2_multimodal.csv"
        fn_pdf = f"rapport_co2_multimodal_{safe_suffix}.pdf" if safe_suffix else "rapport_co2_multimodal.pdf"

        st.subheader("Exporter")
        detail_levels = {
            "Standard (léger, rapide)": {"dpi": 180, "max_zoom": 7},
            "Détaillé (équilibre)": {"dpi": 220, "max_zoom": 9},
            "Ultra (fin mais plus lent)": {"dpi": 280, "max_zoom": 10},
        }
        pdf_base_choice = st.selectbox("Fond de carte du PDF", options=PDF_BASEMAP_LABELS, index=0, help="Identique a la carte Web utilise le style Carto sélectionné.")
        quality_label = st.selectbox("Qualité de rendu PDF", options=list(detail_levels.keys()), index=1)
        detail_params = detail_levels[quality_label]

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.download_button("Télécharger le détail (CSV)", data=csv, file_name=fn_csv, mime="text/csv")
        with col_e2:
            try:
                pdf_buf = generate_pdf_report(df, dossier_transport, tot_d, tot_e, "kg", rows,
                                              pdf_basemap_choice_label=pdf_base_choice,
                                              web_map_style_label=map_style_label, detail_params=detail_params)
                st.download_button("Télécharger le rapport PDF", data=pdf_buf, file_name=fn_pdf, mime="application/pdf")
            except Exception as e:
                st.error(f"Erreur PDF : {e}")
