"""PathSnap evaluator UI (Laptop 2).

Run from the repo root:
    API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501

The UI only talks to the API. It never reads trip files and never receives
the reference route: the map shows raw GPS points and the predicted route only.
"""
import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT_S = 120
MAX_POINTS_ON_MAP = 5000  # draw at most this many GPS markers; long trips are thinned evenly

GPS_COLOUR = "#2d68cf"
ROUTE_COLOUR = "#e4572e"

st.set_page_config(page_title="PathSnap", layout="wide")


# ---------- API calls ----------

def api_get(base: str, path: str) -> dict:
    r = requests.get(f"{base}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def api_predict(base: str, trip_id: str) -> dict:
    r = requests.post(f"{base}/predict", json={"trip_id": trip_id}, timeout=TIMEOUT_S)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"API returned {r.status_code}: {detail}")
    return r.json()


# ---------- map ----------

def build_map(result: dict) -> folium.Map:
    obs = result["obs"]
    lats = [o["lat"] for o in obs]
    lons = [o["lon"] for o in obs]

    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))

    gps = folium.FeatureGroup(name="GPS observations (raw)", show=True)
    step = max(1, len(obs) // MAX_POINTS_ON_MAP)
    for o in obs[::step]:
        folium.CircleMarker(
            location=[o["lat"], o["lon"]], radius=2, color=GPS_COLOUR, weight=0,
            fill=True, fill_color=GPS_COLOUR, fill_opacity=0.7,
            tooltip=f"obs {o['obs_idx']} · t={o['t']:.0f}s",
        ).add_to(gps)
    gps.add_to(m)

    route = folium.FeatureGroup(name="Predicted route", show=True)
    folium.GeoJson(
        result["geojson"],
        style_function=lambda _: {"color": ROUTE_COLOUR, "weight": 5, "opacity": 0.85},
        tooltip=folium.GeoJsonTooltip(fields=["order", "arc_id"], aliases=["Step", "Arc"]),
    ).add_to(route)
    route.add_to(m)

    ends = folium.FeatureGroup(name="Start / end", show=True)
    folium.Marker([obs[0]["lat"], obs[0]["lon"]], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(ends)
    folium.Marker([obs[-1]["lat"], obs[-1]["lon"]], tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(ends)
    ends.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


# ---------- page ----------

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API address", value=DEFAULT_API_URL, key="api_url").rstrip("/")
    try:
        health = api_get(api_url, "/health")
        st.success(f"Connected to {api_url}")
        st.caption(f"Method: **{health['method']}**")
        st.caption(f"Model loaded: {'yes' if health['model_loaded'] else 'no (baseline)'}")
        connected = True
    except requests.RequestException as exc:
        st.error(f"Cannot reach the API at {api_url}. Check that Laptop 1 is running uvicorn "
                 f"with --host 0.0.0.0 and that both laptops are on the same network.")
        st.caption(f"Details: {exc.__class__.__name__}")
        connected = False

st.title("PathSnap")
st.caption("Map-matching noisy GPS trajectories to the road network")

if not connected:
    st.stop()

try:
    trips = api_get(api_url, "/trips")["trips"]
except requests.RequestException as exc:
    st.error(f"Could not load the trip list: {exc}")
    st.stop()

if not trips:
    st.warning("The API has no trips to serve. Check EVAL_DIR on Laptop 1.")
    st.stop()

labels = {t["trip_id"]: f"{t['trip_id']}  ({t['n_points']:,} GPS points)" for t in trips}
col_pick, col_run = st.columns([3, 1], vertical_alignment="bottom")
with col_pick:
    trip_id = st.selectbox("Evaluation trip", options=list(labels), format_func=labels.get, key="trip_id")
with col_run:
    run = st.button("Run prediction", type="primary", use_container_width=True, key="run")

if run:
    with st.spinner(f"Running the pipeline on trip {trip_id}…"):
        try:
            st.session_state["result"] = api_predict(api_url, trip_id)
            st.session_state.pop("error", None)
        except (requests.RequestException, RuntimeError) as exc:
            st.session_state.pop("result", None)
            st.session_state["error"] = str(exc)

if "error" in st.session_state:
    st.error(st.session_state["error"])

result = st.session_state.get("result")
if result is None:
    st.info("Pick a trip and press **Run prediction**.")
    st.stop()

metrics = result["metrics"]
latency = result["latency_ms"]
st.subheader(f"Trip {result['trip_id']}")
st.caption(f"Predicted by: **{result['method']}**")

cols = st.columns(6)
cols[0].metric("Top-1 accuracy", pct(metrics["top1_accuracy"]))
cols[1].metric("Route F1", pct(metrics["route_f1"]))
cols[2].metric("Route precision", pct(metrics["route_precision"]))
cols[3].metric("Route recall", pct(metrics["route_recall"]))
cols[4].metric("Continuity", pct(metrics["continuity"]))
cols[5].metric("Total time", f"{latency['total'] / 1000:.2f} s")

st.markdown(
    f"<span style='color:{GPS_COLOUR}'>●</span> Raw GPS observations ({len(result['obs']):,}) &nbsp;&nbsp; "
    f"<span style='color:{ROUTE_COLOUR}'>━━</span> Predicted route ({len(result['predicted_arcs']):,} arcs) &nbsp;&nbsp; "
    f"🟢 Start &nbsp; 🔴 End",
    unsafe_allow_html=True,
)
st_folium(build_map(result), height=620, use_container_width=True, returned_objects=[], key="map")

with st.expander("Timing per stage"):
    st.table({stage: f"{ms:.0f} ms" for stage, ms in latency.items()})
st.caption("Metrics are computed on the API server against its reference data. "
           "The reference route is never sent to this interface.")
