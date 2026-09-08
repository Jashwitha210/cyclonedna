import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

from src.data_load import (
    load_data,
    get_named_cyclones,
    get_cyclone
)

from src.sate import (
    get_satellite_image,
    get_sst
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CycloneDNA AI",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main { background-color: #f4f7fb; }

    .source-box {
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #d9e2ec;
        background: white;
        margin-bottom: 15px;
    }

    .source-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .dna-box {
        padding: 22px;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827, #243b64);
        color: white;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .alert-high {
        padding: 18px;
        border-radius: 15px;
        background: #ffe5e5;
        border: 2px solid #e05252;
        margin: 10px 0;
    }

    .alert-medium {
        padding: 18px;
        border-radius: 15px;
        background: #fff4d6;
        border: 2px solid #e0a52f;
        margin: 10px 0;
    }

    .alert-low {
        padding: 18px;
        border-radius: 15px;
        background: #e8f5e9;
        border: 2px solid #62a968;
        margin: 10px 0;
    }

    .feature-card {
        padding: 12px;
        border-radius: 12px;
        background: #ffffff;
        border: 1px solid #d9e2ec;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🧬 CycloneDNA AI")

st.write(
    "Explainable Multi-Source Tropical Cyclone "
    "Behaviour Intelligence System"
)

st.caption(
    "Satellite + Ocean + Atmospheric observations → "
    "Feature Extraction → Fusion → Alerts & Suggestions"
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_backend():
    return load_data()


df = load_backend()
cyclones = get_named_cyclones(df).copy()

if cyclones.empty:
    st.error("No named cyclones were found in the backend dataset.")
    st.stop()

cyclones["LABEL"] = (
    cyclones["NAME"].astype(str)
    + " — "
    + cyclones["SEASON"].astype(str)
)

selected_label = st.selectbox(
    "Choose a cyclone for analysis",
    cyclones["LABEL"].tolist()
)

selected_row = cyclones[
    cyclones["LABEL"] == selected_label
].iloc[0]

selected_sid = selected_row["SID"]
selected_name = selected_row["NAME"]
selected_year = selected_row["SEASON"]


# ============================================================
# SELECTED CYCLONE DATA
# ============================================================

storm = get_cyclone(df, selected_sid).copy()

numeric_columns = [
    "LAT",
    "LON",
    "USA_WIND",
    "USA_PRES",
    "STORM_SPEED",
    "STORM_DIR"
]

for column in numeric_columns:
    if column in storm.columns:
        storm[column] = pd.to_numeric(
            storm[column],
            errors="coerce"
        )

storm["ISO_TIME"] = pd.to_datetime(
    storm["ISO_TIME"],
    errors="coerce"
)

storm = storm.sort_values("ISO_TIME")

valid_storm = storm.dropna(
    subset=["LAT", "LON", "USA_WIND", "USA_PRES"]
)

if valid_storm.empty:
    st.error("No valid observations available.")
    st.stop()

latest = valid_storm.iloc[-1]

latest_date = latest["ISO_TIME"]
latest_lat = float(latest["LAT"])
latest_lon = float(latest["LON"])


# ============================================================
# ATMOSPHERIC FEATURE EXTRACTION
# ============================================================

storm["WIND_CHANGE"] = storm["USA_WIND"].diff()
storm["PRESSURE_CHANGE"] = storm["USA_PRES"].diff()

# IBTrACS observations are commonly 3/6-hourly. This is a
# normalized change indicator rather than a physical forecast.
storm["WIND_INTENSIFICATION"] = storm["WIND_CHANGE"] / 3.0


def classify_behaviour(row):
    wind_change = row["WIND_INTENSIFICATION"]
    pressure_change = row["PRESSURE_CHANGE"]

    if pd.isna(wind_change) or pd.isna(pressure_change):
        return "Unknown"

    if wind_change > 0 and pressure_change < 0:
        return "Strengthening"

    if wind_change < 0 and pressure_change > 0:
        return "Weakening"

    return "Stable"


storm["BEHAVIOUR"] = storm.apply(
    classify_behaviour,
    axis=1
)

current_behaviour = storm.iloc[-1]["BEHAVIOUR"]

wind_change = float(storm.iloc[-1]["WIND_CHANGE"]) if pd.notna(
    storm.iloc[-1]["WIND_CHANGE"]
) else 0.0

pressure_change = float(storm.iloc[-1]["PRESSURE_CHANGE"]) if pd.notna(
    storm.iloc[-1]["PRESSURE_CHANGE"]
) else 0.0


# ============================================================
# CURRENT STATUS
# ============================================================

st.subheader("📊 Current Cyclone Status")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Cyclone", str(selected_name))

with c2:
    st.metric("Year", str(selected_year))

with c3:
    st.metric("Wind", f"{latest['USA_WIND']:.0f} kt")

with c4:
    st.metric("Pressure", f"{latest['USA_PRES']:.0f} hPa")

with c5:
    st.metric("Behaviour", current_behaviour)


st.divider()


# ============================================================
# MULTI-SOURCE EXTRACTION
# ============================================================

st.header("🔗 Multi-Source Input & Feature Extraction")

st.write(
    "CycloneDNA extracts three complementary parameter groups "
    "from the selected observation: satellite structure, ocean "
    "thermal conditions and atmospheric dynamics."
)

# ============================================================
# USER SATELLITE IMAGE INPUT
# ============================================================

st.subheader("📤 Satellite Image Input")

input_mode = st.radio(
    "Choose how to provide the satellite input",
    ["Automatically retrieve satellite image", "Upload satellite image"],
    horizontal=True
)

uploaded_satellite = None

if input_mode == "Upload satellite image":
    uploaded_satellite = st.file_uploader(
        "Upload a satellite/cyclone image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload a satellite image for visual feature extraction."
    )

    if uploaded_satellite is not None:
        try:
            uploaded_satellite_image = Image.open(uploaded_satellite)
            st.image(
                uploaded_satellite_image,
                caption="User-provided satellite image",
                use_container_width=True
            )
            st.success("Uploaded satellite image ready for feature extraction.")
        except Exception:
            st.error("The uploaded file could not be read as an image.")
            uploaded_satellite_image = None
    else:
        uploaded_satellite_image = None
else:
    uploaded_satellite_image = None

sat_col, ocean_col, atmos_col = st.columns(3)


# ============================================================
# SATELLITE INPUT + HEURISTIC FEATURE EXTRACTION
# ============================================================

satellite_image = None
satellite_features = {
    "brightness": None,
    "contrast": None,
    "edge_density": None,
    "cloud_activity": "Unavailable",
    "organization": "Unavailable",
    "confidence": "Low"
}


def extract_satellite_features(image):
    """
    Lightweight image-analysis layer for the prototype.

    This does NOT claim to identify a cyclone eye using a
    trained medical/meteorological model. It extracts visual
    statistics from the retrieved satellite image and converts
    them into explainable indicators.
    """
    if image is None:
        return None

    if isinstance(image, Image.Image):
        pil_image = image.convert("L")
    else:
        try:
            pil_image = Image.open(image).convert("L")
        except Exception:
            return None

    arr = np.asarray(pil_image, dtype=np.float32) / 255.0

    if arr.size == 0:
        return None

    brightness = float(arr.mean())
    contrast = float(arr.std())

    # Approximate edge density using horizontal/vertical gradients.
    gx = np.abs(np.diff(arr, axis=1))
    gy = np.abs(np.diff(arr, axis=0))
    gradient = np.concatenate([gx.ravel(), gy.ravel()])
    edge_density = float(np.mean(gradient > 0.12))

    # These are visual heuristics only.
    if contrast >= 0.22 and edge_density >= 0.10:
        organization = "High visual organization"
        cloud_activity = "Strong cloud/convective structure"
    elif contrast >= 0.14 or edge_density >= 0.06:
        organization = "Moderate visual organization"
        cloud_activity = "Moderate cloud structure"
    else:
        organization = "Low visual organization"
        cloud_activity = "Limited visual structure"

    return {
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "cloud_activity": cloud_activity,
        "organization": organization,
        "confidence": "Heuristic"
    }


with sat_col:
    st.markdown(
        """
        <div class="source-box">
        <div class="source-title">🛰️ Satellite</div>
        <b>Source:</b> NASA imagery or user upload<br>
        <b>Extraction:</b> Visual image features
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(f"Time: {latest_date}")
    st.write(
        f"Location: {latest_lat:.2f}°, {latest_lon:.2f}°"
    )

    try:
        # User-uploaded image takes priority over automatic retrieval.
        if uploaded_satellite_image is not None:
            satellite_image = uploaded_satellite_image
            st.info("Using the user-uploaded satellite image.")
        else:
            satellite_image = get_satellite_image(
                date=latest_date.strftime("%Y-%m-%d"),
                lat=latest_lat,
                lon=latest_lon
            )

            if satellite_image is not None:
                st.image(
                    satellite_image,
                    caption=f"Automatically retrieved satellite observation — {selected_name}",
                    use_container_width=True
                )
                st.success("Satellite image dynamically retrieved.")
            else:
                st.warning("Satellite image unavailable.")

        if satellite_image is not None:
            extracted = extract_satellite_features(
                satellite_image
            )

            if extracted:
                satellite_features.update(extracted)
                st.success("Satellite features extracted successfully.")
            else:
                st.warning("Image available, but feature extraction failed.")

    except Exception:
        st.warning(
            "Satellite image could not be processed for this observation."
        )

    st.markdown("**Extracted satellite parameters**")

    if satellite_features["brightness"] is not None:
        st.metric(
            "Image brightness",
            f"{satellite_features['brightness']:.3f}"
        )
        st.metric(
            "Image contrast",
            f"{satellite_features['contrast']:.3f}"
        )
        st.metric(
            "Edge density",
            f"{satellite_features['edge_density']:.3f}"
        )

        st.write(
            f"**Cloud activity:** {satellite_features['cloud_activity']}"
        )
        st.write(
            f"**Organization:** {satellite_features['organization']}"
        )
        st.caption(
            "Satellite indicators are lightweight visual heuristics; "
            "they are not a trained cyclone-intensity classifier."
        )
    else:
        st.info("Satellite parameters unavailable.")


# ============================================================
# OCEAN INPUT + FEATURE EXTRACTION
# ============================================================

sst = None

with ocean_col:
    st.markdown(
        """
        <div class="source-box">
        <div class="source-title">🌊 Ocean</div>
        <b>Source:</b> NOAA OISST<br>
        <b>Extraction:</b> Sea-surface thermal conditions
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        f"Time: {latest_date.strftime('%Y-%m-%d')}"
    )
    st.write(
        f"Location: {latest_lat:.2f}°, {latest_lon:.2f}°"
    )

    try:
        sst = get_sst(
            date=latest_date,
            lat=latest_lat,
            lon=latest_lon
        )

        if sst is not None:
            sst = float(sst)
    except Exception:
        sst = None

    if sst is not None:
        st.metric(
            "🌡️ Sea Surface Temperature",
            f"{sst:.2f} °C"
        )
        st.success("Ocean parameter extracted.")

        if sst >= 30:
            ocean_condition = "Very warm — strong thermal support"
        elif sst >= 28:
            ocean_condition = "Warm — favourable thermal support"
        elif sst >= 26:
            ocean_condition = "Moderate thermal support"
        else:
            ocean_condition = "Cooler — less favourable thermal support"

        st.write(f"**Ocean condition:** {ocean_condition}")

        if sst >= 28:
            ocean_signal = 1
        elif sst >= 26:
            ocean_signal = 0.5
        else:
            ocean_signal = 0
    else:
        st.info("SST unavailable for this observation.")
        ocean_condition = "Unavailable"
        ocean_signal = None

    st.markdown("**Extracted ocean parameters**")
    st.write("• Sea Surface Temperature")
    st.write("• Thermal support classification")
    st.write("• Ocean intensification signal")


# ============================================================
# ATMOSPHERIC INPUT + FEATURE EXTRACTION
# ============================================================

with atmos_col:
    st.markdown(
        """
        <div class="source-box">
        <div class="source-title">🌪️ Atmosphere</div>
        <b>Source:</b> IBTrACS<br>
        <b>Extraction:</b> Storm dynamics
        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric(
        "Wind",
        f"{latest['USA_WIND']:.0f} kt"
    )

    st.metric(
        "Pressure",
        f"{latest['USA_PRES']:.0f} hPa"
    )

    if pd.notna(latest["STORM_SPEED"]):
        st.metric(
            "Movement Speed",
            f"{latest['STORM_SPEED']:.1f}"
        )

    if pd.notna(latest["STORM_DIR"]):
        st.metric(
            "Movement Direction",
            f"{latest['STORM_DIR']:.0f}°"
        )

    st.markdown("**Extracted atmospheric parameters**")
    st.write(f"• Wind change: {wind_change:+.1f} kt")
    st.write(f"• Pressure change: {pressure_change:+.1f} hPa")
    st.write(f"• Behaviour: {current_behaviour}")

    if wind_change > 0 and pressure_change < 0:
        atmospheric_signal = 1.0
    elif wind_change > 0 or pressure_change < 0:
        atmospheric_signal = 0.5
    elif wind_change < 0 and pressure_change > 0:
        atmospheric_signal = 0.0
    else:
        atmospheric_signal = 0.25


# ============================================================
# MULTI-SOURCE FUSION / RISK ENGINE
# ============================================================

st.divider()
st.header("🧠 Multi-Source Fusion & Decision Engine")

st.write(
    "The three extracted parameter groups are combined into an "
    "explainable prototype risk assessment. The result is a "
    "decision-support indicator, not an official warning."
)


# Satellite signal
sat_signal = None

if satellite_features["organization"] != "Unavailable":
    if satellite_features["organization"] == "High visual organization":
        sat_signal = 1.0
    elif satellite_features["organization"] == "Moderate visual organization":
        sat_signal = 0.5
    else:
        sat_signal = 0.2


signals = [
    x for x in [sat_signal, ocean_signal, atmospheric_signal]
    if x is not None
]

if signals:
    fusion_score = float(np.mean(signals)) * 100
else:
    fusion_score = 0.0


# ============================================================
# ALERT GENERATION
# ============================================================

alerts = []
suggestions = []

# Atmospheric alerts
if wind_change >= 15 and pressure_change <= -5:
    alerts.append(
        (
            "HIGH",
            "🚨 Rapid intensification signal",
            "Wind is increasing rapidly while central pressure is falling."
        )
    )
    suggestions.append(
        "Increase monitoring frequency and prepare for possible rapid intensification."
    )
elif wind_change > 0 and pressure_change < 0:
    alerts.append(
        (
            "MEDIUM",
            "⚠️ Strengthening trend",
            "Wind is increasing while pressure is decreasing."
        )
    )
    suggestions.append(
        "Continue close monitoring because atmospheric conditions indicate strengthening."
    )

# Ocean alerts
if sst is not None:
    if sst >= 30:
        alerts.append(
            (
                "HIGH",
                "🌊 Very warm ocean support",
                f"SST is {sst:.2f} °C, indicating strong thermal support."
            )
        )
        suggestions.append(
            "Watch for further intensification while the cyclone remains over very warm water."
        )
    elif sst >= 28:
        alerts.append(
            (
                "MEDIUM",
                "🌊 Warm ocean support",
                f"SST is {sst:.2f} °C, providing favourable thermal conditions."
            )
        )
        suggestions.append(
            "Track changes in SST and atmospheric intensity together."
        )

# Satellite alert
if sat_signal is not None:
    if sat_signal >= 1:
        alerts.append(
            (
                "MEDIUM",
                "🛰️ Strong satellite organization",
                "The retrieved image shows high visual organization and strong structural contrast."
            )
        )
        suggestions.append(
            "Review the latest satellite frames for persistent organization or eye development."
        )
    elif sat_signal >= 0.5:
        alerts.append(
            (
                "LOW",
                "🛰️ Moderate satellite organization",
                "The retrieved image shows moderate visual organization."
            )
        )

# Multi-source consensus
available_source_count = len(signals)

if (
    atmospheric_signal is not None
    and ocean_signal is not None
    and sat_signal is not None
    and atmospheric_signal >= 0.5
    and ocean_signal >= 0.5
    and sat_signal >= 0.5
):
    alerts.append(
        (
            "HIGH",
            "🧬 Multi-source intensification consensus",
            "Satellite, ocean and atmospheric indicators are simultaneously supportive."
        )
    )
    suggestions.append(
        "Treat the situation as a higher-priority monitoring case and verify with official advisories."
    )

# Weakening
if current_behaviour == "Weakening":
    alerts.append(
        (
            "LOW",
            "🟢 Weakening trend",
            "Wind is decreasing while pressure is increasing."
        )
    )
    suggestions.append(
        "Continue monitoring; current atmospheric indicators suggest weakening."
    )


# ============================================================
# OVERALL LEVEL
# ============================================================

if any(a[0] == "HIGH" for a in alerts):
    overall_level = "HIGH"
elif any(a[0] == "MEDIUM" for a in alerts):
    overall_level = "MEDIUM"
elif alerts:
    overall_level = "LOW"
else:
    overall_level = "NORMAL"


if overall_level == "HIGH":
    st.error(f"🚨 OVERALL ALERT LEVEL: {overall_level}")
elif overall_level == "MEDIUM":
    st.warning(f"⚠️ OVERALL ALERT LEVEL: {overall_level}")
else:
    st.success(f"🟢 OVERALL ALERT LEVEL: {overall_level}")


# ============================================================
# SOURCE CONSENSUS SCORE
# ============================================================

score_cols = st.columns(4)

with score_cols[0]:
    st.metric(
        "Fusion Score",
        f"{fusion_score:.0f}/100"
    )

with score_cols[1]:
    st.metric(
        "Sources Available",
        f"{available_source_count}/3"
    )

with score_cols[2]:
    st.metric(
        "Satellite",
        "Available" if sat_signal is not None else "Unavailable"
    )

with score_cols[3]:
    st.metric(
        "Ocean",
        "Available" if ocean_signal is not None else "Unavailable"
    )


# ============================================================
# ALERTS
# ============================================================

st.subheader("🚨 Alerts")

if not alerts:
    st.success(
        "No significant rule-based alert was generated from the available inputs."
    )
else:
    for level, title, reason in alerts:
        if level == "HIGH":
            css_class = "alert-high"
        elif level == "MEDIUM":
            css_class = "alert-medium"
        else:
            css_class = "alert-low"

        st.markdown(
            f"""
            <div class="{css_class}">
                <h4>{title}</h4>
                <p><b>Reason:</b> {reason}</p>
                <p><b>Level:</b> {level}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SUGGESTIONS
# ============================================================

st.subheader("💡 Suggestions")

# Remove duplicate suggestions while preserving order.
suggestions = list(dict.fromkeys(suggestions))

if suggestions:
    for i, suggestion in enumerate(suggestions, start=1):
        st.info(f"**{i}.** {suggestion}")
else:
    st.info(
        "Continue routine monitoring and compare new observations "
        "as they become available."
    )

st.caption(
    "⚠️ CycloneDNA suggestions are prototype decision-support outputs. "
    "They do not replace official meteorological warnings, forecasts or evacuation instructions."
)


# ============================================================
# FUSION VISUALIZATION
# ============================================================

st.subheader("🧬 Three-Source Cyclone Fingerprint")

fingerprint_labels = ["Satellite", "Ocean", "Atmosphere"]
fingerprint_values = [
    sat_signal * 100 if sat_signal is not None else 0,
    ocean_signal * 100 if ocean_signal is not None else 0,
    atmospheric_signal * 100 if atmospheric_signal is not None else 0
]

fig_fusion, ax_fusion = plt.subplots(figsize=(9, 4))
ax_fusion.bar(
    fingerprint_labels,
    fingerprint_values
)
ax_fusion.set_ylim(0, 100)
ax_fusion.set_ylabel("Support score")
ax_fusion.set_title("Multi-Source Indicator Strength")
st.pyplot(fig_fusion, use_container_width=True)
plt.close(fig_fusion)


# ============================================================
# CYCLONE DNA
# ============================================================

st.markdown(
    """
    <div class="dna-box">
    <h2>🧬 Dynamic Cyclone DNA</h2>
    <p>
    Current behavioural fingerprint created from atmospheric
    observations and multi-source environmental indicators.
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

dna_features = [
    "USA_WIND",
    "USA_PRES",
    "STORM_SPEED",
    "WIND_INTENSIFICATION"
]

dna_data = storm[dna_features].dropna()

if len(dna_data) > 1:

    latest_dna = dna_data.iloc[-1]
    minimum = dna_data.min()
    maximum = dna_data.max()

    denominator = (maximum - minimum).replace(0, 1)

    normalized = (latest_dna - minimum) / denominator

    labels = [
        "Wind",
        "Pressure",
        "Movement",
        "Intensification"
    ]

    values = normalized.values.tolist()
    values += values[:1]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(labels),
        endpoint=False
    ).tolist()

    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw={"polar": True}
    )

    ax.plot(
        angles,
        values,
        linewidth=3
    )

    ax.fill(
        angles,
        values,
        alpha=0.25
    )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)

    ax.set_title(
        "Current Cyclone Behavioural Fingerprint",
        pad=25
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# TRACK MAP
# ============================================================

st.header("🌍 Cyclone Track")

track = storm.dropna(subset=["LAT", "LON"])

map_data = track[
    ["LAT", "LON"]
].rename(
    columns={
        "LAT": "lat",
        "LON": "lon"
    }
)

if not map_data.empty:
    st.map(
        map_data,
        zoom=4
    )


# ============================================================
# INTENSITY EVOLUTION
# ============================================================

st.header("📈 Cyclone Evolution")

chart_data = storm[
    [
        "ISO_TIME",
        "USA_WIND",
        "USA_PRES"
    ]
].dropna()

if not chart_data.empty:

    chart_data = chart_data.set_index("ISO_TIME")

    st.line_chart(
        chart_data,
        height=400
    )


# ============================================================
# BEHAVIOUR ANALYSIS
# ============================================================

st.header("🔬 Atmospheric Behaviour Analysis")

if current_behaviour == "Strengthening":

    st.success(
        "🟢 Strengthening detected: wind is increasing while "
        "pressure is decreasing."
    )

elif current_behaviour == "Weakening":

    st.warning(
        "🟠 Weakening detected: wind is decreasing while "
        "pressure is increasing."
    )

else:

    st.info(
        "🔵 Behaviour is currently stable based on the latest "
        "available atmospheric observation."
    )


# ============================================================
# RAW DATA
# ============================================================

with st.expander("📊 View Backend Observations"):

    columns = [
        "ISO_TIME",
        "LAT",
        "LON",
        "USA_WIND",
        "USA_PRES",
        "STORM_SPEED",
        "STORM_DIR",
        "WIND_CHANGE",
        "PRESSURE_CHANGE",
        "WIND_INTENSIFICATION",
        "BEHAVIOUR"
    ]

    available = [
        c for c in columns
        if c in storm.columns
    ]

    st.dataframe(
        storm[available],
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CycloneDNA AI — Prototype | "
    "Satellite + Ocean + Atmospheric Feature Extraction → "
    "Multi-Source Fusion → Alerts & Suggestions"
)