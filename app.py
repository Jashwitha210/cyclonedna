import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    .main {
        background-color: #f4f7fb;
    }

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
        background: linear-gradient(
            135deg,
            #111827,
            #243b64
        );
        color: white;
        margin-top: 20px;
        margin-bottom: 20px;
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
    "Dynamic Cyclone DNA"
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_backend():

    return load_data()


df = load_backend()

cyclones = get_named_cyclones(df)


# ============================================================
# CYCLONE SELECTOR
# ============================================================

st.subheader("🌪️ Select Cyclone")

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

storm = get_cyclone(
    df,
    selected_sid
).copy()


numeric_columns = [
    "LAT",
    "LON",
    "USA_WIND",
    "USA_PRES",
    "STORM_SPEED",
    "STORM_DIR"
]

for column in numeric_columns:

    storm[column] = pd.to_numeric(
        storm[column],
        errors="coerce"
    )


storm["ISO_TIME"] = pd.to_datetime(
    storm["ISO_TIME"],
    errors="coerce"
)

storm = storm.sort_values(
    "ISO_TIME"
)


valid_storm = storm.dropna(
    subset=[
        "LAT",
        "LON",
        "USA_WIND",
        "USA_PRES"
    ]
)


if valid_storm.empty:

    st.error(
        "No valid observations available."
    )

    st.stop()


latest = valid_storm.iloc[-1]


latest_date = latest["ISO_TIME"]

latest_lat = float(latest["LAT"])

latest_lon = float(latest["LON"])


# ============================================================
# ATMOSPHERIC BEHAVIOUR
# ============================================================

storm["WIND_CHANGE"] = (
    storm["USA_WIND"].diff()
)

storm["PRESSURE_CHANGE"] = (
    storm["USA_PRES"].diff()
)

storm["WIND_INTENSIFICATION"] = (
    storm["WIND_CHANGE"] / 3
)


def classify_behaviour(row):

    wind_change = row["WIND_INTENSIFICATION"]

    pressure_change = row["PRESSURE_CHANGE"]

    if pd.isna(wind_change) or pd.isna(pressure_change):

        return "Unknown"

    if (
        wind_change > 0
        and pressure_change < 0
    ):

        return "Strengthening"

    if (
        wind_change < 0
        and pressure_change > 0
    ):

        return "Weakening"

    return "Stable"


storm["BEHAVIOUR"] = storm.apply(
    classify_behaviour,
    axis=1
)


current_behaviour = (
    valid_storm
    .assign(
        BEHAVIOUR=storm["BEHAVIOUR"]
    )
    .iloc[-1]["BEHAVIOUR"]
)


# ============================================================
# CURRENT STATUS
# ============================================================

st.subheader("📊 Current Cyclone Status")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.metric(
        "Cyclone",
        str(selected_name)
    )

with c2:

    st.metric(
        "Year",
        str(selected_year)
    )

with c3:

    st.metric(
        "Wind",
        f"{latest['USA_WIND']:.0f} kt"
    )

with c4:

    st.metric(
        "Pressure",
        f"{latest['USA_PRES']:.0f} hPa"
    )

with c5:

    st.metric(
        "Behaviour",
        current_behaviour
    )


st.divider()


# ============================================================
# MULTI-SOURCE INPUTS
# ============================================================

st.header("🔗 Multi-Source Data Inputs")

st.write(
    "The selected cyclone is used to retrieve and organize "
    "three information sources before generating the Cyclone DNA."
)


sat_col, ocean_col, atmos_col = st.columns(3)


# ============================================================
# SATELLITE INPUT
# ============================================================

with sat_col:

    st.markdown(
        """
        <div class="source-box">
        <div class="source-title">🛰️ Satellite Input</div>
        <b>Source:</b> NASA satellite imagery<br>
        <b>Role:</b> Visual cyclone structure
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        f"Time: {latest_date}"
    )

    st.write(
        f"Location: "
        f"{latest_lat:.2f}°, "
        f"{latest_lon:.2f}°"
    )

    satellite_image = None

    try:

        satellite_image = get_satellite_image(
            date=latest_date.strftime("%Y-%m-%d"),
            lat=latest_lat,
            lon=latest_lon
        )

        st.image(
            satellite_image,
            caption=(
                f"Dynamic satellite view — "
                f"{selected_name}"
            ),
            use_container_width=True
        )

        st.success(
            "Satellite image dynamically retrieved."
        )

    except Exception as e:

        st.warning(
            "Satellite image could not be retrieved "
            "for this observation."
        )


    st.write(
        "**Satellite features to extract:**"
    )

    st.write(
        "• Cloud organisation\n"
        "• Eye formation\n"
        "• Spiral structure\n"
        "• Cloud symmetry\n"
        "• Cloud-top characteristics"
    )


# ============================================================
# OCEAN INPUT
# ============================================================

with ocean_col:

    st.markdown(
        """
        <div class="source-box">
        <div class="source-title">🌊 Ocean Input</div>
        <b>Source:</b> NOAA OISST<br>
        <b>Role:</b> Ocean thermal conditions
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        f"Time: "
        f"{latest_date.strftime('%Y-%m-%d')}"
    )

    st.write(
        f"Location: "
        f"{latest_lat:.2f}°, "
        f"{latest_lon:.2f}°"
    )

    sst = None

    try:

        sst = get_sst(
            date=latest_date,
            lat=latest_lat,
            lon=latest_lon
        )

    except Exception:

        sst = None


    if sst is not None:

        st.metric(
            "🌡️ Sea Surface Temperature",
            f"{sst:.2f} °C"
        )

        st.success(
            "Ocean data retrieved dynamically."
        )

    else:

        st.info(
            "SST unavailable for this observation."
        )


    st.write(
        "**Ocean features:**"
    )

    st.write(
        "• Sea Surface Temperature\n"
        "• Ocean thermal conditions\n"
        "• Ocean energy indicators"
    )


# ============================================================
# ATMOSPHERIC INPUT
# ============================================================

with atmos_col:

    st.markdown(
        """
        <div class="source-box">
        <div class="source-title">🌪️ Atmospheric Input</div>
        <b>Source:</b> IBTrACS<br>
        <b>Role:</b> Storm dynamics
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

    st.write(
        "**Atmospheric features:**"
    )

    st.write(
        "• Wind speed\n"
        "• Pressure\n"
        "• Movement\n"
        "• Pressure change\n"
        "• Intensification"
    )


# ============================================================
# FUSION PIPELINE
# ============================================================

st.divider()

st.header("🔗 Multi-Source Fusion")

st.info(
    "Satellite + Ocean + Atmosphere + Behaviour "
    "are combined into the Cyclone DNA representation."
)


fusion1, fusion2, fusion3, fusion4 = st.columns(4)

with fusion1:

    st.markdown(
        "### 🛰️\nSatellite"
    )

with fusion2:

    st.markdown(
        "### 🌊\nOcean"
    )

with fusion3:

    st.markdown(
        "### 🌪️\nAtmosphere"
    )

with fusion4:

    st.markdown(
        "### 🧬\nDNA"
    )


# ============================================================
# CYCLONE DNA
# ============================================================

st.markdown(
    """
    <div class="dna-box">
    <h2>🧬 Dynamic Cyclone DNA</h2>
    <p>
    Current behavioural fingerprint of the selected cyclone.
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


dna_data = storm[
    dna_features
].dropna()


if len(dna_data) > 1:

    latest_dna = dna_data.iloc[-1]

    minimum = dna_data.min()

    maximum = dna_data.max()

    denominator = (
        maximum - minimum
    ).replace(
        0,
        1
    )

    normalized = (
        (latest_dna - minimum)
        / denominator
    )

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
        subplot_kw={
            "polar": True
        }
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

    ax.set_xticks(
        angles[:-1]
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_title(
        "Current Cyclone Behavioural Fingerprint",
        pad=25
    )

    st.pyplot(
        fig,
        use_container_width=True
    )


# ============================================================
# TRACK MAP
# ============================================================

st.header("🌍 Cyclone Track")

track = storm.dropna(
    subset=[
        "LAT",
        "LON"
    ]
)


map_data = track[
    [
        "LAT",
        "LON"
    ]
].rename(
    columns={
        "LAT": "lat",
        "LON": "lon"
    }
)


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

    chart_data = chart_data.set_index(
        "ISO_TIME"
    )

    st.line_chart(
        chart_data,
        height=400
    )


# ============================================================
# BEHAVIOUR
# ============================================================

st.header("🔬 Behaviour Analysis")

if current_behaviour == "Strengthening":

    st.success(
        "🟢 Strengthening detected: "
        "wind is increasing while pressure is decreasing."
    )

elif current_behaviour == "Weakening":

    st.warning(
        "🟠 Weakening detected."
    )

else:

    st.info(
        "🔵 Behaviour is currently stable."
    )


# ============================================================
# RAW DATA
# ============================================================

with st.expander(
    "📊 View Backend Observations"
):

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
    "Satellite + Ocean + Atmospheric Behaviour Analysis"
)