import requests
from PIL import Image
from io import BytesIO
from datetime import datetime


# ============================================================
# SATELLITE IMAGE
# ============================================================

def get_satellite_image(
    date,
    lat,
    lon,
    size=7
):
    """
    Dynamically retrieve satellite imagery around
    the selected cyclone location and date.
    """

    west = lon - size
    east = lon + size
    south = lat - size
    north = lat + size

    url = (
        "https://gibs.earthdata.nasa.gov/"
        "wms/epsg4326/best/wms.cgi"
    )

    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.3.0",
        "LAYERS": "MODIS_Terra_CorrectedReflectance_TrueColor",
        "STYLES": "",
        "CRS": "EPSG:4326",
        "BBOX": f"{south},{west},{north},{east}",
        "WIDTH": 1000,
        "HEIGHT": 700,
        "FORMAT": "image/jpeg",
        "TIME": date
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return Image.open(
        BytesIO(response.content)
    )


# ============================================================
# OCEAN SST
# ============================================================

def get_sst(
    date,
    lat,
    lon
):
    """
    Retrieve approximate sea surface temperature
    near the cyclone from NOAA OISST through ERDDAP.
    """

    date_string = str(date)[:10]

    base_url = (
        "https://coastwatch.pfeg.noaa.gov/"
        "erddap/griddap/"
        "ncdcOisst21NrtAgg_LonPM180.csv"
    )

    # NOAA OISST uses 0.25 degree grid.
    query = (
        f"?sst[({date_string}T12:00:00Z)]"
        f"[({lat})]"
        f"[({lon})]"
    )

    response = requests.get(
        base_url + query,
        timeout=20
    )

    response.raise_for_status()

    lines = response.text.strip().splitlines()

    if len(lines) < 2:
        return None

    header = lines[0].split(",")

    values = lines[1].split(",")

    try:

        sst_index = header.index("sst")

        sst_value = float(
            values[sst_index]
        )

        return sst_value

    except (ValueError, IndexError):

        return None