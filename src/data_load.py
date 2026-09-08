import pandas as pd
from pathlib import Path


def load_data():
    """
    Load the IBTrACS North Indian Ocean dataset.
    """

    project_root = Path(__file__).resolve().parent.parent

    file_path = (
        project_root
        / "data"
        / "raw"
        / "ibtracs.NI.list.v04r01.csv"
    )

    df = pd.read_csv(
        file_path,
        skiprows=[1]
    )

    # Convert important columns to numeric
    numeric_columns = [
        "SEASON",
        "LAT",
        "LON",
        "USA_WIND",
        "USA_PRES",
        "STORM_SPEED",
        "STORM_DIR"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Convert time
    df["ISO_TIME"] = pd.to_datetime(
        df["ISO_TIME"],
        errors="coerce"
    )

    return df


def get_named_cyclones(df):
    """
    Return a clean list of named cyclones.
    """

    cyclones = (
        df[df["NAME"] != "UNNAMED"]
        [["SID", "SEASON", "NAME"]]
        .drop_duplicates()
        .sort_values(
            ["SEASON", "NAME"],
            ascending=[False, True]
        )
    )

    return cyclones


def get_cyclone(df, sid):
    """
    Return all observations for one cyclone.
    """

    storm = df[df["SID"] == sid].copy()

    storm = storm.sort_values("ISO_TIME")

    return storm