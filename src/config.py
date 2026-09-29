"""Project-wide settings. Change YEARS or HOME_AIRPORT here and every script follows."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
APP_DATA_DIR = ROOT / "app" / "data"      # small, aggregated tables the live app reads
MODELS_DIR = ROOT / "app" / "models"      # trained model shipped with the app

for _d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR, APP_DATA_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

YEARS = [2023, 2024, 2025]
MONTHS = list(range(1, 13))
HOME_AIRPORT = "STL"  # St. Louis Lambert: gets its own spotlight in the app

# Bureau of Transportation Statistics, "Reporting Carrier On-Time Performance (1987-present)"
BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# BTS column name -> our column name
RAW_COLUMNS = {
    "FlightDate": "flight_date",
    "Reporting_Airline": "airline",
    "Flight_Number_Reporting_Airline": "flight_number",
    "Origin": "origin",
    "OriginCityName": "origin_city",
    "Dest": "dest",
    "DestCityName": "dest_city",
    "CRSDepTime": "crs_dep_time",
    "DepDelay": "dep_delay",
    "ArrDelay": "arr_delay",
    "ArrDel15": "is_delayed",
    "Cancelled": "cancelled",
    "Diverted": "diverted",
    "Distance": "distance",
    "CarrierDelay": "carrier_delay",
    "WeatherDelay": "weather_delay",
    "NASDelay": "nas_delay",
    "SecurityDelay": "security_delay",
    "LateAircraftDelay": "late_aircraft_delay",
}

CAUSE_COLUMNS = ["carrier_delay", "weather_delay", "nas_delay",
                 "security_delay", "late_aircraft_delay"]

CARRIER_NAMES = {
    "AA": "American Airlines", "AS": "Alaska Airlines", "B6": "JetBlue",
    "DL": "Delta Air Lines", "F9": "Frontier", "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines", "MQ": "Envoy Air", "NK": "Spirit Airlines",
    "OH": "PSA Airlines", "OO": "SkyWest", "UA": "United Airlines",
    "WN": "Southwest Airlines", "YX": "Republic Airways", "9E": "Endeavor Air",
    "QX": "Horizon Air", "EV": "ExpressJet",
}
