"""
Project 26: Electric Vehicle Charging Station Analysis  (robust version)
Libraries: numpy, pandas only  (plus os/sys from the standard library)

Run:   python ev_charging_analysis_final.py
   or: python ev_charging_analysis_final.py path/to/26_ev_charging_sessions.csv
"""
import os
import sys
import re
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

DEFAULT_CSV = "26_ev_charging_sessions.csv"


def heading(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ----------------------------------------------------------------------------
# Helpers: locate the CSV file and detect column names automatically
# ----------------------------------------------------------------------------
def locate_csv():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    if os.path.exists(DEFAULT_CSV):
        return DEFAULT_CSV
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for folder in (os.getcwd(), here):
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".csv") and ("charging" in f.lower() or "ev" in f.lower()):
                if "cleaned" not in f.lower() and "ranking" not in f.lower() and "comparison" not in f.lower():
                    return os.path.join(folder, f)
    sys.exit(f"CSV file not found. Put '{DEFAULT_CSV}' in this folder: {os.getcwd()}")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_col(columns, exact, contains=(), exclude=()):
    """Return the real column name matching normalised names, else keyword match, else None."""
    mapping = {norm(c): c for c in columns}
    for name in exact:
        if norm(name) in mapping:
            return mapping[norm(name)]
    for key, real in mapping.items():
        if any(k in key for k in contains) and not any(x in key for x in exclude):
            return real
    return None


def to_number(series):
    """Convert to numbers even if the text has units/symbols (e.g. '12.5 kWh')."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


# ----------------------------------------------------------------------------
# TASK 1: Load and inspect
# ----------------------------------------------------------------------------
heading("TASK 1: Load and inspect")
path = locate_csv()
print("Reading file:", path)
df = pd.read_csv(path)
df.columns = [str(c).strip() for c in df.columns]
print("Shape:", df.shape)
print(df.head())
print()
df.info()
print(df.describe(include="all").T)

cols = list(df.columns)
SESSION = find_col(cols, ["Session_ID", "SessionID"], ["session"])
STATION = find_col(cols, ["Station_ID", "Station_Name", "Station"], ["station"], exclude=["location"])
LOCATION = find_col(cols, ["Station_Location", "Location"], ["location", "city", "site"])
CHARGER = find_col(cols, ["Charger_Type", "ChargerType"], ["chargertype", "chargetype", "connector", "charger"])
ENERGY = find_col(cols, ["Energy_kWh", "Energy_Delivered_kWh"], ["energy", "kwh"])
DURATION = find_col(cols, ["Charging_Duration_Mins", "Charging_Duration_Minutes"], ["duration"])
START = find_col(cols, ["Charging_Start_Time", "Start_Time"], ["start"])
END = find_col(cols, ["Charging_End_Time", "End_Time"], ["end", "stop"])

if STATION is None and LOCATION is None:
    sys.exit(f"Could not find a station/location column. Columns are: {cols}")
if STATION is None:
    STATION = LOCATION
if LOCATION is None:
    LOCATION = STATION
if ENERGY is None or START is None:
    sys.exit(f"Could not find energy/start-time columns. Columns are: {cols}")
if SESSION is None:
    SESSION = "Session_ID"
    df[SESSION] = np.arange(1, len(df) + 1)
if CHARGER is None:
    CHARGER = "Charger_Type"
    df[CHARGER] = "Unknown"

# Convert the start time early so the duration can be rebuilt from an end time if needed
_raw_start = df[START].copy()
_a = pd.to_datetime(_raw_start, errors="coerce")
_b = pd.to_datetime(_raw_start.astype(str), errors="coerce", dayfirst=True)
df[START] = _b if _b.isnull().sum() < _a.isnull().sum() else _a  # keep the parse with fewer invalid dates
if DURATION is None:
    if END is None:
        sys.exit(f"Could not find a duration column. Columns are: {cols}")
    DURATION = "Charging_Duration_Mins"
    df[DURATION] = (pd.to_datetime(df[END], errors="coerce") - df[START]).dt.total_seconds() / 60

print("\nColumns used ->", {"session": SESSION, "station": STATION, "location": LOCATION,
                             "charger": CHARGER, "energy": ENERGY, "duration": DURATION, "start": START})

# ----------------------------------------------------------------------------
# TASK 2: Missing values and duplicate Session_IDs
# ----------------------------------------------------------------------------
heading("TASK 2: Missing values & duplicates")
print("Missing values per column:\n", df.isnull().sum())
dup_count = int(df.duplicated(subset=SESSION).sum())
print("Duplicate Session_ID records:", dup_count)
df = df.drop_duplicates(subset=SESSION, keep="first").copy()
print("Shape after removing duplicates:", df.shape)

for c in {STATION, LOCATION, CHARGER}:
    df[c] = df[c].astype(object).where(df[c].notnull(), "Unknown")
    df[c] = df[c].astype(str).str.strip()

# ----------------------------------------------------------------------------
# TASK 3: Handle missing Energy / Duration (with justification)
# ----------------------------------------------------------------------------
heading("TASK 3: Handle missing Energy_kWh / Duration")
df[ENERGY] = to_number(df[ENERGY])
df[DURATION] = to_number(df[DURATION])

# Zero/negative values are impossible for a real session -> treat as missing
df.loc[df[ENERGY] <= 0, ENERGY] = np.nan
df.loc[df[DURATION] <= 0, DURATION] = np.nan

# Median (not mean): charging data is right-skewed, so a few very long sessions
# would inflate the mean. Filling within the same charger type is more accurate
# because fast chargers deliver more energy in less time than slow ones.
for col in (ENERGY, DURATION):
    n_missing = int(df[col].isnull().sum())
    df[col] = df[col].fillna(df.groupby(CHARGER)[col].transform("median"))
    df[col] = df[col].fillna(df[col].median())
    print(f"{col}: filled {n_missing} missing/invalid values with charger-type median")

if df[ENERGY].isnull().all() or df[DURATION].isnull().all():
    sys.exit("Energy or duration column has no usable numbers.")

# ----------------------------------------------------------------------------
# TASK 4: Datetime features
# ----------------------------------------------------------------------------
heading("TASK 4: Datetime features")
bad_dates = int(df[START].isnull().sum())
print("Rows with invalid start time (dropped):", bad_dates)
df = df.dropna(subset=[START]).copy()
if df.empty:
    sys.exit("No valid start times found in the data.")
df["Hour"] = df[START].dt.hour
df["Weekday"] = df[START].dt.day_name()
df["Month"] = df[START].dt.month_name()
print(df[[START, "Hour", "Weekday", "Month"]].head())

# ----------------------------------------------------------------------------
# TASK 5: Totals
# ----------------------------------------------------------------------------
heading("TASK 5: Totals")
total_sessions = len(df)
total_energy = df[ENERGY].sum()
avg_duration = df[DURATION].mean()
print(f"Total charging sessions : {total_sessions}")
print(f"Total energy delivered  : {total_energy:,.2f} kWh")
print(f"Average charging time   : {avg_duration:.2f} minutes")

# ----------------------------------------------------------------------------
# TASK 6: Average energy per session
# ----------------------------------------------------------------------------
heading("TASK 6: Average energy per session")
print(f"{df[ENERGY].mean():.2f} kWh")

# ----------------------------------------------------------------------------
# TASK 7: Rank stations
# ----------------------------------------------------------------------------
heading("TASK 7: Station ranking")
station_rank = (
    df.groupby(STATION)
    .agg(Sessions=(SESSION, "count"), Total_Energy_kWh=(ENERGY, "sum"))
    .sort_values(["Sessions", "Total_Energy_kWh"], ascending=False)
)
station_rank["Total_Energy_kWh"] = station_rank["Total_Energy_kWh"].round(2)
station_rank["Rank_by_Sessions"] = station_rank["Sessions"].rank(ascending=False, method="min").astype(int)
station_rank["Rank_by_Energy"] = station_rank["Total_Energy_kWh"].rank(ascending=False, method="min").astype(int)
print(station_rank.head(10))

# ----------------------------------------------------------------------------
# TASK 8: Charger-type comparison
# ----------------------------------------------------------------------------
heading("TASK 8: Charger-type comparison")
charger_cmp = (
    df.groupby(CHARGER)
    .agg(Sessions=(SESSION, "count"),
         Avg_Duration_Mins=(DURATION, "mean"),
         Avg_Energy_kWh=(ENERGY, "mean"))
    .sort_values("Avg_Duration_Mins", ascending=False)
    .round(2)
)
print(charger_cmp)

# ----------------------------------------------------------------------------
# TASK 9: Peak hours and busiest days
# ----------------------------------------------------------------------------
heading("TASK 9: Peak hours & busiest days")
hourly = df.groupby("Hour")[SESSION].count()
print("Top 5 peak hours:\n", hourly.sort_values(ascending=False).head(5))
peak_hour = int(hourly.idxmax())
print(f"\nPeak hour: {peak_hour}:00 ({int(hourly.max())} sessions)")

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily = df.groupby("Weekday")[SESSION].count().reindex(day_order).fillna(0).astype(int)
busiest_day = daily.idxmax()
print("\nSessions by weekday:\n", daily)
print("Busiest day:", busiest_day)

# ----------------------------------------------------------------------------
# TASK 10: Energy consumption across locations
# ----------------------------------------------------------------------------
heading("TASK 10: Energy by location")
loc_energy = (
    df.groupby(LOCATION)[ENERGY]
    .agg(Total_kWh="sum", Avg_kWh="mean", Sessions="count")
    .sort_values("Total_kWh", ascending=False)
    .round(2)
)
print(loc_energy.head(10))

# ----------------------------------------------------------------------------
# TASK 11: Tuple (fixed charger types) and set (unique station locations)
# ----------------------------------------------------------------------------
heading("TASK 11: Tuple and set")
charger_types = tuple(sorted(df[CHARGER].unique()))
unique_locations = set(df[LOCATION].unique())
print("Charger types (tuple):", charger_types)
print("Number of unique locations (set):", len(unique_locations))

# ----------------------------------------------------------------------------
# TASK 12: Dictionary of station session counts + function
# ----------------------------------------------------------------------------
heading("TASK 12: Dictionary + function")
station_sessions = {}
for st in df[STATION]:
    station_sessions[st] = station_sessions.get(st, 0) + 1
print("Sample of station session counts:", dict(list(station_sessions.items())[:5]))


def avg_energy_per_session(station_name):
    """Average energy (kWh) per session for the given station."""
    if station_name not in station_sessions:
        return None
    total_kwh = df.loc[df[STATION] == station_name, ENERGY].sum()
    return total_kwh / station_sessions[station_name]


selected_station = station_rank.index[0]
print(f"Average energy per session at '{selected_station}': "
      f"{avg_energy_per_session(selected_station):.2f} kWh")

# ----------------------------------------------------------------------------
# TASK 13: NumPy statistics of charging duration
# ----------------------------------------------------------------------------
heading("TASK 13: NumPy statistics")
dur = df[DURATION].to_numpy(dtype=float)
print(f"Mean   : {np.mean(dur):.2f} mins")
print(f"Median : {np.median(dur):.2f} mins")
print(f"90th % : {np.percentile(dur, 90):.2f} mins")

# ----------------------------------------------------------------------------
# TASK 14: IQR outliers + duration groups
# ----------------------------------------------------------------------------
heading("TASK 14: Outliers (IQR) & duration groups")
q1, q3 = np.percentile(dur, [25, 75])
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
df["Long_Outlier"] = df[DURATION] > upper_fence
n_outliers = int(df["Long_Outlier"].sum())
print(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, upper fence={upper_fence:.2f}")
print("Unusually long sessions:", n_outliers)

# Short = up to Q1, Long = above Q3, Medium = in between
df["Duration_Group"] = np.where(df[DURATION] <= q1, "Short",
                         np.where(df[DURATION] > q3, "Long", "Medium"))
print(df["Duration_Group"].value_counts())

# ----------------------------------------------------------------------------
# TASK 15: ChargingSession class + recommendations
# ----------------------------------------------------------------------------
heading("TASK 15: ChargingSession class")


class ChargingSession:
    def __init__(self, session_id, station, energy_kwh, duration_mins=None, charger_type=None):
        self.session_id = session_id
        self.station = station
        self.energy_kwh = float(energy_kwh)
        self.duration_mins = None if duration_mins is None else float(duration_mins)
        self.charger_type = charger_type

    def avg_power_kw(self):
        """Average charging power = energy / time (hours)."""
        if not self.duration_mins:
            return None
        return self.energy_kwh / (self.duration_mins / 60)

    def __repr__(self):
        return (f"ChargingSession(id={self.session_id}, station={self.station}, "
                f"energy={self.energy_kwh:.1f} kWh)")


row = df.iloc[0]
s = ChargingSession(row[SESSION], row[STATION], row[ENERGY], row[DURATION], row[CHARGER])
power = s.avg_power_kw()
print(s, "| avg power:", "n/a" if power is None else f"{power:.2f} kW")

heading("FIVE OPERATIONAL RECOMMENDATIONS (data-driven)")
top_station = station_rank.index[0]
top_sessions = int(station_rank.iloc[0]["Sessions"])
slow_type = charger_cmp.index[0]
slow_mins = charger_cmp.iloc[0]["Avg_Duration_Mins"]
print(f"""
1. Add chargers at high-demand stations (e.g. '{top_station}', {top_sessions} sessions)
   to cut waiting time and prevent overload.
2. Use time-based pricing/incentives to move demand away from the peak hour
   ({peak_hour}:00) and the busiest day ({busiest_day}) toward off-peak periods.
3. Upgrade or limit '{slow_type}' chargers (longest average duration, {slow_mins} mins)
   or add faster units, so each bay serves more vehicles per day.
4. Set idle/overstay fees and send alerts for sessions beyond {upper_fence:.0f} mins
   ({n_outliers} unusually long sessions) to free up bays.
5. Run preventive maintenance and monitor under-used stations/locations to find faults,
   poor placement, or visibility problems; relocate or promote them.
""")

# Save cleaned data and summary tables (skipped quietly if the folder is read-only)
try:
    df.to_csv("cleaned_ev_charging_sessions.csv", index=False)
    station_rank.to_csv("station_ranking.csv")
    charger_cmp.to_csv("charger_type_comparison.csv")
    print("Saved: cleaned_ev_charging_sessions.csv, station_ranking.csv, charger_type_comparison.csv")
except OSError as e:
    print("Could not save output files:", e)