import fastf1
from fastf1 import plotting
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------
# Streamlit Title
# --------------------------
st.title("🏎️ F1 Telemetry Dashboard + Overtake Prediction")

# --------------------------
# FastF1 Cache Setup
# --------------------------
if not os.path.exists("cache"):
    os.makedirs("cache")
fastf1.Cache.enable_cache("cache")

# --------------------------
# Sidebar Inputs
# --------------------------
st.sidebar.header("Session Selection")
year = st.sidebar.number_input("Year", min_value=2018, max_value=2025, value=2024)
event_name = st.sidebar.text_input("Grand Prix (e.g. Monza, Bahrain)", value="Monza")
session_type = st.sidebar.selectbox("Session Type", ["R", "Q", "FP1", "FP2", "FP3"])
driver_code = st.sidebar.text_input("Driver Code (e.g. VER, HAM, NOR)", value="VER")

# --------------------------
# Load Session Data
# --------------------------
st.write("Loading session data... ⏳")
try:
    session = fastf1.get_session(year, event_name, session_type)
    session.load()
    st.success(f"Session loaded: {year} {event_name} {session_type}")
except Exception as e:
    st.error(f"Error loading session: {e}")
    st.stop()

# --------------------------
# Laps for Driver
# --------------------------
laps = session.laps.pick_driver(driver_code)
if laps.empty:
    st.error(f"No lap data found for {driver_code}")
    st.stop()

st.subheader(f"Laps for {driver_code}")
st.dataframe(laps[['LapNumber', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'SpeedI1', 'SpeedI2', 'SpeedFL']])

# --------------------------
# Telemetry Plot
# --------------------------
st.subheader(f"Telemetry for {driver_code}")
selected_lap = st.selectbox("Select Lap", laps['LapNumber'])
lap = laps.loc[laps['LapNumber'] == selected_lap].iloc[0]
telemetry = lap.get_telemetry()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(telemetry['Distance'], telemetry['Speed'], color='red', label='Speed (km/h)')
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Speed (km/h)")
ax.set_title(f"{driver_code} Lap {selected_lap} Speed Trace")
ax.legend()
st.pyplot(fig)

# --------------------------
# Detect Overtakes
# --------------------------
st.subheader("📊 Detected Overtakes")

pos_data = session.laps[['Driver', 'LapNumber', 'Position']]
overtakes = []

for drv in pos_data['Driver'].unique():
    drv_laps = pos_data[pos_data['Driver'] == drv].sort_values('LapNumber')
    drv_laps['PrevPos'] = drv_laps['Position'].shift(1)
    changes = drv_laps[drv_laps['Position'] < drv_laps['PrevPos']]
    for _, row in changes.iterrows():
        overtakes.append({
            "Driver": drv,
            "Lap": int(row['LapNumber']),
            "From": int(row['PrevPos']),
            "To": int(row['Position'])
        })

overtakes_df = pd.DataFrame(overtakes)
if not overtakes_df.empty:
    st.success(f"Detected {len(overtakes_df)} overtakes.")
    st.dataframe(overtakes_df)
else:
    st.info("No overtakes detected in this session.")

# --------------------------
# Prepare Data for Prediction
# --------------------------
st.subheader("🤖 Predict Overtake Probability")

def prepare_overtake_dataset(session):
    df = session.laps[['Driver', 'LapNumber', 'Position', 'LapTime', 'Compound', 'Stint']].copy()
    df = df.dropna(subset=['Position'])
    df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
    df['PositionChange'] = df.groupby('Driver')['Position'].diff(-1)  # if positive -> gained position next lap
    df['OvertakeNextLap'] = (df['PositionChange'] > 0).astype(int)
    df['TireAge'] = df.groupby('Driver')['LapNumber'].rank(method='first').astype(int)
    df['IsTop5'] = (df['Position'] <= 5).astype(int)
    df['IsLast5'] = (df['Position'] >= 16).astype(int)
    df = df.dropna()
    return df

# Build dataset
data = prepare_overtake_dataset(session)

if data.empty or len(data) < 10:
    st.warning("Not enough data to train prediction model.")
    st.stop()

# Feature selection
features = ['Position', 'LapTimeSeconds', 'TireAge', 'IsTop5', 'IsLast5']
X = data[features]
y = data['OvertakeNextLap']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
st.write(f"✅ Model trained (Accuracy: {acc:.2f})")

# --------------------------
# Predict Overtake Probability for Selected Driver
# --------------------------
driver_lap = data[data['Driver'] == driver_code].sort_values('LapNumber').iloc[-1]  # last lap
driver_features = driver_lap[features].to_frame().T
prob = model.predict_proba(driver_features)[0][1]  # probability of overtake next lap

st.metric(label=f"Predicted Overtake Probability for {driver_code}", value=f"{prob*100:.1f}%")

# Show feature importance
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
st.bar_chart(importances)
