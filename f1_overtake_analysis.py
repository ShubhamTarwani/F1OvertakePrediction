# F1 Telemetry Analysis & Overtake Prediction
# Fixed Version for Current FastF1 API

import subprocess
import sys

# Install required packages
def install_packages():
    packages = ['fastf1', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn']
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

# Now import the packages
import fastf1 as f1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("✅ Packages imported successfully!")

# Configure cache and load data
print("📥 Loading F1 session data...")
try:
    f1.Cache.enable_cache('./f1_cache')
except:
    print("Cache setup failed, continuing without cache...")

# Load a recent race session
try:
    session = f1.get_session(2024, 'Bahrain', 'R')
    session.load()
    print(f"✅ Loaded: {session.event['EventName']} {session.name}")
except Exception as e:
    print(f"Error loading 2024 Bahrain: {e}")
    try:
        # Fallback to 2023 if 2024 not available
        session = f1.get_session(2023, 'Bahrain', 'R') 
        session.load()
        print(f"✅ Loaded: {session.event['EventName']} {session.name}")
    except Exception as e2:
        print(f"Error loading 2023 Bahrain: {e2}")
        # Try one more fallback
        try:
            session = f1.get_session(2023, 'Monza', 'R')
            session.load()
            print(f"✅ Loaded: {session.event['EventName']} {session.name}")
        except Exception as e3:
            print(f"All session load attempts failed: {e3}")
            sys.exit(1)

# Data preparation function
def prepare_overtake_dataset(session):
    """Prepare dataset for overtake prediction"""
    print("🛠 Preparing dataset...")
    
    laps = session.laps
    drivers = session.drivers
    
    overtake_data = []
    
    for driver in drivers:
        try:
            driver_laps = laps.pick_driver(driver).copy()
            if len(driver_laps) < 2:  # Need at least 2 laps for position change
                continue
                
            driver_laps = driver_laps.sort_values('LapNumber')
            driver_laps = driver_laps[driver_laps['Position'].notna()]
            
            # Calculate position changes (overtakes)
            driver_laps['NextLapPosition'] = driver_laps['Position'].shift(-1)
            driver_laps['PositionChange'] = driver_laps['Position'] - driver_laps['NextLapPosition']
            
            # Target: Overtake in next lap (gained 1+ positions)
            driver_laps['OvertakeNextLap'] = (driver_laps['PositionChange'] > 0).astype(int)
            
            # Feature engineering
            if 'LapTime' in driver_laps.columns and driver_laps['LapTime'].notna().any():
                driver_laps['LapTimeSeconds'] = driver_laps['LapTime'].dt.total_seconds()
            else:
                driver_laps['LapTimeSeconds'] = 0
                
            if 'Stint' in driver_laps.columns:
                driver_laps['TireAge'] = driver_laps['LapNumber'] - driver_laps['Stint']
            else:
                driver_laps['TireAge'] = 0
            
            # Add more features
            driver_laps['Position'] = driver_laps['Position'].fillna(20)  # Fill missing positions
            driver_laps['IsTop5'] = (driver_laps['Position'] <= 5).astype(int)
            driver_laps['IsLast5'] = (driver_laps['Position'] >= 15).astype(int)
            
            # Keep only laps where we have target value
            driver_laps = driver_laps[driver_laps['OvertakeNextLap'].notna()]
            
            overtake_data.append(driver_laps)
            
        except Exception as e:
            print(f"Error processing driver {driver}: {e}")
            continue
    
    if not overtake_data:
        raise ValueError("No data could be processed")
        
    df = pd.concat(overtake_data, ignore_index=True)
    print(f"✅ Dataset prepared: {len(df)} laps")
    return df

# Prepare the dataset
try:
    df = prepare_overtake_dataset(session)
except Exception as e:
    print(f"Error preparing dataset: {e}")
    sys.exit(1)

# Display dataset info
print(f"\n📊 Dataset Overview:")
print(f"Total laps: {len(df)}")
print(f"Overtake events: {df['OvertakeNextLap'].sum()} ({df['OvertakeNextLap'].mean()*100:.1f}%)")
print(f"Drivers in dataset: {df['Driver'].nunique()}")

# Basic EDA Visualizations
print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Overtake distribution
df['OvertakeNextLap'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'coral'])
axes[0,0].set_title('Overtake Distribution (0=No, 1=Yes)')
axes[0,0].set_xlabel('Overtake Occurred')
axes[0,0].set_ylabel('Count')

# Plot 2: Position vs Overtakes
position_overtakes = df.groupby('Position')['OvertakeNextLap'].mean()
position_overtakes.plot(ax=axes[0,1], color='green', marker='o')
axes[0,1].set_title('Overtake Probability by Position')
axes[0,1].set_xlabel('Position')
axes[0,1].set_ylabel('Overtake Probability')

# Plot 3: Tire age vs Overtakes
tire_overtakes = df.groupby('TireAge')['OvertakeNextLap'].mean()
tire_overtakes.plot(ax=axes[1,0], color='red', marker='s')
axes[1,0].set_title('Overtake Probability by Tire Age')
axes[1,0].set_xlabel('Tire Age (laps)')
axes[1,0].set_ylabel('Overtake Probability')

# Plot 4: Lap time distribution
df['LapTimeSeconds'].hist(ax=axes[1,1], bins=30, alpha=0.7, color='purple')
axes[1,1].set_title('Lap Time Distribution')
axes[1,1].set_xlabel('Lap Time (seconds)')
axes[1,1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('f1_analysis_plots.png')  # Save instead of showing
print("✅ Plots saved as 'f1_analysis_plots.png'")

# Prepare features for machine learning
print("\n🤖 Preparing machine learning model...")

# Select features (using only available columns)
feature_columns = ['Position', 'LapTimeSeconds', 'TireAge', 'IsTop5', 'IsLast5', 'LapNumber']

# Filter out rows with missing values in our features
df_clean = df[feature_columns + ['OvertakeNextLap']].dropna()

if len(df_clean) == 0:
    print("❌ No data available after cleaning. Using fallback features...")
    # Use only basic features that are guaranteed to exist
    feature_columns = ['Position', 'LapNumber']
    df_clean = df[feature_columns + ['OvertakeNextLap']].dropna()

X = df_clean[feature_columns]
y = df_clean['OvertakeNextLap']

print(f"Features used: {feature_columns}")
print(f"Final dataset shape: {X.shape}")

if len(X) == 0:
    print("❌ No data available for modeling. Exiting.")
    sys.exit(1)

# Build and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("✅ Model training completed!")

# Model evaluation
print("\n📊 Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 Feature Importance:")
print(feature_importance)

# Final visualizations
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix')

# Add this after loading the session
print(f"Event: {session.event['EventName']} - {session.event['EventDate']}")
print(f"Session: {session.name}")
print(f"Total Laps: {session.total_laps}")
print("\nDrivers Participating:")
for driver in session.drivers:
    drv_info = session.get_driver(driver)
    print(f"  {drv_info['Abbreviation']}: {drv_info['FullName']} - {drv_info['TeamName']}")

print(f"\nTrack: {session.event['Location']} - {session.event['Country']}")

# Feature Importance Plot
sns.barplot(data=feature_importance, x='importance', y='feature', ax=axes[1], palette='viridis')
axes[1].set_title('Feature Importance')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('f1_model_results.png')  # Save instead of showing
print("✅ Model results saved as 'f1_model_results.png'")

# Example predictions
print("\n🎯 Example Predictions:")
sample_data = X_test_scaled[:5]
sample_predictions = model.predict(sample_data)
sample_probabilities = model.predict_proba(sample_data)[:, 1]

for i in range(min(5, len(sample_predictions))):
    actual = y_test.iloc[i]
    pred = sample_predictions[i]
    prob = sample_probabilities[i]
    status = "✅ CORRECT" if actual == pred else "❌ WRONG"
    print(f"Prediction: {pred} (prob: {prob:.2f}) | Actual: {actual} {status}")

print("\n" + "="*50)
print("🎉 ANALYSIS COMPLETE!")
print(f"📈 Overtake Prediction Model Ready")
print(f"🔮 Can predict overtakes with {accuracy_score(y_test, y_pred):.1%} accuracy")
print("="*50)

# Keep plots open (if running in interactive mode)
plt.show()