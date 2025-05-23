# Crime-Rate-Analysis-Project
This project focuses on analyzing and visualizing crime rates across Indian states using Python. 
Crime rate analysis is crucial for understanding patterns, trends, and factors influencing criminal 
activities in different cities. This study leverages Python for data collection, processing, and 
visualization to analyze crime rates across multiple urban areas. By utilizing datasets from 
government and law enforcement agencies, we employ statistical and machine learning 
techniques to identify correlations between crime rates and socio-economic factors such as 
population density, income levels, and unemployment rates. Various Python libraries, including 
Pandas, NumPy, Matplotlib, and Scikit-learn, are used for data manipulation, visualization, and 
predictive modeling. The analysis aims to provide insights into crime trends, aiding policymakers 
and law enforcement agencies in strategic decision-making for crime prevention and urban 
planning. 
# Code(APP)
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import base64

# --- FILE PATHS ---
crime_data_path = "dataset/india_crime_data.csv"
geojson_path = "dataset/india_states.geojson"

# --- CACHING DATA ---
@st.cache_resource
def load_data():
    """Load crime dataset and India map"""
    try:
        df = pd.read_csv(crime_data_path)
        df.columns = df.columns.str.lower().str.strip()
    except FileNotFoundError:
        st.error(f"❌ Error: Crime dataset not found at {crime_data_path}")
        return None, None, None

    required_columns = ["state", "year", "crime type", "crime rate (per 100k)"]
    if not all(col in df.columns for col in required_columns):
        st.error(f"❌ Dataset missing columns: {set(required_columns) - set(df.columns)}")
        return None, None, None

    try:
        india_map = gpd.read_file(geojson_path)
    except FileNotFoundError:
        st.error(f"❌ Error: GeoJSON file not found at {geojson_path}")
        return None, None, None

    possible_columns = ["NAME_1", "NAME_0", "VARNAME_1"]
    state_column = next((col for col in possible_columns if col in india_map.columns), None)

    if state_column is None:
        st.error(f"❌ No valid state column found! Available: {india_map.columns.tolist()}")
        return None, None, None

    india_map[state_column] = india_map[state_column].str.lower()
    df["state"] = df["state"].str.lower()

    return df, india_map, state_column

# --- LOAD DATA ---
df, india_map, state_column = load_data()
if df is None or india_map is None:
    st.stop()

# --- FUNCTION TO ENCODE IMAGE TO BASE64 ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None  # Avoid errors if the file is missing

# --- FILE PATHS FOR BACKGROUNDS ---
background_page_img = "assets/background.jpg"
background_map_img = "assets/map_bg.jpg"
background_graph_img = "assets/graph_bg.jpg"

# --- APPLY CSS FOR STYLING ---
page_bg_base64 = get_base64_image(background_page_img)
map_bg_base64 = get_base64_image(background_map_img)
graph_bg_base64 = get_base64_image(background_graph_img)

if page_bg_base64:
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{page_bg_base64}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# --- STREAMLIT UI ---
st.title("📌 Crime Rate Analysis in Indian States")

# --- FILTER OPTIONS ---
with st.sidebar:
    st.header("🔍 Filter Options")
    selected_state = st.selectbox("Select State", ["All"] + sorted(df["state"].unique()))
    selected_year = st.selectbox("Select Year", ["All"] + sorted(df["year"].unique()))
    selected_crime = st.selectbox("Select Crime Type", ["All"] + sorted(df["crime type"].unique()))

# --- APPLY FILTERS ---
filtered_df = df.copy()
if selected_state != "All":
    filtered_df = filtered_df[filtered_df["state"] == selected_state.lower()]
if selected_year != "All":
    filtered_df = filtered_df[filtered_df["year"] == selected_year]
if selected_crime != "All":
    filtered_df = filtered_df[filtered_df["crime type"] == selected_crime]

# --- AGGREGATE CRIME RATE ---
state_crime = filtered_df.groupby("state", as_index=False)["crime rate (per 100k)"].mean()

# Merge with India map
merged = india_map.set_index(state_column).join(state_crime.set_index("state")).reset_index()

if "crime rate (per 100k)" not in merged.columns:
    st.error("❌ 'crime rate (per 100k)' column missing after merge. Check state name mismatches.")
    st.stop()

merged["crime rate (per 100k)"] = merged["crime rate (per 100k)"].fillna(0)

# --- DISPLAY SELECTED FILTERS ---
st.markdown("## Selected Filters")
st.markdown(f"- **State:** {selected_state}")
st.markdown(f"- **Year:** {selected_year}")
st.markdown(f"- **Crime Type:** {selected_crime}")

# --- MAP VISUALIZATION ---
st.subheader("🗺️ Crime Rate in Indian States")
with st.container():
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    choropleth = folium.Choropleth(
        geo_data=india_map,
        name="Crime Rate",
        data=state_crime,
        columns=["state", "crime rate (per 100k)"],
        key_on=f"feature.properties.{state_column}",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Crime Rate (per 100K)"
    ).add_to(m)

    # Fix mouse hover delay by adding tooltip labels to each state
    folium.GeoJsonTooltip(fields=[state_column]).add_to(choropleth.geojson)

    # Render map
    st_folium(m, width=800, height=500)
    st.markdown('</div>', unsafe_allow_html=True)

# --- BAR CHART VISUALIZATION ---
st.subheader("📊 Crime Rate Distribution")
with st.container():
    st.markdown('<div class="graph-container">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    if not merged.empty:
        merged.sort_values("crime rate (per 100k)", ascending=False).plot(
            kind="bar", x=state_column, y="crime rate (per 100k)", legend=False, ax=ax, color="red"
        )
        plt.xticks(rotation=90)
        plt.ylabel("Crime Rate per 100K")
        plt.xlabel("State")
        plt.title("Crime Rate per 100K Population by State")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No data available for the selected filters.")
    
    st.markdown('</div>', unsafe_allow_html=True)
# Code(CHECK COLUMNS)
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/india_crime_data.csv")

# Print column names
print("Column names in the dataset:", df.columns.tolist())
# Code(CHECH_JEOJSON)
import geopandas as gpd

# Load the India map GeoJSON file
geojson_path = "dataset/india_states.geojson"

try:
    india_map = gpd.read_file(geojson_path)
    print("✅ India map loaded successfully!")
    print("🔍 Available columns in GeoJSON:", india_map.columns.tolist())  # Print column names
except Exception as e:
    print("❌ Error loading file:", e)

# Code(MAP_PLOT)
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# File paths (Ensure these files exist)
crime_data_path = "dataset/india_crime_data.csv"
geojson_path = "dataset/india_states.geojson"

# ✅ Load crime dataset
try:
    df = pd.read_csv(crime_data_path)
    print("✅ Crime dataset loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: Crime dataset not found at {crime_data_path}")
    exit()

# 🔹 Convert column names to lowercase & strip spaces
df.columns = df.columns.str.lower().str.strip()

# 🔹 Ensure "state" column exists
if "state" not in df.columns:
    print("❌ Error: 'state' column not found in dataset. Available columns:", df.columns.tolist())
    exit()

# ✅ Aggregate crime rate per state
state_crime = df.groupby("state")["crime rate (per 100k)"].mean().reset_index()

# ✅ Load India map GeoJSON file
try:
    india_map = gpd.read_file(geojson_path)
    print("✅ India map loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: GeoJSON file not found at {geojson_path}")
    exit()

# 🔹 Use "NAME_1" as the state column
state_column = "NAME_1"  # Based on available columns
print(f"✅ Using '{state_column}' as the state column in GeoJSON.")

# 🔹 Ensure correct format for merging
india_map[state_column] = india_map[state_column].str.lower()
state_crime["state"] = state_crime["state"].str.lower()

# ✅ Merge data with map
merged = india_map.set_index(state_column).join(state_crime.set_index("state"))

# 🔹 Fill missing values (states with no crime data)
merged["crime rate (per 100k)"] = merged["crime rate (per 100k)"].fillna(0)

# ✅ **Plot Map with Crime Rate**
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
merged.plot(column="crime rate (per 100k)", cmap="RdYlGn_r", linewidth=0.8, edgecolor="black", legend=True, ax=ax)

# 🔹 Title & display
plt.title("Crime Rate Per 100K Population in Indian States", fontsize=14)
plt.axis("off")
plt.show()
# Code(MODEL)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("dataset/crime_data.csv")

# Selecting features
X = df[["Population", "Incidents"]]
y = df["Crime Rate (per 100K)"]

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
model = LinearRegression()
model.fit(X_train, y_train)

# Save trained model
with open("crime_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
# Code(PRE-PROCESS)
import pandas as pd

def load_and_preprocess_data(file_path):
    """Loads the dataset and applies basic preprocessing."""
    df = pd.read_csv(file_path)

    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Convert column names to lowercase and strip spaces
    df.columns = df.columns.str.lower().str.strip()

    return df

if __name__ == "__main__":
    file_path = "dataset/india_crime_data.csv"
    df = load_and_preprocess_data(file_path)
    print("Dataset Loaded Successfully!")
    print(df.head())
# Code(VISUALIZE)

import pandas as pd
import matplotlib.pyplot as plt



# Load dataset and fix column names
df = pd.read_csv("dataset/india_crime_data.csv")
df.columns = df.columns.str.lower().str.strip()

# Group data by state
crime_per_state = df.groupby("state")["incidents"].sum().sort_values(ascending=False)
print(crime_per_state.head())


# Bar Chart - Total Crimes per State
plt.figure(figsize=(12,6))
crime_per_state.plot(kind="bar", color="orange")
plt.title("Total Crimes per State (2001-2024)")
plt.xlabel("State")
plt.ylabel("Total Incidents")
plt.xticks(rotation=75)
plt.grid()
plt.show()


