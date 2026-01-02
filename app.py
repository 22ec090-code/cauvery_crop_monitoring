import streamlit as st
import ee
import geemap
import joblib
import numpy as np
from datetime import date

# ----------- AUTHENTICATE EE (FIRST TIME YOU'LL DO ee.Authenticate()) ----------
import ee
import os

# Authenticate using service account
key_file = "service_account.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file

ee.Initialize()
Initialize()

# ----------- STUDY AREA ----------
cauvery_delta = ee.Geometry.Rectangle([78.0, 10.3, 79.8, 11.6])

# ----------- LOAD MODEL ----------
model = joblib.load("model.pkl")

# ----------- STREAMLIT UI ----------
st.title("Cauvery Delta — AI Crop Monitoring System")
st.write("Live Crop Classification from Sentinel-2 + Machine Learning")

start = st.date_input("Start date", date(2024, 6, 1))
end = st.date_input("End date", date(2024, 6, 15))

# ----------- FETCH SENTINEL-2 ----------
s2 = (ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(cauvery_delta)
        .filterDate(str(start), str(end))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .median())

# ----------- NDVI ----------
ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')

# ----------- SELECT FEATURES ----------
features = s2.select(['B2','B3','B4','B8','B11','B12']).addBands(ndvi)

# ----------- SAMPLE POINTS ----------
sample = features.sample(
    region=cauvery_delta,
    scale=10,
    numPixels=5000,
    geometries=True
).getInfo()

# Convert to numpy for prediction
X = []
coords = []

for f in sample['features']:
    props = f['properties']
    X.append([
        props['B2'], props['B3'], props['B4'],
        props['B8'], props['B11'], props['B12'],
        props['NDVI']
    ])
    coords.append(f['geometry']['coordinates'])

X = np.array(X)

# ----------- PREDICT ----------
y_pred = model.predict(X)

# ----------- MAP ----------
m = geemap.Map(center=[10.8, 78.8], zoom=8)

crop_layer = geemap.ee_vector_to_ee(sample)

m.addLayer(ndvi, {'min':0, 'max':1, 'palette':['red','yellow','green']}, "NDVI")

st.write(" NDVI Layer Added")

m.to_streamlit(height=600)

