# app.py
# ML Practical 1 – KNN Weather Classification

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier


# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Weather Classifier",
    layout="wide"
)


# -------------------------------
# Title & description
# -------------------------------
st.title("K-Nearest Neighbour Weather Classification")

st.markdown("""
Hello everyone, let’s proceed.

This app uses **K-Nearest Neighbour (KNN)** from scikit-learn to classify
weather conditions based on **temperature** and **humidity**.
""")


# -------------------------------
# Training data
# -------------------------------
# Features: [Temperature, Humidity]
X = np.array([
    [30, 70],
    [27, 60],
    [31, 65],
    [23, 85],
    [20, 75]
])

# Labels: 0 = Sunny, 1 = Rainy
y = np.array([0, 1, 0, 1, 1])


# -------------------------------
# Label mapping
# -------------------------------
label_map = {
    0: "Sunny",
    1: "Rainy"
}


# -------------------------------
# Sidebar – user input
# -------------------------------
st.sidebar.header("Input Parameters")

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=20,
    max_value=35,
    value=26,
    step=1
)

humidity = st.sidebar.slider(
    "Humidity (%)",
    min_value=30,
    max_value=90,
    value=75,
    step=1
)

k = st.sidebar.slider(
    "K value (Number of Neighbours)",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)


# -------------------------------
# Train the model
# -------------------------------
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)


# -------------------------------
# Make prediction
# -------------------------------
new_weather = np.array([[temperature, humidity]])

pred = knn.predict(new_weather)[0]
pred_proba = knn.predict_proba(new_weather)[0]

weather_label = label_map[pred]
confidence = pred_proba[pred] * 100


# -------------------------------
# Sidebar – result
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Result")

if pred == 0:
    st.sidebar.success(f"Weather : {weather_label}")
else:
    st.sidebar.info(f"Weather : {weather_label}")

st.sidebar.metric("Confidence", f"{confidence:.2f} %")


# -------------------------------
# Main content – visualization
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Classification Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Separate points
    sunny = X[y == 0]
    rainy = X[y == 1]

    ax.scatter(
        sunny[:, 0], sunny[:, 1],
        color="orange",
        label="Sunny",
        s=100,
        edgecolor="k",
        alpha=0.7
    )

    ax.scatter(
        rainy[:, 0], rainy[:, 1],
        color="blue",
        label="Rainy",
        s=100,
        edgecolor="k",
        alpha=0.7
    )

    # New prediction point
    colors = ["orange", "blue"]
    ax.scatter(
        new_weather[0, 0],
        new_weather[0, 1],
        color=colors[pred],
        s=300,
        edgecolor="black",
        marker="*",
        label=f"New day ({weather_label})",
        zorder=5
    )

    ax.set_xlabel("Temperature (°C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Humidity (%)", fontsize=12, fontweight="bold")
    ax.set_title("Weather Classification Model", fontsize=14, fontweight="bold")

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(20, 35)
    ax.set_ylim(30, 90)

    st.pyplot(fig)


with col2:
    st.subheader("Input Summary")

    st.write("Temperature :", temperature, "°C")
    st.write("Humidity :", humidity, "%")
    st.write("K value :", k)
    st.write("Predicted Weather :", weather_label)
    st.write("Confidence :", f"{confidence:.2f} %")
