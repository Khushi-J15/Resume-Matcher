import streamlit as st
import pandas as pd
import os
from io import BytesIO
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Admin Dashboard", page_icon="📊", layout="centered")
st.title("📊 Admin Dashboard")

# Load similarity data from file
SIMILARITY_DATA_FILE = "similarity_data4.pkl"

if os.path.exists(SIMILARITY_DATA_FILE):
    with open(SIMILARITY_DATA_FILE, "rb") as f:
        df_sim = pickle.load(f)

    st.markdown("### 📊 Resume Similarity Chart")
    st.dataframe(df_sim.head(10))

    st.bar_chart(df_sim.set_index("Name").head(10))

    if st.checkbox("Show Pie Chart of Top 5 Similarity"):
        top5 = df_sim.head(5)
        fig, ax = plt.subplots()
        ax.pie(top5["Similarity"], labels=top5["Name"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
else:
    st.info("ℹ️ Run the main resume matcher to generate similarity data.")
