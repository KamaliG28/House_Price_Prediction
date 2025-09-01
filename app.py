import streamlit as st
import pandas as pd
import seaborn as sns

# Title
st.title("🏠 House Price Prediction App")

# Load data
csv_path = "C:/Users/gks/Downloads/Deploy/house_prices.csv"
try:
    df = pd.read_csv(csv_path)
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error("CSV file not found. Please check the path.")
    st.stop()

# Show raw data
if st.checkbox("Show raw data"):
    st.write(df)

# Basic stats
st.subheader("📊 Summary Statistics")
st.write(df.describe())

# Select column to visualize
numeric_cols = df.select_dtypes(include='number').columns.tolist()
selected_col = st.selectbox("Select a column to visualize", numeric_cols)

# Histogram
st.subheader(f"📈 Distribution of {selected_col}")
fig, ax = plt.subplots()
sns.histplot(df[selected_col], kde=True, ax=ax)
st.pyplot(fig)

# Correlation heatmap
st.subheader("🔍 Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

