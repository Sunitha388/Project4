import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity  #It measures the similarity between vectors based on their angle in a multi-dimensional space.
from sklearn.cluster import KMeans     # It groups restaurants into clusters based on feature similarity.

# --- Load data ---
@st.cache_data
def load_data():
    cleaned = pd.read_csv(r"C:\Users\LENOVO\Desktop\Newfolder\Project4\output4\cleaned_data1.csv")
    encoded = pd.read_csv(r"C:\Users\LENOVO\Desktop\Newfolder\Project4\output4\encoded_numeric.csv")
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return cleaned, encoded, encoder

cleaned_df, encoded_df, encoder = load_data()

st.title("🍽️ Restaurant Recommendation System")

# --- Sidebar user inputs ---
st.sidebar.header("🔎 Search Preferences")

cities = cleaned_df['city'].dropna().unique()
cuisines = cleaned_df['cuisine'].dropna().unique()

city_choice = st.sidebar.selectbox("Select City", options=np.append(["Any"], cities))
cuisine_choice = st.sidebar.selectbox("Preferred Cuisine", options=np.append(["Any"], cuisines))
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.0, step=0.1)
max_price = st.sidebar.number_input("Maximum Price (Approx.)", min_value=0, value=1000)

if st.sidebar.button("Get Recommendations"):
    # --- Filter cleaned data based on user input ---
    filtered = cleaned_df.copy()

    if city_choice != "Any":
        filtered = filtered[filtered['city'] == city_choice]
    if cuisine_choice != "Any":
        filtered = filtered[filtered['cuisine'] == cuisine_choice]
    filtered = filtered[filtered['rating'] >= min_rating]
    filtered = filtered[filtered['cost'] <= max_price]

    if filtered.empty:
        st.warning("❌ No restaurants match your preferences.")
    else:
        # --- Generate recommendations using similarity ---
        encoded_filtered = encoded_df.loc[filtered.index]
        similarity_matrix = cosine_similarity(encoded_filtered, encoded_filtered)
        similarity_scores = similarity_matrix[0]

        sorted_indices = np.argsort(similarity_scores)[::-1]
        recommended_similarity = cleaned_df.iloc[sorted_indices]

        # --- Clustering-based recommendations ---
        num_clusters = 10  # Adjust cluster count as needed
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(encoded_df)

        idx = sorted_indices[0]  # Reference restaurant
        cluster = cluster_labels[idx]

        cluster_members = np.where(cluster_labels == cluster)[0]
        recommendations = [i for i in cluster_members if i != idx][:5]  # Exclude the reference

        recommended_cluster = cleaned_df.iloc[recommendations]

        # --- Display Results ---
        st.success(f"✅ Found {len(recommended_similarity)} similarity-based recommendations:")
        st.dataframe(recommended_similarity[['name', 'city', 'cuisine', 'rating', 'cost', 'address']].reset_index(drop=True))

        st.success(f"✅ Found {len(recommended_cluster)} cluster-based recommendations:")
        st.dataframe(recommended_cluster[['name', 'city', 'cuisine', 'rating', 'cost', 'address']].reset_index(drop=True))

else:
    st.info("Please set your preferences and click **Get Recommendations**.")



