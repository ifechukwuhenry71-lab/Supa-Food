import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="Nigerian Food Recommender", page_icon="🇳🇬", layout="wide"
)

st.title("🇳🇬 Nigerian Food Recommendation System")
st.write(
    "Discover delicious Nigerian dishes tailored to your taste, region, and"
    " preferences using smart content-based filtering!"
)


# Error-trapped data loading function
@st.cache_data
def load_data():
  file_path = r"C:\Users\User\Desktop\supa-foods-app\dataset.csv.csv"
  if not os.path.exists(file_path):
    st.error(
        f"🚨 Error: '{file_path}' not found in your project folder. Please"
        " make sure your dataset is placed in the root directory alongside"
        " app.py."
    )
    return None

  try:
    df = pd.read_csv(file_path)
    return df
  except Exception as e:
    st.error(f"🚨 Error loading dataset: {e}")
    return None


df = load_data()

if df is not None:
  # Validate that all 8 required columns exist in the dataset
  expected_columns = [
      "Food_Name",
      "Main_Ingredients",
      "Description",
      "Food_Health",
      "Food_Class",
      "Region",
      "Spice_Level",
      "Price_Range",
  ]
  missing_cols = [col for col in expected_columns if col not in df.columns]

  if missing_cols:
    st.error(
        f"🚨 Error: Your dataset is missing required columns: {missing_cols}"
    )
  else:
    # Handle missing values to prevent processing errors
    df = df.fillna("")

    # Combine text features for TF-IDF Vectorization
    df["combined_features"] = (
        df["Food_Name"].astype(str)
        + " "
        + df["Main_Ingredients"].astype(str)
        + " "
        + df["Description"].astype(str)
        + " "
        + df["Food_Health"].astype(str)
        + " "
        + df["Food_Class"].astype(str)
        + " "
        + df["Region"].astype(str)
        + " "
        + df["Spice_Level"].astype(str)
        + " "
        + df["Price_Range"].astype(str)
    )

    # Compute TF-IDF matrix and Cosine Similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Sidebar Navigation and Preferences
    st.sidebar.header("🔍 Recommendation Modes")
    search_mode = st.sidebar.radio(
        "Choose how you want to search:",
        ["Find Similar Dishes", "Filter by Preferences"],
    )

    if search_mode == "Find Similar Dishes":
      selected_food = st.sidebar.selectbox(
          "Select a Dish You Like:", df["Food_Name"].values
      )

      if st.sidebar.button("Get Recommendations"):
        idx = df[df["Food_Name"] == selected_food].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top 5 similar recommendations (skipping the food itself at index 0)
        sim_scores = sim_scores[1:6]
        food_indices = [i[0] for i in sim_scores]

        st.subheader(
            f"Because you like **{selected_food}**, you might also enjoy:"
        )

        for i in food_indices:
          dish = df.iloc[i]
          with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
              st.image(
                  "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=500&auto=format&fit=crop&q=60",
                  use_container_width=True,
              )
            with col2:
              st.markdown(f"### 🍲 {dish['Food_Name']}")
              st.write(f"**Description:** {dish['Description']}")
              st.write(f"**Main Ingredients:** {dish['Main_Ingredients']}")
              st.markdown(
                  f"*Region:* {dish['Region']} | *Spice Level:*"
                  f" {dish['Spice_Level']} | *Price:* {dish['Price_Range']}"
              )
            st.divider()

    else:
      # Interactive Filter Mode
      regions = ["All"] + list(df["Region"].unique())
      spice_levels = ["All"] + list(df["Spice_Level"].unique())
      price_ranges = ["All"] + list(df["Price_Range"].unique())

      selected_region = st.sidebar.selectbox("Region", regions)
      selected_spice = st.sidebar.selectbox("Spice Level", spice_levels)
      selected_price = st.sidebar.selectbox("Price Range", price_ranges)

      filtered_df = df.copy()
      if selected_region != "All":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]
      if selected_spice != "All":
        filtered_df = filtered_df[filtered_df["Spice_Level"] == selected_spice]
      if selected_price != "All":
        filtered_df = filtered_df[filtered_df["Price_Range"] == selected_price]

      st.subheader(f"Found {len(filtered_df)} matching Nigerian dishes:")

      if filtered_df.empty:
        st.warning(
            "No dishes match your exact combination of filters. Try adjusting"
            " them!"
        )
      else:
        for _, dish in filtered_df.iterrows():
          with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
              st.image(
                  "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=500&auto=format&fit=crop&q=60",
                  use_container_width=True,
              )
            with col2:
              st.markdown(f"### 🍲 {dish['Food_Name']}")
              st.write(f"**Description:** {dish['Description']}")
              st.write(f"**Main Ingredients:** {dish['Main_Ingredients']}")
              st.markdown(
                  f"*Region:* {dish['Region']} | *Spice Level:*"
                  f" {dish['Spice_Level']} | *Price:* {dish['Price_Range']}"
              )
            st.divider()