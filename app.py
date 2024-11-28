import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the saved scaler and standardized dataset
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load your dataset
data_standardized = pd.read_csv("data_standardized.csv")  # Update with your file path
features = ['danceability', 'energy', 'tempo', 'popularity']  # Features used in the recommendation

# Recommendation function
def recommend_songs(input_track, input_artist, data, features, num_recommendations=3):
    """
    Recommend songs based on the input track and artist.
    """
    # Find the input track in the dataset
    input_data = data[
        (data['track_name'] == input_track) & 
        (data['artist_name'] == input_artist)
    ]
    if input_data.empty:
        return None

    # Extract features of the input track
    input_features = input_data[features].values
    # Calculate similarity
    similarity_scores = cosine_similarity(input_features, data[features])
    data['similarity'] = similarity_scores[0]

    # Sort by similarity and exclude the input track
    recommendations = data[
        (data['track_name'] != input_track) | 
        (data['artist_name'] != input_artist)
    ].sort_values(by='similarity', ascending=False).head(num_recommendations)

    return recommendations[['track_name', 'artist_name', 'similarity']]

# Streamlit app
st.title("Song Recommender")

# User selects a song and artist
st.write("Search for a song to get recommendations:")

# Dropdown to select track
input_track = st.selectbox("Select a song:", data_standardized['track_name'].unique())
# Dropdown to select artist based on the selected track
possible_artists = data_standardized[data_standardized['track_name'] == input_track]['artist_name'].unique()
input_artist = st.selectbox("Select the artist:", possible_artists)

# Button to get recommendations
if st.button("Search and Recommend"):
    if input_track and input_artist:
        # Get recommendations
        recommendations = recommend_songs(input_track, input_artist, data_standardized, features)

        if recommendations is not None and not recommendations.empty:
            st.write(f"### Recommended Songs for: **{input_track}** by **{input_artist}**")
            for idx, row in recommendations.iterrows():
                st.write(f"- **{row['track_name']}** by {row['artist_name']}")
        else:
            st.warning("No recommendations found. Try selecting another song or artist.")
    else:
        st.warning("Please select a song and an artist.")

# Footer
st.markdown("Thank you ❤️ ")