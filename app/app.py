import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tmdb_helper import get_movie_details, get_movie_details_batch

load_dotenv()

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="CineMatch - Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# NETFLIX-STYLE CSS
# ============================================
st.markdown("""
<style>
    .stApp {
        background-color: #141414;
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #333;
    }
    .movie-card {
        background-color: #1f1f1f;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #333;
        margin-bottom: 16px;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: scale(1.02);
        border: 1px solid #e50914;
    }
    .movie-poster {
        width: 100%;
        border-radius: 8px 8px 0 0;
        min-height: 280px;
        object-fit: cover;
    }
    .movie-info {
        padding: 12px;
    }
    .movie-title {
        font-size: 14px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 6px;
        line-height: 1.3;
        min-height: 36px;
    }
    .movie-rating {
        color: #f5c518;
        font-size: 13px;
        margin-bottom: 4px;
    }
    .movie-year {
        color: #999;
        font-size: 12px;
        margin-bottom: 6px;
    }
    .genre-tag {
        display: inline-block;
        background-color: #e50914;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        margin: 2px;
    }
    .movie-overview {
        color: #ccc;
        font-size: 12px;
        line-height: 1.4;
        margin-top: 8px;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .main-header {
        background: linear-gradient(135deg, #e50914, #b20710);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 48px;
        font-weight: 900;
        margin: 0;
        letter-spacing: -1px;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 18px;
        margin: 10px 0 0 0;
    }
    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #ffffff;
        margin: 30px 0 15px 0;
        padding-left: 10px;
        border-left: 4px solid #e50914;
    }
    .search-result-card {
        background-color: #1f1f1f;
        border-radius: 8px;
        padding: 16px;
        border: 2px solid #e50914;
        margin-bottom: 20px;
    }
    .search-result-title {
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
    }
    .search-result-meta {
        color: #999;
        font-size: 14px;
        margin-top: 4px;
    }
    .no-results {
        text-align: center;
        color: #999;
        padding: 40px;
        font-size: 16px;
    }
            
    /* Skeleton loading card */
    .skeleton-card {
        background-color: #1f1f1f;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #333;
        margin-bottom: 16px;
    }
    .skeleton-img {
        width: 100%;
        height: 280px;
        background: linear-gradient(90deg, #2a2a2a 25%, #333 50%, #2a2a2a 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
    }
    .skeleton-text {
        padding: 12px;
    }
    .skeleton-line {
        height: 12px;
        border-radius: 4px;
        margin-bottom: 8px;
        background: linear-gradient(90deg, #2a2a2a 25%, #333 50%, #2a2a2a 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
    }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* Recommendation count badge */
    .rec-badge {
        display: inline-block;
        background-color: #e50914;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: bold;
        margin-left: 10px;
        vertical-align: middle;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #666;
    }
    .empty-state-icon {
        font-size: 48px;
        margin-bottom: 16px;
    }
    .empty-state-text {
        font-size: 18px;
        color: #999;
        margin-bottom: 8px;
    }
    .empty-state-subtext {
        font-size: 14px;
        color: #666;
    }
                    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD DATA & BUILD MODEL
# ============================================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ratings = pd.read_csv(os.path.join(base_dir, 'data', 'ratings_clean.csv'))
    movies = pd.read_csv(os.path.join(base_dir, 'data', 'movies_clean.csv'))
    links = pd.read_csv(os.path.join(base_dir, 'data', 'ml-latest-small', 'links.csv'))
    return ratings, movies, links

@st.cache_data
def build_model():
    ratings, movies, links = load_data()

    user_movie_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)

    sparse_matrix = csr_matrix(user_movie_matrix.values)
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_factors = svd.fit_transform(sparse_matrix)

    user_similarity = cosine_similarity(user_factors)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )

    return user_movie_matrix, user_similarity_df, movies, links, ratings


def search_movies(query, movies, top_n=8):
    """Search for movies by name"""
    query = query.lower().strip()
    matches = movies[movies['title'].str.lower().str.contains(query, na=False)]
    return matches.head(top_n)


def get_recommendations_from_movie(movie_id, user_movie_matrix, user_similarity_df, movies, links, num_recommendations=10):
    """
    Given a movie, find users who loved it,
    then recommend what else those users loved
    """
    # Find users who rated this movie highly (4.0+)
    movie_lovers = user_movie_matrix[user_movie_matrix[movie_id] >= 4.0].index.tolist()

    if not movie_lovers:
        # Fallback: find users who rated it at all
        movie_lovers = user_movie_matrix[user_movie_matrix[movie_id] > 0].index.tolist()

    if not movie_lovers:
        return pd.DataFrame()

    # For each lover, find their similar users
    recommended_movies = {}

    for lover_id in movie_lovers[:20]:  # limit to top 20 lovers
        # Find users similar to this lover
        similar_users = user_similarity_df[lover_id].sort_values(ascending=False)
        similar_users = similar_users.drop(lover_id)
        top_similar = similar_users.head(5).index.tolist()

        for similar_user in top_similar:
            similarity_score = user_similarity_df[lover_id][similar_user]

            # Get their highly rated movies
            highly_rated = user_movie_matrix.columns[
                user_movie_matrix.loc[similar_user] >= 4.0
            ].tolist()

            for mid in highly_rated:
                if mid != movie_id:  # Don't recommend the same movie
                    if mid not in recommended_movies:
                        recommended_movies[mid] = 0
                    recommended_movies[mid] += similarity_score

    if not recommended_movies:
        return pd.DataFrame()

    # Get top recommendations
    top_movie_ids = sorted(
        recommended_movies,
        key=recommended_movies.get,
        reverse=True
    )[:num_recommendations]

    recommendations = movies[movies['movieId'].isin(top_movie_ids)].copy()
    recommendations = recommendations.merge(
        links[['movieId', 'tmdbId']], on='movieId', how='left'
    )
    recommendations['score'] = recommendations['movieId'].map(recommended_movies)
    recommendations = recommendations.sort_values('score', ascending=False)

    return recommendations


def get_popular_movies(ratings, movies, links, n=10):
    """Get most popular movies for homepage"""
    popular = ratings.groupby('movieId').agg(
        rating_count=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    popular = popular[popular['rating_count'] >= 50]
    popular['score'] = popular['rating_count'] * popular['avg_rating']
    popular = popular.sort_values('score', ascending=False).head(n)
    popular = popular.merge(movies[['movieId', 'title']], on='movieId')
    popular = popular.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    return popular

def render_skeleton_cards(n=10):
    """Render placeholder cards while content loads"""
    cols = st.columns(5)
    for i in range(n):
        with cols[i % 5]:
            st.markdown("""
            <div class="skeleton-card">
                <div class="skeleton-img"></div>
                <div class="skeleton-text">
                    <div class="skeleton-line" style="width:80%"></div>
                    <div class="skeleton-line" style="width:50%"></div>
                    <div class="skeleton-line" style="width:65%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_movie_card(details):
    """Render a Netflix-style movie card"""
    genres_html = ''.join([
        f'<span class="genre-tag">{g}</span>'
        for g in details['genres'][:2]
    ])
    rating = details['rating']
    stars = '⭐' * min(round(rating / 2), 5)

    return f"""
    <div class="movie-card">
        <img 
            src="{details['poster']}" 
            class="movie-poster"
            onerror="this.src='https://placehold.co/500x750/1f1f1f/999999?text=No+Poster'"
        />
        <div class="movie-info">
            <div class="movie-title">{details['title']}</div>
            <div class="movie-rating">{stars} {rating}/10</div>
            <div class="movie-year">📅 {details['release_date']}</div>
            <div style="margin: 6px 0">{genres_html}</div>
            <div class="movie-overview">{details['overview']}</div>
        </div>
    </div>
    """


# ============================================
# MAIN APP
# ============================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🎬 CineMatch</h1>
    <p>Type a movie you love — we'll find your next favourite</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("🎬 Loading CineMatch engine..."):
    user_movie_matrix, user_similarity_df, movies, links, ratings = build_model()

# ============================================
# SEARCH BAR
# ============================================
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    query = st.text_input(
        "",
        placeholder="🔍 Search a movie you love... e.g. Inception, Titanic, The Matrix",
        label_visibility="collapsed"
    )

# ============================================
# SEARCH RESULTS & RECOMMENDATIONS
# ============================================
if query and len(query) >= 2:
    matches = search_movies(query, movies)

    if matches.empty:
        st.markdown(f'<div class="no-results">😔 No movies found for "<b>{query}</b>". Try a different title!</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="section-header">🔍 Search Results for "{query}"</div>', unsafe_allow_html=True)

        # Show search results as selectable options
        movie_options = dict(zip(matches['title'], matches['movieId']))
        selected_title = st.selectbox(
            "Select the movie you meant:",
            options=list(movie_options.keys()),
            label_visibility="collapsed"
        )

        selected_movie_id = movie_options[selected_title]

        # Fetch and show selected movie details
        selected_tmdb = links[links['movieId'] == selected_movie_id]['tmdbId'].values
        if len(selected_tmdb) > 0 and str(selected_tmdb[0]) != 'nan':
            with st.spinner("Loading movie details..."):
                selected_details = get_movie_details(selected_tmdb[0])

            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"""
                <div class="search-result-card">
                    <img src="{selected_details['poster']}" style="width:100%; border-radius:8px; margin-bottom:12px"/>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="search-result-card">
                    <div class="search-result-title">{selected_details['title']}</div>
                    <div class="search-result-meta">
                        ⭐ {selected_details['rating']}/10 &nbsp;|&nbsp; 
                        📅 {selected_details['release_date']} &nbsp;|&nbsp;
                        🎭 {', '.join(selected_details['genres'])}
                    </div>
                    <p style="color:#ccc; margin-top:12px; line-height:1.6">{selected_details['overview']}</p>
                </div>
                """, unsafe_allow_html=True)

        # Get & show recommendations
        rec_title = selected_title.split("(")[0].strip()
        st.markdown(f'<div class="section-header">✨ Because you liked {rec_title}... <span class="rec-badge">AI Picks</span></div>', unsafe_allow_html=True)

        with st.spinner("Finding recommendations..."):
            recs = get_recommendations_from_movie(
                selected_movie_id,
                user_movie_matrix,
                user_similarity_df,
                movies,
                links,
                num_recommendations=10
            )

        if not recs.empty:
            # ---- GENRE FILTER ----
            # Extract all unique genres from recommendations
            all_genres = set()
            for genre_str in recs['genres'].dropna():
                for g in genre_str.split('|'):
                    all_genres.add(g.strip())
            all_genres = sorted(all_genres)

            st.markdown("**🎭 Filter by Genre:**")
            selected_genres = st.multiselect(
                "",
                options=all_genres,
                default=[],
                placeholder="Select genres to filter...",
                label_visibility="collapsed"
            )

            # Apply genre filter if any selected
            if selected_genres:
                filtered_recs = recs[
                    recs['genres'].apply(
                        lambda x: any(g in str(x) for g in selected_genres)
                    )
                ]
            else:
                filtered_recs = recs

            if filtered_recs.empty:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">🎭</div>
                    <div class="empty-state-text">No matches for selected genres</div>
                    <div class="empty-state-subtext">Try removing some genre filters</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Show count badge
                st.markdown(f'<p style="color:#999; margin-bottom:16px">Showing <b style="color:white">{len(filtered_recs)}</b> recommendations</p>', unsafe_allow_html=True)

                tmdb_ids = filtered_recs['tmdbId'].tolist()
                rec_details = get_movie_details_batch(tmdb_ids)

                cols = st.columns(5)
                for i, details in enumerate(rec_details):
                    with cols[i % 5]:
                        st.markdown(render_movie_card(details), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🎬</div>
                <div class="empty-state-text">No recommendations found</div>
                <div class="empty-state-subtext">This movie may not have enough ratings. Try a more popular title!</div>
            </div>
            """, unsafe_allow_html=True)
            
# ============================================
# POPULAR MOVIES (shown when no search)
# ============================================
else:
    st.markdown('<div class="section-header">🔥 Popular Movies to Get You Started</div>', unsafe_allow_html=True)

    with st.spinner("Loading popular movies..."):
        popular = get_popular_movies(ratings, movies, links, n=10)
        tmdb_ids = popular['tmdbId'].tolist()
        popular_details = get_movie_details_batch(tmdb_ids)

    if popular_details:
        cols = st.columns(5)
        for i, details in enumerate(popular_details):
            with cols[i % 5]:
                st.markdown(render_movie_card(details), unsafe_allow_html=True)