import requests
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
API_KEY = os.getenv("API_KEY")

BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER = "https://placehold.co/500x750/1f1f1f/999999?text=No+Poster"

def get_movie_details(tmdb_id):
    """Fetch full movie details from TMDB"""
    try:
        url = f"{BASE_URL}/movie/{int(float(tmdb_id))}?api_key={API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()

        poster = f"{POSTER_BASE_URL}{data['poster_path']}" if data.get('poster_path') else PLACEHOLDER

        return {
            'tmdb_id': tmdb_id,
            'title': data.get('title', 'Unknown'),
            'overview': data.get('overview', 'No description available'),
            'rating': round(data.get('vote_average', 0), 1),
            'release_date': data.get('release_date', 'N/A')[:4] if data.get('release_date') else 'N/A',
            'poster': poster,
            'genres': [g['name'] for g in data.get('genres', [])],
        }
    except Exception as e:
        return {
            'tmdb_id': tmdb_id,
            'title': 'Unknown',
            'overview': 'No description available',
            'rating': 0,
            'release_date': 'N/A',
            'poster': PLACEHOLDER,
            'genres': [],
        }

def get_movie_details_batch(tmdb_ids):
    """Fetch details for multiple movies IN PARALLEL"""
    # Filter out invalid IDs
    valid_ids = [
        tid for tid in tmdb_ids
        if tid and str(tid) != 'nan'
    ]

    if not valid_ids:
        return []

    # map() keeps results in the SAME ORDER as input
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(get_movie_details, valid_ids))

    return results