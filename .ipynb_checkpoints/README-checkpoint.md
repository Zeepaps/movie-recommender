# 🎬 CineMatch — Movie Recommendation System

A Netflix-style movie recommendation platform built with Python, Machine Learning, and Streamlit.

## How It Works
CineMatch uses a **Hybrid Recommendation System** combining:
- **Collaborative Filtering** — finds users with similar taste and recommends what they loved
- **Content-Based Filtering** — matches movies by genre as a fallback
- **SVD (Singular Value Decomposition)** — discovers hidden taste patterns across 100,000+ ratings

## 📊 Dataset
- **MovieLens Small** — 100,836 ratings across 9,724 movies by 610 users
- **TMDB API** — movie posters, descriptions, and metadata

## Features
- 🔍 Search any movie and get instant personalized recommendations
- 🎭 Filter recommendations by genre
- 🖼️ Netflix-style UI with real movie posters
- 🔥 Popular movies homepage for new users
- ⚡ Parallel API fetching for fast load times

## Tech Stack
| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Scikit-learn | SVD & ML tools |
| Streamlit | Web UI |
| TMDB API | Movie metadata & posters |

## ⚙️ Setup & Installation

1. Clone the repository
```bash
git clone https://github.com/zeepaps/movie-recommender.git
cd movie-recommender
```

2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root folder
   API_KEY=your_tmdb_api_key_here

5. Download the MovieLens dataset
```bash
python -c "import urllib.request, zipfile; urllib.request.urlretrieve('https://files.grouplens.org/datasets/movielens/ml-latest-small.zip', 'data/ml-latest-small.zip'); zipfile.ZipFile('data/ml-latest-small.zip').extractall('data/')"
```

6. Run the app
```bash
streamlit run app/app.py
```

## 📁 Project Structure
movie-recommender/
├── app/
│   ├── app.py              ← Main Streamlit application
│   └── tmdb_helper.py      ← TMDB API helper
├── data/                   ← MovieLens dataset
├── models/                 ← Saved model files
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_building.ipynb
├── .env                    ← API key (not on GitHub)
├── .gitignore
├── requirements.txt
└── README.md

## 👨‍💻 Author - Ayomide Zaccheaus
Built as an end-to-end ML project exploring recommendation systems.