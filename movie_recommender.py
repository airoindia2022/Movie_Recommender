"""
Movie Recommendation System (Content-Based Filtering)
------------------------------------------------------
Recommends movies similar to the one you like, based on
genres, keywords and description using TF-IDF + Cosine Similarity.

Run:  python movie_recommender.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_movies(path="movies.csv"):
    df = pd.read_csv(path)
    # Combine text columns into a single "profile" for each movie.
    # Genres are repeated 3x to give them more weight in matching.
    df["profile"] = (
        (df["genres"] + " ") * 3 + df["keywords"] + " " + df["description"]
    )
    return df


def build_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["profile"])
    return cosine_similarity(matrix)


def recommend(title, df, similarity, top_n=5):
    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        return None
    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1: top_n + 1]
    return [(df.iloc[i]["title"], round(score * 100, 1)) for i, score in scores]


def main():
    print("=" * 50)
    print("   MOVIE RECOMMENDATION SYSTEM - AI Project")
    print("=" * 50)

    df = load_movies()
    similarity = build_similarity_matrix(df)

    print(f"\n{len(df)} movies loaded. Available titles:\n")
    for i, t in enumerate(df["title"], 1):
        print(f"  {t}", end="   ")
        if i % 3 == 0:
            print()
    print("\n\nType a movie name to get recommendations. Type 'quit' to exit.\n")

    while True:
        title = input("Enter a movie you like: ").strip()
        if title.lower() == "quit":
            print("Goodbye!")
            break
        results = recommend(title, df, similarity)
        if results is None:
            print("  Movie not found. Please type the exact title from the list.\n")
            continue
        print(f"\n  Because you liked '{title}', you may also enjoy:")
        for movie, score in results:
            print(f"   - {movie}  (match: {score}%)")
        print()


if __name__ == "__main__":
    main()
