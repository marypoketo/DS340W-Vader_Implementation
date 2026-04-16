import os, time, kagglehub, requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

analyzer = SentimentIntensityAnalyzer()

def clean_title(title):
    """Normalizes titles by removing subtitles and leading articles."""
    if not isinstance(title, str): return ""
    t = title.split(':')[0].split(' - ')[0].upper().strip()
    for word in ["THE ", "A ", "AN "]:
        if t.startswith(word): t = t[len(word):]
    return "".join(c for c in t if c.isalnum() or c.isspace()).strip()

def build_success_pool(api_key):
    """Fetches Ground Truth bestseller labels from the NYT API."""
    nyt_titles = set()
    for year in range(2010, 2015):
        for month in ["01", "07"]:
            date = f"{year}-{month}-01"
            url = f"https://api.nytimes.com/svc/books/v3/lists/full-overview.json?published_date={date}&api-key={api_key}"
            try:
                res = requests.get(url).json()
                if 'results' in res:
                    for lst in res['results']['lists']:
                        for b in lst['books']:
                            nyt_titles.add(clean_title(b['title']))
                time.sleep(6) 
            except: pass
    return nyt_titles

def process_amazon_chunks(nyt_titles, target_success=400, max_bow = 50):
    """Memory-safe extraction and feature engineering."""
    path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")
    csv_path = os.path.join(path, "Books_rating.csv")
    
    success_rows, other_rows, found_titles = [], [], set()
    for chunk in pd.read_csv(csv_path, chunksize=100000):
        chunk['clean_name'] = chunk['Title'].apply(clean_title)
        matches = chunk[chunk['clean_name'].isin(nyt_titles)]
        if not matches.empty:
            success_rows.append(matches)
            found_titles.update(matches['clean_name'].unique())
        if len(other_rows) < (target_success * 5):
            others = chunk[~chunk['clean_name'].isin(nyt_titles)].sample(min(500, len(chunk)))
            other_rows.append(others)
        if len(found_titles) >= target_success: break

    df_filtered = pd.concat(success_rows + other_rows)
    
    # 1. RESTORE SENTIMENT CALCULATION (Fixes the KeyError)
    print("🧪 Calculating Sentiment Scores...")
    df_filtered['sentiment'] = df_filtered['review/text'].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )
    
    # 2. DEFINE NOISE FILTER (Ensures 'quot' is removed)
    junk_words = ['quot', 'don', 'br', 'book', 'read', 'just', 'really', 've', 'like', 'good', 'story']
    custom_stop_words = list(text.ENGLISH_STOP_WORDS) + junk_words

    # 3. GROUP REVIEWS (Reception Depth)
    # We group once to keep all stats and text aligned
    book_stats = df_filtered.groupby('clean_name').agg({
        'sentiment': 'mean', 
        'review/score': 'mean', 
        'Title': 'count',
        'review/text': lambda x: ' '.join(str(i) for i in x)
    }).rename(columns={'Title': 'review_count'}).reset_index()

    # 4. EXTRACT BoW FEATURES
    if max_bow > 0:
        print(f"🧪 Extracting {max_bow}-Word Vocabulary...")
        vectorizer = CountVectorizer(max_features=max_bow, stop_words=custom_stop_words)
        bow_matrix = vectorizer.fit_transform(book_stats['review/text'])
        bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        
        # Merge stats with vocabulary
        book_data = pd.concat([book_stats.drop(columns=['review/text']), bow_df], axis=1)
    else:
        # This branch creates your "VADER-Only" baseline
        print("🧪 Baseline Run: Skipping BoW (VADER Only)...")
        book_data = book_stats.drop(columns=['review/text'])

    # 5. LABELING (Applies to both branches)
    book_data['label'] = book_data['clean_name'].apply(lambda x: 1 if x in nyt_titles else 0)
    
    return book_data


def add_bow_features(df, max_features=100):
    """
    Extracts the most frequent words as new features 
    to hit the 0.75 accuracy benchmark[cite: 28].
    """
    # Initialize Vectorizer (removing English stopwords) 
    # Add custom "junk" words to the standard English stop_words list
    junk_words = ['quot', 'don', 'br', 'book', 'read', 'just', 'like', 'really', 'good', 've']
    custom_stop_words = list(text.ENGLISH_STOP_WORDS) + junk_words

    # Use these words in your Vectorizer
    vectorizer = CountVectorizer(max_features=50, stop_words=custom_stop_words)
    
    
    # Fit and transform the review text
    bow_matrix = vectorizer.fit_transform(df['review/text'].astype(str))
    
    # Create a DataFrame of the word counts
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Join with your original data
    return pd.concat([df.reset_index(drop=True), bow_df], axis=1)