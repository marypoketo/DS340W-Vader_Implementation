import os, time, kagglehub, requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def clean_title(title):
    if not isinstance(title, str): return ""
    t = title.split(':')[0].split(' - ')[0].upper().strip()
    for word in ["THE ", "A ", "AN "]:
        if t.startswith(word): t = t[len(word):]
    return "".join(c for c in t if c.isalnum() or c.isspace()).strip()

def build_success_pool(api_key, years=range(2010, 2015)):
    """Fetches Ground Truth bestseller labels from NYT."""
    nyt_titles = set()
    for year in years:
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

def process_amazon_chunks(nyt_titles, target_success=400):
    """Memory-safe processing of the 3GB Amazon dataset."""
    path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")
    csv_path = os.path.join(path, "Books_rating.csv")
    
    success_rows, other_rows = [], []
    for chunk in pd.read_csv(csv_path, chunksize=100000):
        chunk['clean_name'] = chunk['Title'].apply(clean_title)
        
        # VADER Analysis: Qualitative to Quantitative
        chunk['sentiment'] = chunk['review/text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
        
        matches = chunk[chunk['clean_name'].isin(nyt_titles)]
        success_rows.append(matches)
        
        if len(other_rows) < (target_success * 5):
            others = chunk[~chunk['clean_name'].isin(nyt_titles)].sample(min(500, len(chunk)))
            other_rows.append(others)
        
        if pd.concat(success_rows)['clean_name'].nunique() >= target_success: break

    df_filtered = pd.concat(success_rows + other_rows)
    # Aggregating features: Sentiment, Rating, and Volume
    book_data = df_filtered.groupby('clean_name').agg({
        'sentiment': 'mean', 'review/score': 'mean', 'Title': 'count'
    }).rename(columns={'Title': 'review_count'}).reset_index()
    
    book_data['label'] = book_data['clean_name'].apply(lambda x: 1 if x in nyt_titles else 0)
    return book_data