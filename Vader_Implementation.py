import os
import time
import kagglehub
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold

# ---------------------------------------------------------
# 1. SETUP & UTILITIES
# ---------------------------------------------------------
api_key = "R3bsyQSAMqdbkTHIUAzW3PgLFBdbqHRiOGGzdWPaOzyJHout"
analyzer = SentimentIntensityAnalyzer()

def clean_title(title):
    if not isinstance(title, str): return ""
    t = title.split(':')[0].split(' - ')[0].upper().strip()
    for word in ["THE ", "A ", "AN "]:
        if t.startswith(word): t = t[len(word):]
    return "".join(c for c in t if c.isalnum() or c.isspace()).strip()

# ---------------------------------------------------------
# 2. SUCCESS LIST (Historical 2010-2014)
# ---------------------------------------------------------
years = range(2010, 2015)
months = ["01", "07"] 
nyt_titles = set()

print(f"📡 Building Success Pool (2010-2014)...")
for year in years:
    for month in months:
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
print(f"✅ Success Pool: {len(nyt_titles)} titles.")

# ---------------------------------------------------------
# 3. MEMORY-SAFE CHUNKING (3GB Dataset)
# ---------------------------------------------------------
print("📥 Downloading 3GB Amazon dataset...")
path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")
csv_path = os.path.join(path, "Books_rating.csv")

success_rows = []
other_rows = []
target_success_count = 400 # Aiming for a larger sample size

print("📂 Processing 3GB file in chunks...")
for chunk in pd.read_csv(csv_path, chunksize=100000): # Larger chunks for efficiency
    chunk['clean_name'] = chunk['Title'].apply(clean_title)
    
    # Identify matches
    matches = chunk[chunk['clean_name'].isin(nyt_titles)]
    success_rows.append(matches)
    
    # Collect "Other" samples
    if len(other_rows) < (target_success_count * 5):
        others = chunk[~chunk['clean_name'].isin(nyt_titles)].sample(min(500, len(chunk)))
        other_rows.append(others)
    
    found = pd.concat(success_rows)['clean_name'].nunique() if success_rows else 0
    print(f"   ...Found {found} unique Bestsellers", end='\r')
    
    if found >= target_success_count:
        break

df_filtered = pd.concat(success_rows + other_rows)

# ---------------------------------------------------------
# 4. FEATURE ENGINEERING: ADDING VOLUME
# ---------------------------------------------------------
print("\n🧪 Calculating Sentiment & Review Volume...")
df_filtered['sentiment'] = df_filtered['review/text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# Aggregating: Sentiment, Rating, and the all-important Review Count
book_data = df_filtered.groupby('clean_name').agg({
    'sentiment': 'mean',
    'review/score': 'mean',
    'Title': 'count' # Using count of reviews as a Volume Signal
}).rename(columns={'Title': 'review_count'}).reset_index()

book_data['label'] = book_data['clean_name'].apply(lambda x: 1 if x in nyt_titles else 0)

# Balancing the classes (1:1 Ratio)
success_grp = book_data[book_data['label'] == 1]
other_grp = book_data[book_data['label'] == 0]

sample_size = min(len(success_grp), len(other_grp))
df_balanced = pd.concat([
    success_grp.sample(n=sample_size, random_state=42),
    other_grp.sample(n=sample_size, random_state=42)
]).sample(frac=1).reset_index(drop=True)

print(f"✅ Balanced Dataset: {len(df_balanced)} books ({sample_size} per class)")

# ---------------------------------------------------------
# 5. MODELING & CROSS-VALIDATION
# ---------------------------------------------------------
if len(df_balanced) >= 50:
    # Features: Sentiment Intensity, Average Rating, and Review Volume
    X = df_balanced[['sentiment', 'review/score', 'review_count']].values
    y = df_balanced['label'].values
    
    # Standardizing is crucial when features like 'review_count' have large ranges
    X_scaled = StandardScaler().fit_transform(X)

    model = LogisticRegression()
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    results = cross_val_score(model, X_scaled, y, cv=kfold)

    print(f"\n📊 FINAL RESULTS")
    print(f"Average Accuracy: {results.mean():.2f}")
    print(f"Target Benchmark: 0.75")
    
    # Optional: Print feature coefficients to see which matters most
    model.fit(X_scaled, y)
    print(f"Feature Weights (S, R, V): {model.coef_[0]}")
else:
    print("❌ Match count still too low.")