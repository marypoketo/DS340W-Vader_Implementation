import requests
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold


# SETUP
api_key = "R3bsyQSAMqdbkTHIUAzW3PgLFBdbqHRiOGGzdWPaOzyJHout"
analyzer = SentimentIntensityAnalyzer()

def get_vader_score(text):
    if not text:return 0
    return analyzer.polarity_scores(text)['compound']

# Data Collection: Success Group
nyt_url = url = f"https://api.nytimes.com/svc/books/v3/lists/overview.json?api-key={api_key}"
data = requests.get(nyt_url).json()

success_data =[]
for lst in data['results']['lists']:
    for book in lst['books']:
        success_data.append({
            'title': book['title'],
            'description': book['description'],
            'label': 1 # Class 1: Success
        })

# Data Collection: Other Group
success_count = len(success_data)
google_url = f"https://www.googleapis.com/books/v1/volumes?q=subject:fiction&maxResults={min(success_count, 40)}"
google_res = requests.get(google_url).json()

def get_more_google_books(target_count):
    other_books = []
    # Loop until we have enough books or run out of results
    for i in range(0, target_count, 40):
        print(f"Fetching Google Books starting at index {i}...")
        url = f"https://www.googleapis.com/books/v1/volumes?q=subject:fiction&startIndex={i}&maxResults=40"
        res = requests.get(url).json()
        
        if 'items' not in res:
            break
            
        for item in res['items']:
            info = item.get('volumeInfo', {})
            desc = info.get('description', '')
            if desc: # Only keep books with descriptions for VADER
                other_books.append({
                    'title': info.get('title', '').upper(),
                    'description': desc,
                    'label': 0  # 0 = Other [cite: 113]
                })
        
        time.sleep(1) # Small pause to be nice to the API
        
    return pd.DataFrame(other_books)

other_df = get_more_google_books(230)
success_df = pd.DataFrame(success_data)

raw_df = pd.concat([success_df, other_df])

# Balance to a perfect 1:1 ratio based on your 'Other' count (97)
count_other = len(other_df)
balanced_success = raw_df[raw_df['label'] == 1].sample(n=count_other, random_state=42)
balanced_other = raw_df[raw_df['label'] == 0]

# Create your final balanced dataframe
df = pd.concat([balanced_success, balanced_other]).sample(frac=1).reset_index(drop=True)
print(f"New Balanced Dataset Size: {len(df)} books")
df['sentiment'] = df['description'].apply(get_vader_score)

X = df[['sentiment']].values
y = df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
results = cross_val_score(model, X_scaled, y, cv=kfold)


print(f"Dataset Size: {len(df)} books ({len(balanced_success)} Success, {len(balanced_other)} Other)")
print(f"Average Accuracy (10-fold): {results.mean():.2f}")

# Visualisation
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='sentiment', hue='label', fill=True, common_norm=False)
plt.title("Sentiment Density: Success vs. Other")
plt.xlabel("VADER Compound Score")
plt.ylabel("Density")
plt.show()