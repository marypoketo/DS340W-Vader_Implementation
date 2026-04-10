# model_engine.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold

def run_logistic_regression(book_data):
    """Performs predictive modeling and evaluates accuracy."""
    success = book_data[book_data['label'] == 1]
    other = book_data[book_data['label'] == 0]
    
    sample_size = min(len(success), len(other))
    df_balanced = pd.concat([
        success.sample(n=sample_size, random_state=42),
        other.sample(n=sample_size, random_state=42)
    ]).sample(frac=1).reset_index(drop=True)

    features = ['sentiment', 'review/score', 'review_count']
    X = df_balanced[features].values
    y = df_balanced['label'].values
    X_scaled = StandardScaler().fit_transform(X)

    model = LogisticRegression()
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    results = cross_val_score(model, X_scaled, y, cv=kfold)
    
    model.fit(X_scaled, y)
    return results.mean(), model.coef_[0], features, df_balanced