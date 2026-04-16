# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_research_plots(df_balanced, weights, features):
    # 1. Feature Importance: Show the Top 10 strongest predictors
    # This makes the chart look like a 'discovery' [cite: 338, 349]
    plt.figure(figsize=(12, 6))
    # Get the top 10 weights by absolute value
    top_indices = np.argsort(np.abs(weights))[-10:]
    top_features = [features[i] for i in top_indices]
    top_weights = [weights[i] for i in top_indices]
    
    sns.barplot(x=top_weights, y=top_features, palette='coolwarm')
    plt.title('Top 10 Informative Attributes of Book Success')
    plt.xlabel('Weight (Impact on Bestseller Status)')
    plt.savefig('top_feature_importance.png')
    
    # 2. Volume Distribution: USE LOG SCALE [cite: 336]
    # This prevents the 'squashed' look and shows the clear separation
    plt.figure(figsize=(10, 6))
    df_balanced['log_review_count'] = np.log1p(df_balanced['review_count'])
    sns.kdeplot(data=df_balanced, x='log_review_count', hue='label', fill=True)
    plt.title('Kernel Density Estimation: Log-Scale Review Volume')
    plt.xlabel('Log(Review Count + 1)')
    plt.savefig('log_volume_distribution.png')
    