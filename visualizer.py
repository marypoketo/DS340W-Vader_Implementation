# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns

def generate_research_plots(df_balanced, weights, features):
    """Generates figures for the final research paper."""
    # Feature Weight Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=features, y=weights)
    plt.title('Feature Impact on Book Success')
    plt.savefig('research_feature_importance.png')
    
    # Volume Density Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_balanced, x='review_count', hue='label', fill=True)
    plt.title('Distribution of Review Volume')
    plt.savefig('research_volume_distribution.png')