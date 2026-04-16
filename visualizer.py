# visualizer.py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
    plt.close()
    
    # 2. Volume Distribution: USE LOG SCALE [cite: 336]
    # This prevents the 'squashed' look and shows the clear separation
    plt.figure(figsize=(10, 6))
    df_balanced['log_review_count'] = np.log1p(df_balanced['review_count'])
    sns.kdeplot(data=df_balanced, x='log_review_count', hue='label', fill=True)
    plt.title('Kernel Density Estimation: Log-Scale Review Volume')
    plt.xlabel('Log(Review Count + 1)')
    plt.savefig('log_volume_distribution.png')
    plt.close()

def generate_advanced_plots(df_balanced, model, features, X_scaled, y):
    """Generates the 'Intelligent Analysis' suite for the final paper."""
    
    # 1. CONFUSION MATRIX (Model Reliability)
    plt.figure(figsize=(8, 6))
    y_pred = model.predict(X_scaled)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Other', 'Success'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix: Prediction Reliability')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 2. CORRELATION HEATMAP (Feature Interconnectivity)
    plt.figure(figsize=(12, 10))
    # Select top 15 features for clarity
    top_feats = features[:15] + ['label']
    corr = df_balanced[top_feats].corr()
    sns.heatmap(corr, annot=False, cmap='RdBu_r')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('feature_correlation.png')
    plt.close()

def plot_accuracy_comparison(results_df):
    """Generates a line graph showing the 'Push to 0.82'."""
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Vocab Size'], results_df['Accuracy'], marker='o', label='Accuracy', color='blue')
    plt.plot(results_df['Vocab Size'], results_df['Precision'], marker='s', label='Precision', color='green')
    
    plt.title('Impact of Vocabulary Diversity on Model Performance')
    plt.xlabel('Number of BoW Features (max_features)')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('comparative_performance_graph.png')
    plt.close()
    
    