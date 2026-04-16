# main.py
import os
import pandas as pd  # Added missing import
from dotenv import load_dotenv
from data_engine import build_success_pool, process_amazon_chunks
from model_engine import run_logistic_regression
from visualizer import generate_research_plots, generate_advanced_plots, plot_accuracy_comparison

def main():
    load_dotenv()
    api_key = os.getenv("api_key")
    
    print("📡 Stage 1: Building NYT Success Pool...")
    nyt_titles = build_success_pool(api_key)
    
    # Define sizes for your "Comparison Table"
    feature_sizes = [0, 50, 100, 300, 500] 
    comparison_results = []
    
    # We will save the best-performing data for the final plots
    best_df = None
    best_weights = None
    best_feats = None

    print("📂 Stage 2 & 3: Comparative Analysis Loop...")
    for size in feature_sizes:
        print(f"🧪 Testing resolution: {size} BoW features...")
        # Ensure data_engine.py is updated to accept 'max_bow'
        book_data = process_amazon_chunks(nyt_titles, max_bow=size)
        
        acc, prec, rec, weights, feats, df_balanced, model, X_scaled, y = run_logistic_regression(book_data)
    
        comparison_results.append({
            'Vocab Size': size,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec
        })
        
        # Save the 500-feature version for the final Stage 4 plots
        if size == 500:
            best_df, best_weights, best_feats = df_balanced, weights, feats
            final_model, final_X, final_y = model, X_scaled, y

    # Convert to the table for your research paper!
    results_df = pd.DataFrame(comparison_results)
    print("\n📊 COMPARATIVE PERFORMANCE TABLE")
    print(results_df)
    
    print("\n🎨 Stage 4: Generating Research Artifacts...")
    # 1. Standard Plots (Log-scale & Bar chart)
    generate_research_plots(best_df, best_weights, best_feats)
    # 2. Advanced Plots (Confusion Matrix & Heatmap)
    generate_advanced_plots(best_df, final_model, best_feats, final_X, final_y)
    # 3. The Comparison Graph (Accuracy vs Vocabulary)
    plot_accuracy_comparison(results_df)

if __name__ == "__main__":
    main()