# main.py
import os
from dotenv import load_dotenv
from data_engine import build_success_pool, process_amazon_chunks
from model_engine import run_logistic_regression
from visualizer import generate_research_plots

def main():
    load_dotenv()
    api_key = os.getenv("api_key")
    
    print("📡 Stage 1: Building NYT Success Pool...")
    nyt_titles = build_success_pool(api_key)
    
    print("📂 Stage 2: Processing 3GB Amazon Dataset...")
    book_data = process_amazon_chunks(nyt_titles)
    
    print("📊 Stage 3: Executing Logistic Regression (10-Fold CV)...")
    acc, weights, feats, df_balanced = run_logistic_regression(book_data)
    
    print(f"✅ Quantitative Accuracy: {acc:.2f}")
    
    print("🎨 Stage 4: Generating Visualization Artifacts...")
    generate_research_plots(df_balanced, weights, feats)

if __name__ == "__main__":
    main()