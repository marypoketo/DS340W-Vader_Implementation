# main.py
import os
from dotenv import load_dotenv
from data_engine import build_success_pool, process_amazon_chunks
from model_engine import run_logistic_regression
from visualizer import generate_research_plots

def main():
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("NYT_API_KEY")
    
    print("📡 Step 1: Building Success Pool...")
    nyt_titles = build_success_pool(api_key)
    
    print("📂 Step 2: Processing 3GB Amazon reviews...")
    book_data = process_amazon_chunks(nyt_titles)
    
    print("📊 Step 3: Training Model & Running 10-Fold CV...")
    acc, weights, feats, df_balanced = run_logistic_regression(book_data)
    
    print(f"✅ Final Research Accuracy: {acc:.2f}")
    
    print("🎨 Step 4: Generating Visualization Artifacts...")
    generate_research_plots(df_balanced, weights, feats)

if __name__ == "__main__":
    main()