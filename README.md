Bestseller Prediction via Reader Reception Analysis

Author: Maryam Syafiah Binti Mohamed Sharif & Nur Sakinah Binti Mohammad Zaini

Institution: Pennsylvania State University (PSU)

Course: DS340W - Data Science Writing and Research

Project Overview:
- This research project replicates and extends the methodology of the parent paper, "Identifying Informative Attributes for Identifying Best Sellers," which achieved a benchmark accuracy of 0.75. By utilizing VADER Sentiment Analysis and an expanded Bag-of-Words (BoW) model on modern reader reviews, this implementation achieved a 0.82 accuracy rate in classifying New York Times (NYT) Bestsellers versus non-successful titles.

Data Sources:
- Ground Truth Labels: Historical bestseller data (2010–2014) retrieved via the New York Times Books API.

- Reader Reception Data: A 3GB Amazon Review Dataset containing millions of customer ratings and text reviews for over 100,000 unique titles.

Methodology:

1. Data Preprocessing:
   - Normalized book titles by removing subtitles and leading articles to ensure accurate matching across datasets.

2. Memory-Safe Chunking: Processed the 3GB CSV file in chunks of 100,000 rows to avoid memory overflow.

3. Feature Engineering:

- Qualitative Sentiment: Calculated VADER compound scores to measure the emotional intensity of reader reception.

- Vocabulary Diversity: Extracted a 500-word Bag-of-Words (BoW) vocabulary, filtered for structural noise (e.g., HTML artifacts like "quot" and "br").

- Engagement Volume: Utilized review counts, transformed via log-scaling to reveal class discrepancies.

4. Classification: Trained a Logistic Regression model using 10-fold cross-validation on a balanced database (1:1 success-to-other ratio).

Key Results
- Final Accuracy: 0.82.

- Informative Attributes: Review volume and superlatives (e.g., "best", "great", "life") were identified as the strongest predictors of bestseller status.

- Visual Separation: Log-scale Kernel Density Estimation (KDE) plots confirm a distinct "engagement density" for bestsellers compared to other titles.

Project Structure
- data_engine.py: Handles NYT API calls, Amazon dataset chunking, and BoW extraction.

- model_engine.py: Performs data balancing, standardization, and Logistic Regression.

- visualizer.py: Generates research figures (KDE plots and bar charts).

- main.py: The master runner script integrating all stages.

- .env: Stores the NYT API key (excluded from version control).

Getting Started
1. Add your NYT API Key to a .env file: NYT_API_KEY=your_key_here.

2. Install requirements: pip install pandas scikit-learn vadersentiment seaborn kagglehub requests python-dotenv.

3. Run the pipeline: python main.py.
