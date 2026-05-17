# Ikman Electronic Items Fair Price Prediction

This is a machine learning project for predicting fair prices of used electronic items (laptops, monitors, and tablets) on Ikman.lk.

## Project Structure

- `data/raw/`: Raw scraped data from Ikman.
- `data/processed/`: Cleaned and preprocessed data ready for modeling.
- `notebooks/`: Jupyter notebooks for Exploratory Data Analysis (EDA) and initial model training.
- `src/scraper/`: Scripts to scrape data from Ikman.
- `src/preprocessing/`: Scripts for data cleaning and feature engineering.
- `src/modeling/`: Scripts to train, evaluate, and save machine learning models.
- `models/`: Saved model files (e.g., `.pkl` or `.joblib` files) for predicting prices.
- `requirements.txt`: List of Python dependencies for the project.

## Setup Instructions

1. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter Notebook for data exploration:
   ```bash
   jupyter notebook
   ```
