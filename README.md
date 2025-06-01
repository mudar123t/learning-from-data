# Learning From Data — Weather Forecasting

This project analyzes historical weather data to build a predictive model using machine learning techniques. It uses Python data science libraries and a modular structure to handle data ingestion, processing, and modeling.

## Project Context
- Final academic project for machine learning / data science course
- Developed by: Ragab & Mudar
- Dataset: Hourly weather data from Bursa

## Features
- Reads Excel weather datasets
- Cleans and merges raw data
- Trains machine learning models (e.g., regression, classification)
- Evaluates predictions
- Modular structure for easy testing and extension

## Tech Stack
- Python 3.10+
- pandas
- scikit-learn
- matplotlib / seaborn
- Jupyter Notebooks

## How to Run
1. Create virtual environment:
   bash
   python -m venv venv
   source venv/bin/activate  # or venv\\Scripts\\activate on Windows

2. Install dependencies:

bash
pip install -r requirements.txt

3. Run notebooks or main scripts from src/ folder.

## Project Structure

├── data/               # Raw Excel datasets
├── src/                # Core logic
│   ├── data/           # Data collectors and cleaning
│   ├── features/       # Feature engineering
│   └── model/          # Training and evaluation

