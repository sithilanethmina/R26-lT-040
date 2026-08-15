# FairPriceLK

FairPriceLK is a machine learning based price prediction system for the Sri Lankan second-hand marketplace. It helps users decide whether a listing price is fair, overpriced, suspiciously low, or potentially a scam by comparing the visible listing price against an estimated current market value.

The intended user experience is simple: when a user visits a Sri Lankan second-hand marketplace with the browser extension enabled, the system reads the listing they are viewing and returns a fairness score based on the predicted market price for that item.

## Problem We Solve

Second-hand marketplace buyers often ask the same question before making contact with a seller:

`Is this price actually reasonable for the current Sri Lankan market?`

This project answers that question with data-driven price estimation for selected product categories. Instead of relying only on intuition, buyers can use a model-backed fairness score to judge whether the asking price is:

- fair
- too expensive
- unusually cheap
- worth inspecting more carefully

## Supported Categories

This repository currently contains work for the following item groups:

- GPUs
- Mobile phones
- Vehicles
- Electronics

For electronics, the focus is on:

- Tablets
- Laptops
- Monitors

## How The System Works

At a high level, the platform works like this:

1. Marketplace listing data is collected from Sri Lankan sources.
2. The raw listing data is cleaned and standardized.
3. Category-specific machine learning models are trained on historical listing data.
4. A predicted market value is generated for the item being viewed.
5. The predicted value is compared with the seller's asking price.
6. A fairness score is produced for the user through the extension or app layer.

The fairness score is the core user-facing feature. It is meant to help users quickly interpret whether the listing price aligns with the market.

## Main Use Case

Example scenario:

- A user opens a listing for a used phone, GPU, vehicle, laptop, tablet, or monitor on a Sri Lankan second-hand marketplace.
- The extension detects the listing price and relevant item details.
- The appropriate trained model estimates the likely fair market value.
- The system shows a fairness score and a pricing judgment, helping the user understand whether the listing looks reasonable.

## Repository Structure

```text
.
|-- electronics-price-model/
|-- gpu-price-model/
|-- mobile-price-model/
|-- vehicle-price-model/
`-- .github/
```

## Module Overview

### `gpu-price-model/`

Contains the GPU price prediction pipeline, including:

- Sri Lankan listing scrapers
- cleaned and enriched datasets
- model training scripts
- prediction app components
- saved model artifacts

### `mobile-price-model/`

Contains the mobile phone price prediction workflow, datasets, trained models, and Streamlit-based experimentation files.

-Sri Lankan marketplace data scraping modules
-data preprocessing and feature engineering pipelines
-fair price prediction model training scripts
-anomaly detection model implementation
-threshold optimisation experiments
-browser extension source code
-ONNX model export and browser inference components
-real-time listing analysis and verdict overlay system
-evaluation and usability testing modules
-trained model artifacts and datasets

### `vehicle-price-model/`

Contains the vehicle fair-price estimation system for Sri Lankan used vehicle marketplaces.

This module includes:

- automated vehicle listing data collection and preprocessing
- vehicle price prediction pipelines
- fairness score calculation
- NLP-assisted listing analysis
- trained machine learning models and evaluation output

### `electronics-price-model/`

Reserved for electronics price prediction work focused on:

- tablets
- laptops
- monitors

## How to Run

To run the full system locally (Gateway + Microservices + Browser Extension), follow these steps:

### 1. Start the Backend Services

The backend consists of an API Gateway and multiple microservices (GPU, Mobile, Vehicle, Electronics). A single script starts them all.

1. Open your terminal in the project root folder.
2. Create a virtual environment and activate it:
   - **Windows (Command Prompt):**
     ```cmd
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Mac/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
3. Ensure you have the required dependencies installed within the virtual environment:
   ```bash
   pip install fastapi uvicorn httpx flask flask-cors pandas numpy joblib streamlit "scikit-learn==1.8.0" xgboost
   ```
4. Run the startup script:
   ```bash
   cd api-gateway
   python start_all.py
   ```
   _This will start the API Gateway on port 8000, and the microservices on ports 8001-8004._

### 2. Install the Browser Extension

1. Open Google Chrome and go to `chrome://extensions/`.
2. Turn on **Developer mode** (toggle switch in the top right corner).
3. Click the **Load unpacked** button in the top left.
4. Select the `browser-extension` folder from this repository.
5. The **FairPriceLK Checker** extension should now appear in your browser.

### 3. Use the Extension

1. Navigate to a second-hand marketplace listing (e.g., an ikman.lk listing for a phone, GPU, car, or laptop).
2. Click the FairPriceLK extension icon in your browser toolbar.
   - _If the server is running, you will see a green "Server Connected" dot at the bottom._
3. The extension will automatically detect the category (Mobile, GPU, etc.) and pre-fill details from the page.
4. Click **Check Price** to get the predicted market value and the fairness verdict.

## Why This Is Useful

This system is useful for:

- buyers who want a quick fair-price check before contacting a seller
- users trying to avoid overpriced listings
- users trying to spot suspiciously cheap listings that may indicate scams or hidden issues
- building a browser extension that adds real-time pricing intelligence to Sri Lankan marketplace pages

## Machine Learning Focus

This is not a static price lookup table. The project is built around machine learning models that learn patterns from marketplace data, such as:

- item model and variant
- brand
- technical specifications
- year or generation
- listing patterns within Sri Lankan market data

Each product category can use a different training pipeline and feature set, depending on what matters most for that type of item.

## Expected Output

For a given listing, the full system is designed to provide:

- predicted market price
- fairness score
- interpretation of the score
- warning signal when a price appears unusually high or suspiciously low

## Project Vision

The broader goal of this project is to build a practical fairness-checking assistant for Sri Lankan second-hand marketplaces. By combining marketplace scraping, data cleaning, machine learning, and browser-extension integration, FairPriceLK aims to help users make safer and smarter buying decisions.
