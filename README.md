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

## Getting Started

Because this repository is organized by category, each model folder can be worked on independently.

Then install the dependencies for that module and run its training or app scripts as described in the local documentation.

## Project Vision

The broader goal of this project is to build a practical fairness-checking assistant for Sri Lankan second-hand marketplaces. By combining marketplace scraping, data cleaning, machine learning, and browser-extension integration, FairPriceLK aims to help users make safer and smarter buying decisions.
