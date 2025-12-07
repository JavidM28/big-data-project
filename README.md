# Flight Delay Prediction

A machine learning pipeline for predicting airline delays using PySpark and historical flight data.

## Overview

This project implements a distributed data processing and machine learning pipeline to analyze and predict flight delays. Using Apache Spark and PySpark MLlib, we processed over 500,000 flight records to build classification models that predict whether a flight will be delayed based on airline, route, and temporal features.

The work was completed as part of the Big Data Analytics course (INFT-4836) at ADA University.

## Dataset

**Source:** [Airlines Dataset - Kaggle](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay)

The dataset contains 539,382 commercial flight records with the following attributes:
- Airline carrier
- Origin and destination airports  
- Scheduled departure time
- Day of week
- Flight duration
- Delay status (binary: delayed/on-time)

## Methodology

### Data Processing
The analysis pipeline includes:
- Missing value imputation using column means
- Feature extraction from temporal data (hour of day, weekday/weekend)
- Categorical encoding via StringIndexer
- Feature vector assembly for ML pipeline

### Models Implemented
We trained and evaluated four classification algorithms:
- Logistic Regression (L2 regularization)
- Random Forest (100 trees, max depth 10)
- Decision Tree (max depth 15)
- Gradient Boosted Trees (100 iterations, max depth 5)

All models used an 80/20 train-test split with stratified sampling to maintain class distribution.

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 60.72% | 0.6026 | 0.6072 | 0.5990 | 0.6387 |
| Random Forest | 62.89% | 0.6312 | 0.6289 | 0.6242 | 0.6721 |
| Decision Tree | 61.34% | 0.6145 | 0.6134 | 0.6098 | 0.6523 |
| **Gradient Boosted Trees** | **66.64%** | **0.6652** | **0.6664** | **0.6604** | **0.7185** |

The best performing model (Gradient Boosted Trees) achieved 66.64% accuracy with an AUC-ROC of 0.7185. Feature importance analysis revealed that airline carrier accounts for 51.4% of predictive power, followed by departure time (21.6%) and origin airport (12.7%).

## Performance Analysis

The 66.64% accuracy is consistent with published research in flight delay prediction. The relatively modest performance reflects the inherent difficulty of the problem - many delay-causing factors (weather, mechanical issues, crew scheduling, air traffic control) are not captured in the dataset. Our feature set is limited to scheduled flight information available at booking time.

For comparison, academic studies on similar datasets typically report accuracy in the 60-70% range. Higher reported accuracies often indicate data leakage (e.g., using actual departure times to predict delays) or overfitting.

## Technical Stack

- **Distributed Computing:** Apache Spark 3.5.0
- **Language:** Python 3.11
- **ML Framework:** PySpark MLlib
- **Data Processing:** pandas 2.1.0
- **Visualization:** matplotlib 3.8.0, seaborn 0.13.0

## Repository Structure

```
.
├── src/
│   ├── spark_analysis_clean.py    # EDA and data analysis
│   └── mlpart_clean.py             # ML pipeline and model training
├── requirements.txt                # Python dependencies
├── SETUP.md                        # Installation and setup instructions
└── README.md
```

## Installation

Requires Python 3.8+ and Java JDK 11+.

```bash
# Clone repository
git clone https://github.com/yourusername/flight-delay-prediction.git
cd flight-delay-prediction

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle (link above)
# Place Airlines.csv in project root
```

## Usage

```bash
# Run exploratory data analysis
python src/spark_analysis_clean.py

# Train and evaluate models
python src/mlpart_clean.py
```

The analysis script generates visualizations showing delay patterns by airline, time of day, day of week, and route. The ML script outputs model performance metrics, confusion matrices, and feature importance rankings.

Detailed setup instructions are available in `SETUP.md`.

## Implementation Notes

**Handling High-Cardinality Categoricals:** Airport features contain 294 unique values. Tree-based models required `maxBins=300` to accommodate this cardinality.

**Platform Compatibility:** The code includes fixes for Windows-specific issues (temp directory paths, UTF-8 encoding). Tested on Windows 11, should work on macOS/Linux with minimal modification.

**Resource Requirements:** Recommend 8GB RAM. Full dataset processing takes approximately 15-20 minutes on standard hardware.

## Limitations

- No weather data (significant delay factor)
- No real-time air traffic information
- No aircraft maintenance history
- No crew scheduling data
- Limited to historical patterns only

Future work could incorporate external data sources (NOAA weather, FAA traffic data) and explore deep learning approaches for sequence modeling of cascading delays.

## Academic Context

**Institution:** ADA University, Baku, Azerbaijan  
**Course:** INFT-4836 - Introduction to Big Data Analytics  
**Semester:** Fall 2025  

## References

1. Apache Spark MLlib Guide: https://spark.apache.org/docs/latest/ml-guide.html
2. Dataset: https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay
3. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
