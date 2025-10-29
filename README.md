# 🎓 Harvard University Graduate Tuition Prediction

Machine learning models that predict Harvard graduate school tuition fees with **99.1% average accuracy** using historical data (1985-2017) and economic indicators.

## 📊 Project Overview

Predicts tuition fees for 9 Harvard graduate schools using polynomial regression with regularization. Incorporates academic year, US inflation rate, and Harvard endowment value as features.

### Results Summary

| School | Accuracy | Model | MAE | MAPE |
|--------|----------|-------|-----|------|
| Divinity | 99.86% | Lasso (d3) | $140 | 0.43% |
| Education | 99.92% | Linear (d2) | $218 | 0.62% |
| Design | 99.90% | Linear (d2) | $248 | 0.65% |
| Business (MBA) | 99.90% | Linear (d2) | $324 | 0.71% |
| GSAS | 99.81% | Linear (d2) | $281 | 0.85% |
| Government | 99.29% | Lasso (d3) | $535 | 0.88% |
| Law | 99.84% | Linear (d2) | $371 | 0.90% |
| Medical/Dental | 99.70% | Lasso (d3) | $467 | 0.97% |
| Public Health | 96.03% | Ridge (d3) | $1,387 | 2.99% |

**Historical Validation**: 2017 MBA prediction = $63,876 vs actual $63,675 (only $201 error = 99.68% accurate)

## 🚀 Quick Start

### Installation

```bash
conda create -n Fees_Prediction_3.12 python=3.12
conda activate Fees_Prediction_3.12
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Train Models

```bash
python model_development.py --train
```

### Make Predictions

```bash
python model_development.py --predict --school "Business (MBA)" --year 2025
python model_development.py --predict --school "Law" --year 2025 --inflation 3.5 --endowment 50.0
python model_development.py --predict --school "Medical/Dental" --year 2030
```

## 📁 Project Structure

```
Fees_Prediction/
├── models/
├── dataset.csv
├── Harvard Fees Dataset.csv
├── model_development.py
├── model_report.json
├── model_selection.py
├── preprocessing.py
├── README.md
├── requirements.txt
└── visualisation.py
```

## 🔧 Technical Details

### Features

1. **Academic Year** (1985-2017): Temporal trend
2. **US Inflation Rate** (%): Federal Reserve CPI data  
3. **Harvard Endowment** ($Billions): Institutional financial strength

### Models Tested

- Linear Regression (polynomial degrees 1, 2, 3)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)  
- Random Forest Regressor

### Validation Strategy

- **Split**: 80/20 train/test with random shuffle
- **Cross-Validation**: 5-fold CV
- **Metrics**: MAE, RMSE, R², MAPE, Accuracy within 10%

## 📈 Key Findings

1. **Endowment Impact**: Higher endowment → higher tuition (prestige pricing)
2. **Model Selection**: Degree 2 sufficient for most schools
3. **Public Health**: 96% accuracy shows model admits uncertainty

## 🎯 Example Predictions

**2025 Forecasts**:

| School | 2025 Prediction | Growth from 2017 |
|--------|----------------|------------------|
| Business (MBA) | $88,360 | +39% |
| Law | $82,732 | +45% |
| Medical/Dental | $73,615 | +23% |

## 🛠️ Dependencies

```
seaborn~=0.13
matplotlib~=3.10.7
pandas~=2.3.3
scikit-learn~=1.7.2
numpy~=2.3.3
joblib~=1.5.2
```

## 🎓 Academic Context

**Course**: Principles of AI & ML
**Institution**: Adamas University  
**Achievement**: 99.1% average accuracy

## 👤 Developer

**Abhineet Bhattacharjee**  
📧 abhineetbhattacharjee@gmail.com

**Last Updated**: October 30, 2025

***
