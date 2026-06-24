# Customer-Churn-Analysis

## Overview

This project develops an end-to-end customer churn prediction system that identifies customers at risk of leaving and supports retention decision-making through interactive analytics.

The solution combines:

- Logistic Regression based churn prediction
- ROC-driven threshold optimization
- Customer risk segmentation
- Streamlit prediction interface
- Power BI business dashboard

---

## Business Problem

Customer churn directly impacts revenue and customer lifetime value. Retaining existing customers is often significantly cheaper than acquiring new ones.

This project aims to:

- Predict which customers are likely to churn
- Quantify churn risk through probability scores
- Segment customers by risk level
- Support targeted retention strategies
- Visualize churn drivers and model performance

---

## Dataset

The dataset contains customer-level information including:

- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Score
- Estimated Salary
- Active Membership Status
- Exited (Target Variable)

Target:

- `Exited = 1` → Customer Churned
- `Exited = 0` → Customer Retained

---

## Methodology

### 1. Data Preparation

- Cleaned and validated customer records
- Encoded categorical variables
- Scaled numerical features using StandardScaler
- Performed train-test split for model evaluation

### 2. Churn Prediction Model

Implemented Logistic Regression with:

- Class imbalance handling using `class_weight='balanced'`
- Probability-based churn prediction
- Model evaluation using confusion matrix and classification metrics

### 3. Threshold Optimization

Instead of relying on the default 0.5 classification threshold:

- Generated ROC Curve
- Calculated Youden's J Statistic
- Selected an optimized threshold of **0.5255**

This improved churn detection performance and aligned predictions with business objectives.

### 4. Customer Risk Segmentation

Customers were segmented using churn probabilities:

| Risk Level | Churn Probability |
|------------|------------------|
| Low Risk | < 0.30 |
| Medium Risk | 0.30 – 0.60 |
| High Risk | > 0.60 |

This enables prioritization of retention campaigns.

---

## Model Performance

### Optimized Threshold (0.5255)

| Metric | Value |
|----------|----------|
| Accuracy | ~77% |
| Recall | ~71% |
| F1 Score | ~56% |

Key Outcome:

- Successfully identified over 70% of churning customers.
- Prioritized recall over raw accuracy to better support retention efforts.

---

## Streamlit Application

Interactive application allowing users to:

- Enter customer attributes
- Predict churn likelihood
- View churn probability score
- View customer risk category
- Experiment with classification thresholds

### Features

- Real-time predictions
- Probability-based output
- Risk segmentation
- User-friendly interface
<img width="1912" height="876" alt="streamlit1" src="https://github.com/user-attachments/assets/db50b2dc-6585-4419-b1a5-bf2102831195" />

---

## Power BI Dashboard

The dashboard provides business-facing insights through:

### KPI Monitoring

- Total Customers
- Actual Churners
- Churn Rate
- Predicted Churners
- Churners Captured (%)

### Customer Analytics

- Churn by Geography
- Churn by Gender
- Churn by Age Group
- Churn by Balance Group
- Churn by Credit Score Group
- Churn by Number of Products

### Risk Analytics

- Low / Medium / High Risk Segmentation
- Risk-based filtering
- Customer distribution across risk levels

### Threshold Analysis

Comparison of:
- Customers Targeted
- Churners Captured
- Recall
Across multiple decision thresholds to evaluate targeting trade-offs.
<img width="1322" height="742" alt="dashboard1" src="https://github.com/user-attachments/assets/746a03f1-411c-4b44-a55c-f4097c2f53a1" />
<img width="1315" height="737" alt="dashboard2" src="https://github.com/user-attachments/assets/33cb3243-29e8-4468-aef8-69496b0a80ac" />

---

## Tech Stack

### Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Streamlit

### Visualization
- Power BI

---

## Key Learnings

- Handling imbalanced classification problems
- Threshold optimization using ROC analysis
- Evaluating business trade-offs between recall and precision
- Building decision-support dashboards
- Translating ML outputs into actionable business insights

---

## Future Improvements

- Random Forest and XGBoost benchmarking
- Cost-sensitive churn optimization
- Customer Lifetime Value integration
- Automated retention recommendation engine
- Real-time deployment pipeline
