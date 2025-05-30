# Obesity levels based on eating habits and physical condition

This repository presents a supervised learning analysis using the **Obesity Levels Based on Eating Habits and Physical Condition** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition). The primary research question is:

> **To what extent can individual eating patterns, physical activity, and family history predict obesity levels among adults in Latin America?**

---

## Dataset Overview

The dataset contains **2,111 records** with **16 features**, including:
- **Demographics** (e.g., gender, age)
- **Lifestyle habits** (e.g., frequency of vegetable consumption, physical activity, and food intake patterns)
- **Target variable:** `NObeyesdad`, a categorical label indicating **7 obesity levels**:
  - Insufficient Weight
  - Normal Weight
  - Overweight Level I / II
  - Obesity Type I / II / III

---


## Exploratory Data Analysis (EDA)

- Weight distributions vary significantly across obesity categories.
- Pearson correlations reveal weak multicollinearity, which supports stable multivariate modeling.
- Key statistical tests include:
  - **Cramér’s V** for categorical features (e.g., `Gender`, `family_history`)
  - **ANOVA** for numeric features (e.g., `Weight`, `FCVC`, `NCP`, `Age`)

**Key findings:**
- **Weight** is the strongest discriminative feature.
- Lifestyle features like vegetable intake (`FCVC`) and meal frequency (`NCP`) are informative.
- Class distribution is balanced (each ~13–16%), making the dataset suitable for classification without major resampling.


---

## Classification Models

Three classifiers were implemented using **PySpark pipelines**:

| Model                | Best Parameters                                   |
|---------------------|----------------------------------------------------|
| Logistic Regression  | `regParam=0.01`, `elasticNet=1.0`, `maxIter=50`   |
| Decision Tree        | `maxDepth=5`, `impurity=gini`                     |
| Random Forest        | `maxDepth=15`, `numTrees=100`, `impurity=gini`   |

### Performance Summary

| Model                | Accuracy | Macro F1 Score |
|---------------------|----------|----------------|
| Logistic Regression | 81.86%   | 81.31%         |
| Decision Tree       | 90.20%   | 90.14%         |
| Random Forest       | **95.10%** | **95.16%**     |

- **Random Forest** outperforms others in both accuracy and generalisation.
- Confusion matrices show minimal misclassifications, especially for RF.
- Most errors occur between adjacent BMI categories, indicating smooth transitions in predictions.

---

## Feature Importance

Top predictive features (from RF model):
- `Weight`
- `FCVC` (vegetable consumption)
- `NCP` (number of meals)
- `CH20` (water intake)

These align with expected physical and behavioural factors contributing to obesity.

---

## Conclusion

This analysis demonstrates that behavioural, demographic, and anthropometric data can effectively predict obesity levels. The **Random Forest model** shows strong performance, confirming that **individual lifestyle features are highly informative** for obesity classification and can support public health risk stratification.

---

## Project Structure

```
├── Data/ # Cleaned dataset
├── Scripts/ # PySpark pipelines and modeling scripts
├── Output_figure/ # Model artefacts and visualisations
├── Report/ # Final report (PDF/Markdown)
└── README.md # Project overview
```

---
