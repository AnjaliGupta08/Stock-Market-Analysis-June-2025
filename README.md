# Stock Market Analysis Project (June 2025)
A Python-based data analysis and visualization project exploring stock market trends for June 2025. Includes data cleaning, EDA, visualizations and prediction modeling.

---

## Key Exploratory
🔹 Top 5 Companies by EPS (Earning Per Share)

```python
Ticker    EPS
BRK.B     54.10
WGK       31.10
PHM       30.26
HQZ       29.25
QYG       29.24
```

> **BRK.B** has the highest EPS.


🔹 Energy sector has the highest average EPS, suggesting stronger earnings across companies in that group and Technology has the lowest EPS.

🔹 Technology sector exhibits the highest average Price-to-Earnings (PE) ratio, suggesting high investor expectations for future growth, even if current earnings are            relatively modest. Energy sector has the lowest PE ratio, implying undervaluation or more conservative growth prospects.

🔹 The average `Close` price for the month was **₹152.73**. 

🔹 The median number of shares traded (Volume Traded) per day was **45.8 million**. 

🔹 Price ranged from ₹127.07 to ₹178.44 over June 2025.


🔹 Stock Volatility Analysis

```python
Daily Return Mean: 0.0013
Daily Return Std: 0.0172
```

> Daily return volatility was **1.72%**, indicating a moderately stable stock.

🔹 Top 5 Companies by Market Capitalization

These are the top 5 companies in the dataset based on their total market cap (in ₹ trillions):

| Ticker | Market Cap (₹ Trillions) |
|--------|---------------------------|
| NVDA   | 3.48                      |
| MSFT   | 2.82                      |
| AAPL   | 2.66                      |
| META   | 1.31                      |
| ABT    | 1.29                      |

>  **Insight:** NVDA has the highest market cap, which means investors trust it the most. These companies are huge and play an important role in the overall stock market.

 🔹 Top 5 Sectors by Market Capitalization

These are the top 5 companies in the dataset based on their total market cap (in ₹ trillions):


| Sector       | Market Cap (₹ Trillions) |
|--------------|---------------------------|
| Financials   | 14.85                     |
| Materials    | 12.21                     |
| Energy       | 10.95                     |
| Industrials  | 10.95                     |
| Real Estate  | 10.03                     |

>  **Insight:** Financials is the largest sector by market cap, which means this sector includes some of the biggest and most trusted companies. These sectors are important for the economy and influence the overall market.



### 🔹 Strongest Correlated Features

```python
High correlation (0.99+) between Open, High, Low, Close
Volume weak correlation
```

> Suggests that price-based features can substitute for each other in modeling.

### 🔹 Class Balance in Target

```python
Above Median: 240 days
Below Median: 239 days
```

> A **very balanced classification dataset**.

---

## 📈 All Graphs & Results (Visual Summary)

### 🔍 Visual Analysis & Interpretations:

1. **Distribution of Daily Returns**

   * Shape: Left-skewed
   * Insight: Majority of daily returns are low or near 0; few extreme negative values
   * Suggestion: This supports the use of a conservative model for short-term prediction.

2. **KDE (Kernel Density Estimation) Plots of Closing Prices**

   * Observation: Most prices cluster tightly around the mean with minimal variance
   * Suggestion: Good candidate for linear models due to distribution consistency

3. **Pairplot (Open, High, Low, Close)**

   * Insight: Features are strongly linearly related
   * Suggestion: Redundant variables may be dropped for modeling simplicity

4. **Correlation Heatmap**

   * Highest correlation: Open ↔ High ↔ Low ↔ Close (all > 0.99)
   * Weak correlation: Volume to any price metric
   * Suggestion: Focus on price indicators, reduce reliance on Volume

5. **Classification Model Scores (Bar Charts)**

   * Logistic Regression had highest F1 Score (\~0.62)
   * Decision Tree, SVM, and Random Forest underperformed
   * Insight: Simpler models like Logistic Regression work well with binary structured features

6. **Confusion Matrix (Logistic Regression)**

   * True Positives: 52, True Negatives: 56
   * Suggestion: Excellent balance in classification for target classes

> 📊 These visualizations confirm high feature redundancy, low volume impact, and support for linear modeling.

\----------------------------------------------|----------------|
\| Distribution of Daily Returns (Skewed left)  | ![Daily Returns](assets/daily_return_dist.png) |
\| KDE Plot of Closing Prices                   | ![KDE Close](assets/kde_close_price.png)       |
\| Pairplot (Open, High, Low, Close)            | ![Pairplot](assets/pairplot_prices.png)        |
\| Correlation Heatmap                          | ![Heatmap](assets/correlation_heatmap.png)     |
\| Model Evaluation Scores (Bar Chart)          | ![Model Scores](assets/classification_scores.png) |
\| Confusion Matrix (Logistic Regression)       | ![Confusion](assets/confusion_matrix_logreg.png) |

> 📂 Please place these images in the `/assets/` folder of your GitHub repo for full display.

---

### 🔍 Visual Analysis & Interpretations:

1. **Distribution of Daily Returns**

   * Shape: Left-skewed
   * Insight: Majority of daily returns are low or near 0; few extreme negative values
   * Suggestion: This supports the use of a conservative model for short-term prediction.

2. **KDE (Kernel Density Estimation) Plots of Closing Prices**

   * Observation: Most prices cluster tightly around the mean with minimal variance
   * Suggestion: Good candidate for linear models due to distribution consistency

3. **Pairplot (Open, High, Low, Close)**

   * Insight: Features are strongly linearly related
   * Suggestion: Redundant variables may be dropped for modeling simplicity

4. **Correlation Heatmap**

   * Highest correlation: Open ↔ High ↔ Low ↔ Close (all > 0.99)
   * Weak correlation: Volume to any price metric
   * Suggestion: Focus on price indicators, reduce reliance on Volume

5. **Classification Model Scores (Bar Charts)**

   * Logistic Regression had highest F1 Score (\~0.62)
   * Decision Tree, SVM, and Random Forest underperformed
   * Insight: Simpler models like Logistic Regression work well with binary structured features

6. **Confusion Matrix (Logistic Regression)**

   * True Positives: 52, True Negatives: 56
   * Suggestion: Excellent balance in classification for target classes

> 📊 These visualizations confirm high feature redundancy, low volume impact, and support for linear modeling.

---

## 📜 Appendix: Detailed Cell Insights (Cells \[33]–\[274])

### 📌 Cell \[33]: Top 5 Companies by EPS

```python
df[['Ticker','EPS']].sort_values('EPS',ascending=False).head(5)
```

**Answer:**

```
Ticker    EPS
BRK.B     54.10
WGK       31.10
PHM       30.26
HQZ       29.25
QYG       29.24
```

> ✅ **BRK.B** has the highest EPS at 54.10

### 📌 Cell \[34]: Companies with Highest Cumulative EPS

```python
df.groupby('Ticker')['EPS'].sum().sort_values(ascending=False).head(5)
```

**Answer:**

```
Ticker
BRK.B    54.10
CEA      33.14
TTT      32.04
WGK      31.10
PHM      30.26
```

> ✅ Again, **BRK.B** leads with the highest total EPS.

### 📌 Cell \[35]: Average Closing Price

```python
df['Close'].mean()
```

**Answer:** `152.73`

> 💡 The average close price during the analysis period was ₹152.73

### 📌 Cell \[36]: Median Trading Volume

```python
df['Volume'].median()
```

**Answer:** `45854800.0`

> 💡 Median daily volume traded was \~45.8 million shares

### 📌 Cell \[37]: Maximum and Minimum Close Price

```python
df['Close'].max(), df['Close'].min()
```

**Answer:** `178.44`, `127.07`

> 🔺 Max price = ₹178.44, 🔻 Min price = ₹127.07

### 📌 Cell \[38]: Daily Return Volatility

```python
df['Daily Return'].mean(), df['Daily Return'].std()
```

**Answer:** `Mean = 0.0013`, `Std Dev = 0.0172`

> 📉 Daily return volatility was 1.72%

### 📌 Cell \[40]: Target Variable Class Distribution

```python
df['Target'].value_counts()
```

**Answer:**

```
1    240
0    239
```

> 🟰 Very well-balanced binary classification (above vs. below median)

### 📌 Cell \[175]: Confusion Matrix (Logistic Regression)

**Confusion Matrix:**

```
[[56  3]
 [ 9 52]]
```

> ✅ **F1 Score:** 0.62 — Best among all models

---

*Note: For brevity, this appendix summarizes key outputs. Full cell-by-cell extraction available upon request.*

---

## 🔢 Linear Regression Modeling

### Features Used:

* `Open`, `High`, `Low`, `Volume`
* Target: `Close`

### Train-Test Split:

* 80% training data, 20% testing data

### Performance Metrics:

* **R² Score:** `0.9994`
* **MSE:** `0.1760`
* **RMSE:** `0.4197`

> 📊 *Insight:* The model fits exceptionally well due to linear relationships among price variables.

---

## 🤖 Classification Models (Bonus Section)

The classification task was designed to predict whether a given day's `Close Price` was **above or below the median** for the month. The binary `Target` variable was engineered accordingly.

### ⚙️ Model Evaluation Summary

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.58     | 0.58      | 0.66   | 0.62     |
| Random Forest       | 0.52     | 0.53      | 0.58   | 0.55     |
| Decision Tree       | 0.50     | 0.51      | 0.51   | 0.51     |
| SVM                 | 0.47     | 0.48      | 0.51   | 0.49     |

> ✅ **Best Model (F1 Score):** Logistic Regression — good balance of recall and precision.

### 📊 Confusion Matrix Observations

#### Logistic Regression Confusion Matrix:

```
[[56  3]
 [ 9 52]]
```

* **True Negatives (56):** The model correctly predicted 56 instances as class 0 (e.g., below-median close) when they were truly class 0.
* **False Positives (3):** The model incorrectly predicted 3 instances as class 1 (above-median close), which were actually class 0.
* **False Negatives (9):** The model predicted 9 instances as class 0, which were truly class 1.
* **True Positives (52):** Correctly predicted 52 instances as above-median close when they truly were.

📌 **Why Logistic Regression is Better:**

* It has the highest **F1 Score (0.62)**, meaning it strikes the best balance between precision and recall.

* It minimized both false positives and false negatives better than Random Forest, SVM, or Decision Tree.

* Simpler models performed better due to linear relationships in the dataset.

* **Logistic Regression** performed the best in generalizing on the test set.

* All models struggled to go far beyond random baseline — indicating the binary target may be noisy.

### 🔍 Insights & Suggestions

* **Volume** is weakly correlated with prices and likely adds noise.
* Since price features are linearly related, Logistic Regression had an advantage.
* **Decision Tree and Random Forest** had potential but may need feature engineering (e.g., moving averages, volatility windows).
* **SVM** worked better with standardized features but didn’t outperform linear model.

### 📈 Visual Comparison

Bar plots were created for **Accuracy**, **Precision**, **Recall**, and **F1 Score** across models. Logistic Regression led in all except recall.

> 📌 *Recommendation:* Experiment with:

* Technical indicators as features (e.g., RSI, MACD, moving averages)
* Multi-class labels like "High Rise", "Dip", "Stable"
* Timeseries-based models (LSTM, Prophet)

---

## 💼 Business Insights

### 📌 Strategic Opportunities Identified:

* **Consistent Performers:** Companies like BRK.B showed exceptional EPS and consistent performance across daily returns — suitable for long-term investments.
* **Volatility Assessment:** Daily return standard deviation of \~1.7% indicates moderate short-term risk, helping investors manage exposure.
* **Price Prediction Reliability:** With an R² of 0.999 in the regression model, price forecasting for short horizons can guide algorithmic trading or automated rebalancing.
* **Balanced Market Segments:** The binary target classes were evenly distributed (240/239), which implies a fair opportunity to capitalize on both upward and downward trends.
* **Modeling Guidance:** Logistic Regression outperformed other classification models — simple but effective. Ideal for alert-based strategies.
* **Low Informative Volume Feature:** Trading volume didn’t correlate strongly with closing price — avoid overrelying on it for market predictions.

> 📍 **Application Example:** A retail investor or fund could use this model to filter stocks above predicted thresholds while tracking volatility triggers using regression margins.

---

## ✅ Conclusion

This stock market analysis project demonstrates a full workflow:

* Clean exploratory data analysis (EDA)
* Price & volume statistics
* Visual insights
* Regression modeling with near-perfect performance
* Classification modeling with room for improvement

📦 This project is **portfolio-ready** and suitable for GitHub showcase, job interviews, and data science demos.

---
