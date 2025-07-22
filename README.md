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


🔹 Average Dividend Yield by Sector
>  **Insight:** Sectors like **Utilities**-__ 3.55 %__, **Real Estate**-__ 3.50 % __, and **Consumer Staples**-__ 3.46 % __ offer the highest dividend yields, making them attractive for income-focused investors. In contrast, **Technology**-__1.05 %__ has the lowest average yield.

🔹 **META** was the most volatile stock of the month with an average daily price swing of **23.15**.This stock experienced the largest intraday price fluctuations during the month.

🔹**Financials** was the most volatile stock of the month with an average daily price swing of ** 6.068036**.This stock experienced the largest intraday price fluctuations during the month.

🔹**05-06-2025** saw the highest average market movement, with a **3.83%** increase in daily price change across all stocks.

🔹Stocks with Price Increase Despite High Volume

| Date       | Ticker | Open Price | Close Price | Volume Traded |
|------------|--------|------------|-------------|----------------|
| 01-06-2025 | YPY    | 260.55     | 265.28      | 14,012,358     |
| 01-06-2025 | VKD    | 182.43     | 186.89      | 14,758,143     |
| 01-06-2025 | GPH    | 88.06      | 90.51       | 14,070,514     |
| 01-06-2025 | HXJ    | 156.26     | 160.38      | 14,091,615     |
| 01-06-2025 | TDW    | 270.11     | 271.45      | 8,083,607      |

>  **Insight:** Despite high trading volumes, these stocks managed to close higher than they opened, suggesting **strong buying interest or positive sentiment** in the market.

🔹Sectors with **smaller market caps** like Utilities and Real Estate tend to offer **higher dividend yields**, while **larger sectors** like Materials and Industrials typically provide **lower yields**.



---


🔹 Strongest Correlated Features

```python
High correlation (0.99+) between Open, High, Low, Close
Volume weak correlation
```

> Suggests that price-based features can substitute for each other in modeling.

---

🔹## All Graphs & Results 

 Distribution of Daily Returns

   * Shape: Left-skewed
   * Insight: Majority of daily returns are low or near 0; few extreme negative values
   * Suggestion: This supports the use of a conservative model for short-term prediction.
     
 Pairplot (Open, High, Low, Close)

   * Insight: Features are strongly linearly related
   * Suggestion: Redundant variables may be dropped for modeling simplicity


 Classification Model Scores (Bar Charts)

 Model Performance Comparison – Classification Metrics

| Model                | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.5836   | 0.5813    | 0.6556 | 0.6162   |
| Random Forest       | 0.5184   | 0.5250    | 0.5833 | 0.5526   |
| Decision Tree       | 0.5014   | 0.5112    | 0.5056 | 0.5084   |
| SVM                 | 0.4674   | 0.4789    | 0.5056 | 0.4919   |

---
>  Key Insights
- **Logistic Regression** delivers the **best overall performance** across all metrics, making it the most effective model in this case.
- **Random Forest** shows decent precision and recall but still underperforms compared to Logistic Regression.
- **Decision Tree** and **SVM** perform the worst, suggesting poor generalization or insufficient feature separation.
- All models show **moderate accuracy**, indicating potential for further improvement through better features or tuning.
  
---

##  Linear Regression Modeling

### Features Used:

* `Open`, `High`, `Low`, `Volume`
* Target: `Close`

### Train-Test Split:

* 80% training data, 20% testing data

### Performance Metrics:

- ***MAE of 0.93** — a sign of high accuracy.
- **MSE of 1.67** shows low error varianc.
- **R² Score of 0.9998** suggests the model explains **over 99.97% of the variance** in the target — an **excellent fit**.


### 🔍 Insights & Suggestions 

-Invest more in Financials, Materials, and Energy — these are the biggest and most trusted sectors in the market.

-Sectors like Utilities and Real Estate pay higher dividends. These are good for building reliable income over time.

-Stocks like META had big price swings and high trading volume. This means people are paying a lot of attention — useful for spotting trends or risks.

-On days like June 5, the market moved a lot. These could be good or risky times to make big announcements or financial moves.

-For classifying market trends (up or down), Logistic Regression works best based on the current data.

