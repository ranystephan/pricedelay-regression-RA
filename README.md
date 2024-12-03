This repository analyzes the relationship between price delay and stock/cryptocurrency market returns across multiple regions (cryptocurrency, U.S., and international markets). It employs regression models to quantify the impact of price delay—a measure of market inefficiency—on asset returns.

---

## **What is Price Delay?**

Price delay measures the extent to which a stock's price incorporates new information. It reflects **market frictions** such as liquidity constraints, information asymmetry, or limited investor recognition. Price delay is crucial in understanding asset pricing anomalies and improving trading strategies.

### **Why Do We Need It?**
1. **Market Efficiency**: Delayed price responses can highlight inefficiencies in information dissemination.
2. **Predictive Power**: Assets with higher delays often offer a return premium due to risks or frictions.
3. **Portfolio Optimization**: Identifying delayed assets can enhance investment strategies.

### **Metrics Used**
Three main delay measures are derived using regression on lagged returns:
1. **\(D_1\)**: Fraction of return variance explained by lagged market returns.
   $$D_1 = 1 - \frac{R^2_{\text{restricted}}}{R^2_{\text{unrestricted}}}$$
   Where \(R^2_{\text{restricted}}\) excludes lagged returns and \(R^2_{\text{unrestricted}}\) includes them.
2. **\(D_2\)**: Weighted lag coefficients:
   \[
   D_2 = \frac{\sum_{n=1}^4 n \beta_{-n}}{\beta_0 + \sum_{n=1}^4 \beta_{-n}}
   \]
3. **\(D_3\)**: Weighted lag coefficients normalized by standard error:
   \[
   D_3 = \frac{\sum_{n=1}^4 n \frac{\beta_{-n}}{\text{SE}(\beta_{-n})}}{\frac{\beta_0}{\text{SE}(\beta_0)} + \sum_{n=1}^4 \frac{\beta_{-n}}{\text{SE}(\beta_{-n})}}
   \]

These metrics quantify how much and how quickly prices respond to market-wide news.

---

## **Repository Overview**

### **Structure**
- **crypto-pricedelay-regression**: Analysis of price delay in cryptocurrency markets.
- **international-exceptUS-pricedelay-regression**: Focus on international equity markets (excluding U.S.).
- **usa-pricedelay-regression**: Price delay analysis for U.S. markets.

---

## **Code Explanation**

### **1. Data Preprocessing**
Each analysis begins with data cleaning and preparation:
- **Remove Missing Values**: Handles missing return observations.
- **Standardization**: Normalizes variables for regression consistency.
- **Feature Engineering**: Creates lagged return features for regression.

Example snippet from preprocessing (pseudocode):
```python
# Create lagged market returns
for lag in range(1, 5):
    data[f'market_return_lag{lag}'] = data['market_return'].shift(lag)
```

---

### **2. Regression Analysis**

#### **Model**
The regression model captures the relationship between individual stock returns and lagged market returns:
\[
r_{i,t} = \alpha + \beta_0 R_{m,t} + \sum_{n=1}^4 \beta_{-n} R_{m,t-n} + \epsilon
\]
Where:
- \(r_{i,t}\): Return of stock \(i\) at time \(t\).
- \(R_{m,t-n}\): Lagged market return.
- \(\beta_{-n}\): Coefficients on lagged returns indicate delay.

The regression outcomes help compute delay metrics:
- High \(\beta_{-n}\) values indicate significant lag, contributing to higher price delay.

#### **Implementation**
- **Crypto Analysis** (`crypto_analysis.ipynb`): Implements the regression using pandas and statsmodels.
- **U.S. Analysis** (`usa_analysis.ipynb`): Similar structure but tailored to U.S. data.

Example code:
```python
import statsmodels.api as sm

X = data[['market_return', 'market_return_lag1', 'market_return_lag2', 'market_return_lag3', 'market_return_lag4']]
y = data['stock_return']
X = sm.add_constant(X)  # Add intercept

model = sm.OLS(y, X).fit()
print(model.summary())
```

---

### **3. Portfolio Formation**

To improve regression precision and reduce noise, stocks are grouped into portfolios based on size and delay. Portfolio-level delay is computed as the average of individual delays.

---

### **4. Results and Insights**

#### **Cryptocurrency**
- High transaction delays in cryptocurrencies lead to slower information incorporation, evidenced by significant \(\beta_{-n}\) coefficients.

#### **U.S. Markets**
- Delayed firms (small, illiquid) show a 1-2% monthly return premium.

#### **International Markets**
- Delayed stocks, especially in developing markets, exhibit strong post-announcement drift.

---

## **Math Behind Price Delay**

The regression decomposes return variance:
- **Immediate response**: Captured by \(\beta_0\).
- **Delayed response**: Distributed across \(\beta_{-1}\) to \(\beta_{-4}\).

The metrics \(D_1\), \(D_2\), and \(D_3\) summarize:
- Variance fraction explained (\(D_1\)).
- Weighted delay impact (\(D_2\)).
- Normalized precision-weighted delay (\(D_3\)).

---

## **Why It Matters**

Price delay uncovers inefficiencies in markets and offers actionable insights for:
1. **Hedge Funds**: Identify mispriced assets.
2. **Regulators**: Assess market health.
3. **Academia**: Explore links between delay and asset characteristics.

---

## **Getting Started**

### **Dependencies**
- pandas
- numpy
- matplotlib
- statsmodels

Install via:
```bash
pip install pandas numpy matplotlib statsmodels
```

---

## **Contributing**
Fork and submit a pull request for improvements.

---

## **License**
MIT License.

---

This repository provides a robust framework to study price delay, its causes, and its impact, empowering traders, researchers, and policymakers to navigate market inefficiencies.
