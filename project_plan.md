rer# Project Plan: Bitcoin Price Direction Prediction with XGBoost & HMM

## 1. Data Analysis & Preparation
**Goal:** Demonstrate the properties of the data and justify the classification approach.

*   **Load Data:** Use `load_and_merge_data` from `LGBM_breakout.py`.
*   **Stationarity Analysis:**
    *   Calculate Log-Returns of Close Price.
    *   Perform Augmented Dickey-Fuller (ADF) test on Close Price vs. Log-Returns.
    *   **Visualization:** Plot Close Price vs. Log-Returns on a small sample to visually show non-stationarity vs. stationarity.
*   **Regression Failure Demo:**
    *   Create a simple XGBoost Regressor to predict the *next bar's return*.
    *   Train on a small subset (e.g., 1 month).
    *   **Metric:** Calculate $R^2$ and MSE.
    *   **Visualization:** Plot Predicted vs. Actual returns. Show that predictions cluster around zero and fail to capture volatility, proving that simple regression is ineffective for price prediction.

## 2. Target Variable Generation (The "Oracle" Labeling)
**Goal:** Create a high-quality, lag-free target variable for training.

*   **Method:** Use a **Centered Moving Average** (or Gaussian Smoothing with `sigma`) on the *entire* dataset. This uses future data to create a perfect smooth trend line (Zero Lag).
*   **Class Definition:**
    *   Calculate the slope (derivative) of the smoothed line.
    *   Define 3 Classes:
        *   **UP (2):** Slope > Threshold
        *   **DOWN (0):** Slope < -Threshold
        *   **SIDEWAYS (1):** |Slope| <= Threshold
*   **Visualization:** Plot the original Close Price, the Smoothed Target Line, and the resulting Labels (colored background) on a zoomed-in chart to verify alignment.

## 3. Feature Engineering & Selection
**Goal:** Iteratively build a robust feature set.

*   **Base Features:**
    *   **Volume:** Volume Delta, Cumulative Volume Delta (CVD).
    *   **Momentum:** RSI, MACD.
    *   **Volatility:** Bollinger Bands (Width, %B).
*   **Iterative Process:**
    1.  Train XGBoost Classifier with one group of features.
    2.  Evaluate Feature Importance (Gain/Cover).
    3.  Keep high-impact features, discard noise.
    4.  Repeat with next group.
*   **Leakage Check:** Rigorously verify that no feature uses future data (unlike the target).

## 4. HMM Integration (Hidden Markov Models)
**Goal:** Capture market regimes (Volatility/Trend states) as features.

*   **Library:** `hmmlearn`.
*   **Setup:**
    *   Define 3 Hidden States (e.g., Low Volatility, Bull Trend, Bear Trend).
    *   **Input Features for HMM:** Log-Returns, Range (High-Low), Volume.
*   **Training:**
    *   Fit HMM **only on the Training Set**.
    *   Predict State Probabilities for both Train and Test sets.
*   **Feature Creation:** Add the predicted state probabilities (State_0, State_1, State_2) as features to the main dataset.
*   **Visualization:** Plot Price colored by HMM State to interpret what the states represent.

## 5. XGBoost Classifier Development
**Goal:** Build and optimize the final prediction model.

*   **Model:** XGBoost Classifier (Multi-class: 3 classes).
*   **Training Strategy:**
    *   **Split:** Train/Validation/Test (e.g., 70%/15%/15%).
    *   **Grid Search:** RandomizedSearchCV or custom loop (max 200 iterations) for hyperparameters (`max_depth`, `learning_rate`, `n_estimators`, `subsample`).
*   **Evaluation:**
    *   Confusion Matrix.
    *   Classification Report (Precision, Recall, F1-score).
    *   **Comparison:** Compare performance **With HMM Features** vs. **Without HMM Features**.

## 6. Visualization & Educational Reporting
**Goal:** Provide clear visual evidence and explanations.

*   **Plots:**
    *   Feature Importance Bar Chart.
    *   Confusion Matrix Heatmap.
    *   **Final Prediction Plot:** Price chart with background colored by *Predicted Class* vs. *Actual Class* dots.
*   **Commentary:**
    *   Explain why regression failed.
    *   Explain the concept of "Zero-Lag" labeling and why it's valid for *training targets* but not *features*.
    *   Interpret the HMM states (e.g., "State 0 captures high volatility crashes").

## 7. Execution Steps
1.  **Setup:** Create `project_notebook.ipynb` (or python scripts).
2.  **Data & Analysis:** Implement Step 1 & 2.
3.  **Baseline:** Implement Step 3.
4.  **HMM:** Implement Step 4.
5.  **Classifier:** Implement Step 5 (Iterative feature addition + Grid Search).
6.  **Final Report:** Compile results and plots.