# PROJECT-1 Mental Health Dataset – Exploratory Data Analysis

## Project Overview

This project focuses on performing **Exploratory Data Analysis (EDA)** on a mental health dataset to understand patterns and relationships among different factors related to mental health and well-being.

The dataset contains **292,364 records and 17 attributes**, including demographic information, occupation, family history, treatment, stress, lifestyle changes, mood swings, coping struggles, work interest, social weakness, and awareness of mental health care options.

The analysis was performed using **Python and Pandas** in Google Colab.

## Objectives

* Understand the structure and characteristics of the dataset.
* Explore the available demographic and mental health-related variables.
* Identify missing values and understand data quality.
* Perform basic data cleaning and preprocessing.
* Examine the distribution of categorical variables.
* Prepare the dataset for further analysis and visualization.

## Dataset Information

* Total Records: 292,364
* Total Features: 17
* Data Type: Primarily categorical/object data
* Missing Values: Present in the `self_employed` column
* Unique Countries: 35
* Gender Categories: 2
* Occupation Categories: 5

The dataset includes variables such as:

* Timestamp
* Gender
* Country
* Occupation
* Self Employment
* Family History
* Treatment
* Days Indoors
* Growing Stress
* Changes in Habits
* Mental Health History
* Mood Swings
* Coping Struggles
* Work Interest
* Social Weakness
* Mental Health Interview
* Care Options

The dataset structure and column information were examined using Pandas functions such as `shape`, `describe()`, `columns`, and `info()`.

##  Technologies Used

* Python *
* Pandas *
* Google Colab*
* Jupyter Notebook*

## EDA Process

### 1. Data Loading

The dataset was imported using Pandas:

```python
import pandas as pd

cs = pd.read_csv("Mental Health Dataset - Mental Health Dataset.csv")
```

### 2. Dataset Exploration

The dataset was explored using:

```python
cs.head()
cs.tail()
cs.shape
cs.columns
cs.info()
cs.describe()
```

These steps helped understand the dataset's dimensions, columns, data types, and categorical distributions.

### 3. Missing Value Analysis

Missing values were checked using:

```python
cs.isnull().sum()
```

The analysis identified **5,202 missing values in the `self_employed` column**, while the other columns contained no missing values.

### 4. Data Cleaning

Rows containing missing values were examined using:

```python
dropped = cs.dropna()
```

After removing missing records, the resulting dataset contained **287,162 complete records**.

An alternative approach was also explored by filling missing values in `self_employed`:

```python
cs_filled = cs.copy()
cs_filled['self_employed'] = cs_filled['self_employed'].fillna("No")
```

## Key Dataset Insights

The initial analysis shows that the dataset contains a large number of responses collected across multiple countries and occupations.

Some important variables investigated include:

* Family history of mental health conditions
* Whether treatment was received
* Time spent indoors
* Growing stress
* Changes in habits
* Mental health history
* Mood swings
* Coping struggles
* Work interest
* Social weakness
* Willingness to discuss mental health
* Awareness of available care options

The dataset contains **35 countries and 5 occupation categories**, providing a broad basis for categorical analysis.

## Project Structure

```text
Mental-Health-EDA/
│
├── Eda_assignment.ipynb
├── Mental Health Dataset - Mental Health Dataset.csv
└── README.md
```

## How to Run

1. Clone this repository.
2. Open `Eda_assignment.ipynb` using Jupyter Notebook or Google Colab.
3. Place the dataset CSV file in the required location.
4. Run the notebook cells sequentially.

## Google Colab

The notebook can be opened directly in Google Colab:

**Eda_assignment.ipynb**

## Author

Dhanasree Rajasekaran

B.Sc. Data Science
Interested in Data Analytics, Machine Learning and Business Intelligence.

## Future Improvements

This project can be extended by:

* Creating visualizations for categorical variables.
* Studying relationships between stress, treatment, and family history.
* Performing country-wise and occupation-wise analysis.
* Applying statistical analysis.
* Building an interactive dashboard.
* Developing machine learning models based on relevant target variables.

Note: This project is intended for educational and analytical purposes. The analysis describes patterns in the dataset and should not be interpreted as medical or clinical advice.



#**PROJECT - 2** 
# Stock Price Movement Predictor

## Project Overview

This project is a **Stock Price Movement Prediction and Analysis application** developed using Python, Machine Learning, technical indicators, and Streamlit. The application retrieves historical stock-market data using **Yahoo Finance (`yfinance`)**, calculates technical indicators such as **SMA, RSI, and MACD**, and uses a **Random Forest Classifier** to predict whether the stock price will move **Up (1)** or **Down (0)** on the following trading day.

The project also includes an interactive **Streamlit dashboard** where users can enter a stock ticker symbol, select a date range, visualize historical prices, examine technical indicators, and view prediction results.

## Objectives

* Analyze historical stock-market data.
* Retrieve real-time/historical stock data using Yahoo Finance.
* Calculate important technical indicators such as **SMA, RSI, and MACD**.
* Create an Up/Down target variable based on the next day's closing price.
* Train a **Random Forest Classification model** for stock price movement prediction.
* Evaluate the model using accuracy, classification report, and confusion matrix.
* Build an interactive **Streamlit web application** for stock analysis and prediction.
* Visualize stock prices and technical indicators to understand market trends.

## Technologies & Libraries Used

* **Python**
* **Pandas** – Data manipulation and preprocessing
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Visualization
* **yfinance** – Historical stock-market data
* **TA (Technical Analysis)** – Technical indicator calculation
* **Scikit-learn** – Machine learning and model evaluation
* **Random Forest Classifier** – Stock movement classification
* **Streamlit** – Interactive web application
* **Pyngrok** – Publicly exposing the Streamlit application during development
* **Google Colab** – Development and experimentation environment

## Data Collection

Historical stock data is collected using the `yfinance` library.

The application supports ticker symbols such as:

```text
AAPL
TSLA
INFY.NS
```

Users can provide a stock symbol and select the required start and end dates.

The downloaded data contains market information such as:

* Open Price
* High Price
* Low Price
* Closing Price
* Trading Volume

## Technical Indicators

The project uses technical indicators as features for the machine-learning model.

### 1. Simple Moving Average (SMA)

SMA calculates the average closing price over a specific number of trading periods.

The project uses a **14-day SMA** for the prediction model and also explores **20-day and 50-day moving averages** for visualization.

### 2. Relative Strength Index (RSI)

RSI is used to measure the momentum of price movements.

A 14-day RSI is calculated and included as one of the model features.

### 3. Moving Average Convergence Divergence (MACD)

MACD is used to analyze momentum and trend direction.

The project calculates MACD values and uses the MACD difference as a feature for the prediction model.

## Machine Learning Approach

The stock movement prediction problem is treated as a **binary classification problem**.

The target variable is created by comparing the next day's closing price with the current closing price:

```python
data['Target'] = data['Close'].shift(-1) > data['Close']
data['Target'] = data['Target'].astype(int)
```

The target values represent:

```text
1 → Price moves Up
0 → Price moves Down
```

### Features Used

The Random Forest model uses:

```text
SMA
RSI
MACD
```

as input features.

## Random Forest Classifier

A **Random Forest Classifier** is trained using the technical indicators.

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The dataset is divided into training and testing sets while maintaining the chronological order of the data:

```python
train_test_split(
    X,
    y,
    shuffle=False,
    test_size=0.2
)
```

This allows the model to be evaluated on a later portion of the dataset rather than randomly mixing observations from different time periods.

## Model Evaluation

The project evaluates the classification model using:

* Accuracy Score
* Classification Report
* Precision
* Recall
* F1-score
* Confusion Matrix
* Cross-validation

The notebook also includes an Iris dataset Random Forest experiment for demonstrating classification evaluation, where the reported test accuracy was **100%** and the mean 5-fold cross-validation accuracy was **96%**.

## Stock Visualization

The project visualizes historical stock prices together with technical indicators.

For example, the AAPL analysis includes:

* Closing Price
* 20-Day SMA
* 50-Day SMA
* RSI
* MACD
* MACD Signal Line

These visualizations provide a better understanding of price trends and momentum indicators.

## Streamlit Application

An interactive Streamlit application was developed as part of the project.

Users can:

1. Enter a stock ticker symbol.
2. Select a start and end date.
3. Download historical stock data.
4. View historical closing prices.
5. View raw stock-market data.
6. Analyze moving averages.
7. Generate stock movement predictions.
8. View the latest prediction results.

The application displays the predicted movement as:

```text
1 = Up
0 = Down
```

## Project Workflow

```text
User Input
    ↓
Stock Ticker & Date Range
    ↓
Yahoo Finance Data Collection
    ↓
Data Preprocessing
    ↓
Technical Indicator Calculation
    ↓
SMA + RSI + MACD
    ↓
Target Creation
    ↓
Train/Test Split
    ↓
Random Forest Classifier
    ↓
Prediction
    ↓
Model Evaluation
    ↓
Streamlit Visualization
```

## Project Structure

```text
Stock-Market-Predictor/
│
├── stock_market.ipynb
├── stock_predictor_app.py
├── app.py
├── model_accuracies_bar_chart.png
├── xgboost_confusion_matrix.png
└── README.md
```

##  How to Run the Project

### 1. Install Required Libraries

```bash
pip install yfinance ta scikit-learn streamlit pyngrok
```

### 2. Run the Streamlit Application

```bash
streamlit run app.py
```

### 3. Open the Application

Streamlit will provide a local URL that can be opened in a web browser.

The project was also tested using **Pyngrok** to expose the Streamlit application through a temporary public URL during development.

##  Key Features

### Historical Data Analysis

Retrieves and displays historical stock-market data for the selected ticker.

### Technical Analysis

Calculates SMA, RSI, and MACD indicators.

### Machine Learning Prediction

Uses Random Forest Classification to classify the expected next-day price movement.

### Visualization

Displays stock-price trends and technical indicators using interactive and static visualizations.

### Interactive Dashboard

Provides a user-friendly Streamlit interface for entering stock symbols and exploring market data.

##  Important Note

This project is developed for **educational and analytical purposes**. Stock-market movements are influenced by many factors, and technical indicators and machine-learning predictions do not guarantee future price movements. The predictions generated by this project should not be considered financial or investment advice.

## Future Improvements

The project can be further improved by:

* Adding additional technical indicators such as Bollinger Bands and Stochastic Oscillator.
* Comparing multiple machine-learning algorithms.
* Implementing time-series models such as LSTM.
* Adding XGBoost and other ensemble models.
* Improving feature engineering.
* Using larger and more recent datasets.
* Adding live prediction functionality.
* Creating interactive technical-analysis charts.
* Tracking model performance over different stocks.
* Deploying the Streamlit application permanently.

##  Author

**Dhanasree Rajasekaran**

B.Sc. Data Science
Interested in Data Analytics, Machine Learning, and Data-driven Applications.


**PROJECT - 3: Weather Prediction & Temperature Forecasting**

##  Project Overview

This project is a **Machine Learning and Deep Learning-based weather prediction system** that performs two major tasks: **weather condition classification** and **future temperature forecasting**.

The first part uses historical weather data and machine-learning algorithms such as **XGBoost, LightGBM, and Random Forest** to classify weather conditions into categories such as **Clear/Other, Cloudy, and Foggy**. The second part uses an **LSTM (Long Short-Term Memory) neural network** to forecast future temperature based on the previous **24 hours of temperature data**.

The project includes data preprocessing, feature engineering, class balancing using **SMOTE**, model comparison, hyperparameter tuning, model evaluation, feature-importance analysis, and model saving for future use.

---

##  Objectives

* Analyze historical weather data.
* Clean and preprocess weather observations.
* Extract useful time-based features from date and time information.
* Classify weather conditions using machine-learning models.
* Compare XGBoost, LightGBM, and Random Forest performance.
* Handle class imbalance using SMOTE.
* Optimize the XGBoost model using GridSearchCV.
* Forecast future temperature using an LSTM neural network.
* Evaluate classification and forecasting performance using appropriate metrics.
* Save trained models and preprocessing objects for future predictions.

---

##  Dataset

The project uses a historical weather dataset named:

```text
weatherHistory.csv
```

The dataset contains weather observations with variables such as:

* Formatted Date
* Temperature
* Apparent Temperature
* Weather Summary
* Other weather-related measurements

The date information is converted into useful temporal features including:

* Month
* Day
* Hour
* Time of Day

The project also creates a new `temp_diff` feature representing the difference between actual and apparent temperature.

---

#  Part 1: Weather Condition Classification

## Data Preprocessing

Several preprocessing steps were performed before training the classification models.

### Date-Time Feature Engineering

The `Formatted Date` column is converted into a datetime format and separated into:

```text
Month
Day
Hour
TimeOfDay
```

The time of day is categorized into:

```text
Night
Morning
Afternoon
Evening
```

This allows the models to capture possible relationships between weather conditions and different times of the day.

### Temperature Feature

A new feature called `temp_diff` is created:

```text
Temperature - Apparent Temperature
```

This represents the difference between the measured temperature and how the temperature feels.

### Data Cleaning

Unnecessary columns such as:

```text
Loud Cover
Daily Summary
```

are removed.

Missing values are handled using forward filling followed by removal of any remaining missing records.

---

## Weather Classification

The original weather summary is simplified into broader categories.

The classification logic groups weather descriptions into categories such as:

```text
Rainy
Snowy
Foggy
Cloudy
Windy
Dry
Humid
Clear/Other
```

The resulting target variable is encoded numerically using `LabelEncoder`.

This transforms the weather prediction task into a **multi-class classification problem**.

---

##  Feature Preparation

Categorical features are converted into numerical representations using **one-hot encoding**.

Numerical features are standardized using:

```python
StandardScaler()
```

Rare target classes containing fewer than six observations are filtered before applying SMOTE.

### SMOTE

**Synthetic Minority Over-sampling Technique (SMOTE)** is used to balance the remaining classes.

This helps prevent the classification model from being overly influenced by classes with more observations.

---

#  Machine Learning Models

Three classification algorithms were evaluated:

### 1. XGBoost

XGBoost was used as the primary model and subsequently optimized through hyperparameter tuning.

### 2. LightGBM

LightGBM was used as another gradient-boosting approach for comparison.

### 3. Random Forest

An optimized Random Forest model was also evaluated.

The models were evaluated using **TimeSeriesSplit cross-validation**, which preserves temporal ordering during validation.

---

##  Model Comparison

The TimeSeriesSplit cross-validation results reported in the notebook were:

| Model         | Mean CV Accuracy |
| ------------- | ---------------: |
| XGBoost       |           89.44% |
| LightGBM      |           89.39% |
| Random Forest |           78.41% |

The corresponding standard deviations were approximately **10.15%, 10.25%, and 19.46%**, respectively.

---

#  XGBoost Hyperparameter Tuning

GridSearchCV was used to optimize the XGBoost model.

The parameters explored included:

```text
n_estimators
max_depth
learning_rate
subsample
```

The best parameters obtained were:

```text
n_estimators = 200
max_depth = 6
learning_rate = 0.1
subsample = 0.7
```

The optimized model achieved a **test accuracy of approximately 95.33%** on the reported test set.

### Classification Performance

The reported test results were:

| Weather Class | Precision | Recall | F1-Score |
| ------------- | --------: | -----: | -------: |
| Clear/Other   |      0.94 |   0.92 |     0.93 |
| Cloudy        |      0.92 |   0.94 |     0.93 |
| Foggy         |      1.00 |   1.00 |     1.00 |

Overall test accuracy:

**95.33%**

---

# Part 2: LSTM Temperature Forecasting

The second component of the project focuses on **time-series temperature forecasting**.

Instead of classifying weather conditions, an LSTM neural network is used to predict future temperature values.

## Why LSTM?

LSTM networks are designed to work with sequential data and can learn patterns from previous observations.

In this project, the model uses the **previous 24 hours of temperature data to predict the temperature for the next hour**.

---

## LSTM Data Preparation

The temperature values are first normalized to the range:

```text
0 to 1
```

using `MinMaxScaler`.

Sequences are then generated using a window of:

```text
24 hours
```

For example:

```text
Previous 24 hours → Next hour temperature
```

The data is split chronologically into training and testing sets without shuffling, which is appropriate for time-series forecasting.

---

## LSTM Architecture

The LSTM model consists of:

```text
Input
  ↓
LSTM Layer – 50 units
  ↓
Dense Layer – 1 output
```

The model uses:

```text
Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Epochs: 10
Batch Size: 32
```

The model is trained using historical temperature sequences.

---

# LSTM Results

The training loss decreased throughout the 10 training epochs.

The final reported validation loss was approximately:

```text
0.000707
```

The model was then evaluated on the test sequence.

### Temperature Forecasting RMSE

The reported Root Mean Squared Error was:

**1.537°C**

The project also generates a visualization comparing:

```text
Actual Temperature
vs
Predicted Temperature
```

to visually assess the forecasting performance.

---

# Model Saving

The trained models and preprocessing components are saved for future use.

### Classification

```text
weather_model.pkl
weather_label_encoder.pkl
```

### Temperature Forecasting

```text
lstm_temperature_model.h5
temp_scaler.pkl
```

The notebook confirms that both the classification model and encoder, as well as the LSTM model and scaler, were saved successfully.

---

# Overall Project Workflow

```text
Historical Weather Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Time-Based Features
        ↓
Temperature Difference
        ↓
        ┌───────────────────────┐
        │                       │
        ↓                       ↓
Weather Classification     Temperature Forecasting
        ↓                       ↓
SMOTE Balancing             24-Hour Sequences
        ↓                       ↓
XGBoost / LightGBM / RF      LSTM Neural Network
        ↓                       ↓
Hyperparameter Tuning        Temperature Prediction
        ↓                       ↓
Weather Classification       RMSE Evaluation
        │                       │
        └───────────┬───────────┘
                    ↓
             Model Evaluation
                    ↓
             Save Trained Models
```

---

# Technologies Used

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
LightGBM
TensorFlow / Keras
Imbalanced-learn
Joblib
```

---

#  Project Structure

```text
Weather-Prediction/
│
├── weather_prediction.ipynb
├── weatherHistory.csv
│
├── weather_model.pkl
├── weather_label_encoder.pkl
│
├── lstm_temperature_model.h5
├── temp_scaler.pkl
│
├── confusion_matrix.png
├── feature_importance.png
└── README.md
```

---

# Key Highlights

* Built a **multi-class weather classification system**.
* Compared **XGBoost, LightGBM, and Random Forest**.
* Used **SMOTE** to address class imbalance.
* Applied **TimeSeriesSplit** for temporal cross-validation.
* Performed **XGBoost hyperparameter tuning** using GridSearchCV.
* Achieved **95.33% reported test accuracy** for the optimized classification model.
* Developed an **LSTM-based temperature forecasting model**.
* Used the previous **24 hours to predict the next hour's temperature**.
* Achieved a reported **1.537°C RMSE** for temperature forecasting.
* Saved trained models and preprocessing objects for future applications.

---

# Limitations

This project is intended as a machine-learning and deep-learning demonstration. Weather prediction is affected by many complex environmental factors, and model performance can vary depending on the dataset, geographical region, time period, and forecasting horizon.

The reported results are specific to the dataset and evaluation procedure used in this notebook and should not be interpreted as guaranteed real-world weather forecasts.

---

# Future Improvements

* Develop a Streamlit dashboard for real-time predictions.
* Incorporate additional meteorological variables.
* Add live weather API integration.
* Perform longer-horizon temperature forecasting.
* Experiment with GRU and Transformer-based time-series models.
* Add automated model retraining.
* Deploy the trained models as an API.
* Provide location-specific weather predictions.

## Author

**Dhanasree Rajasekaran**

B.Sc. Data Science
Interested in Data Analytics, Machine Learning, Deep Learning, and Data-driven Applications.

