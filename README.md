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
