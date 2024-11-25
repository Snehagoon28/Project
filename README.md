
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/Users/sneha/Documents/Python/Titanic/Titanic.csv')

# Check column names
print(df.columns)

# Strip any extra spaces from column names
df.columns = df.columns.str.strip()

# Basic data overview
print(df.info())
print(df.describe())

# Data Cleaning: Fill missing values
# Check if 'Age' exists and fill missing values accordingly
if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())
else:
    print("'Age' column not found in the DataFrame.")

# Filling missing 'Embarked' with mode
if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
else:
    print("'Embarked' column not found in the DataFrame.")

# Univariate Analysis: Histogram of Age
if 'Age' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Age'], kde=True, color='blue')
    plt.title('Distribution of Age')
    plt.xlabel('Age')  # Added x-label for clarity
    plt.ylabel('Frequency')  # Added y-label for clarity
    plt.show()

# Bivariate Analysis: Correlation heatmap
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Boxplot: Age distribution by Pclass
if 'Pclass' in df.columns and 'Age' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Pclass', y='Age', data=df)
    plt.title('Age Distribution by Pclass')
    plt.xlabel('Passenger Class')  # Added x-label for clarity
    plt.ylabel('Age')  # Added y-label for clarity
    plt.show()

# Outlier detection with boxplot for Fare
if 'Fare' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['Fare'])
    plt.title('Fare Distribution with Outliers')
    plt.xlabel('Fare')  # Added x-label for clarity
    plt.show()
