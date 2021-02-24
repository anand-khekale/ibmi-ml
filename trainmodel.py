# NumPy is used to handle the data to put it in a 2-dimensional array.
import numpy as np
import pandas as pd  # Pandas is used to read the data from database or csv.
# Import the LinearRegression class of Sci-kit learn library.
from sklearn.linear_model import LinearRegression
# Import r2_score to measure the performance of our model.
from sklearn.metrics import r2_score
# Split the data as train and test set to test the model.
from sklearn.model_selection import train_test_split
import pickle  # To save the trained model into a file.

# df is a short form of the dataframe, we read the data from the csv and put it in a pandas dataframe object.
df = pd.read_csv('Salary_Data.csv')

# Reshape the data, meaning take one column, the feature, and put it in a 2-D array in a variable named X.
X = df['YearsExperience'].values.reshape(-1, 1)

# Reshape the data,same for the values we are giving to the model based on which it will predict future values in
# variable y.
y = df['Salary'].values.reshape(-1, 1)

# Split the data into the Train and Test sections so that we can measure the performance of the model.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Initialize the Scikit learn LinearRegression Class and fit (train the model with training data)
reg = LinearRegression()
reg.fit(X_train, y_train)

# Use Test data to derive the prediction in y_pred array
y_pred = reg.predict(X_test)

# Print the score of the model by comparing it with predicted value and actual test values.
# Score closer to 1 is better performance.
print(f'Linear Regression model built with score: {r2_score(y_test, y_pred)}')

# Save the model to the disk for future use
filename = 'finalized_model.sav'
pickle.dump(reg, open(filename, 'wb'))

print('Model saved in file', filename)
