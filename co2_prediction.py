# Import libraries
from sklearn import linear_model as lmd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
import seaborn as sb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse

# Load the dataset
df = pd.read_csv('co2.csv')

# Show the distribution of CO2 emissions
sb.countplot(x='out1', data=df)

# Display the correlation matrix
plt.subplots(figsize=(9, 9))
plt.axis('off')
sb.heatmap(df.corr(), annot=True)

# Select input features (X) and target variable (y)
x = df[['engine', 'cylandr', 'fuelcomb']]
y = df['out1']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = tts(
    x, y, test_size=0.2, random_state=42
)

# Create a Linear Regression model
model = lmd.LinearRegression()

# Train the model using the training data
model.fit(x_train, y_train)

# Predict CO2 emissions for the test data
outputbot = model.predict(x_test)

# Calculate the Mean Squared Error (MSE)
mse1 = mse(y_test, outputbot)

# Create a 3D plot to visualize the data and regression plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the actual test data
ax.scatter(
    x_test['engine'],
    x_test['cylandr'],
    y_test
)

# Create values for the engine and cylinder axes
engine = np.linspace(
    x_test['engine'].min(),
    x_test['engine'].max(),
    20
)

cylandr = np.linspace(
    x_test['cylandr'].min(),
    x_test['cylandr'].max(),
    20
)

# Create a coordinate grid for the regression plane
ENGINE, CYLANDR = np.meshgrid(engine, cylandr)

# Keep fuel consumption constant at its average value
FUELCOMB = np.full(
    ENGINE.shape,
    x_test['fuelcomb'].mean()
)

# Calculate the predicted CO2 emissions for the regression plane
OUT1 = (
    model.intercept_
    + model.coef_[0] * ENGINE
    + model.coef_[1] * CYLANDR
    + model.coef_[2] * FUELCOMB
)

# Plot the regression plane
ax.plot_surface(
    ENGINE,
    CYLANDR,
    OUT1,
    alpha=0.4
)

# Add labels to the axes
ax.set_xlabel('Engine')
ax.set_ylabel('Cylinders')
ax.set_zlabel('CO2 Emissions')

# Display the MSE on the graph
ax.text2D(
    0.05,
    0.95,
    f"MSE = {mse1:.2f}",
    transform=ax.transAxes
)

# Show the plot
plt.show()

# Get information about a car from the user
car = input("Enter your car name: ")
Engine = float(input("Enter the engine size in liters: "))
Cylandr = int(input("Enter the number of cylinders: "))
Fuelcomb = float(input("Enter the fuel consumption: "))

# Prepare the user's car data for prediction
car_input = np.array([[Engine, Cylandr, Fuelcomb]])

# Predict CO2 emissions for the user's car
output = model.predict(car_input)

# Display the prediction
print("%s can make %.2f CO2 in each kilometer" % (car, output[0]))
