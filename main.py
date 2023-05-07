# %% Imports and reading data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("HousingData.csv")
nullValues = df.isnull().sum()
print(nullValues)  # checking for null values

# %%Describe the data
decData = df.describe()
print(decData)

# %% finding lowest, highest and mean price of the houses
min = df["MEDV"].min()
max = df["MEDV"].max()
mean = df["MEDV"].mean()
print(
    f"The minimum price is {min * 1000:.2f} and the maximum price is {max * 1000:.2f} and the mean price is {mean * 1000:.2f}")

# %% rescaling the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
rescaledData = scaler.fit_transform(df)

# %% Calculating the correlation between the data
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Correlation between the data")
plt.show()

# %% showing the correlation between median value of homes and the average number of rooms
plt.scatter(df['RM'], df['MEDV'], s=50, c='red', marker='o')
plt.xlabel('Average number of rooms per dwelling')
plt.ylabel('Median value of owner-occupied homes in $1000s')
plt.title('Scatter plot of RM and MEDV')
plt.show()

# %% showing the correlation between the per capital crime rate and the median value of homes
plt.scatter(df['CRIM'], df['MEDV'], s=50, c='blue', marker='o')
plt.xlabel('Per capital crime rate')
plt.ylabel('Median value of owner-occupied homes in $1000s')
plt.title('Scatter plot of CRIM and MEDV')
plt.show()

# %% showing the correlation between the nitric oxides concentration and the median value of homes
plt.scatter(df['NOX'], df['MEDV'], s=50, c='green', marker='o')
plt.xlabel('Nitric oxides concentration')
plt.ylabel('Median value of owner-occupied homes in $1000s')
plt.title('Scatter plot of NOX and MEDV')
plt.show()
