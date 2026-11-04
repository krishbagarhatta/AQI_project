import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

S0 = pd.read_csv("D:/5th sem/int 234 python/AQI project dataset.csv")
S1 = S0.copy()

#checks missing values
print("Missing Values:")
print(S1.isna().sum())

#remove rows with missing values
S1 = S1.dropna(subset=["pollutant_min",
                       "pollutant_max",
                       "pollutant_avg"])

#checks duplicate rows
print("Duplicate Rows:", S1.duplicated().sum())

#remove duplicates
S1 = S1.drop_duplicates()

#checks final dataset
print("Final Dataset Shape:", S1.shape)
print(S1.isna().sum())

#save cleaned dataset
S1.to_csv("AQI_Preprocessed.csv", index=False)
print("Preprocessing Completed")

#EDA
print("\nSTATISTICAL SUMMARY")
print(S1.describe())

print("\nDATASET INFORMATION")
print(S1.info())

print("\nMISSING VALUES")
print(S1.isna().sum())

# distribution of pollutant average
plt.figure(figsize=(8, 5))
sns.histplot(S1["pollutant_avg"], bins=30, kde=True)
plt.xlabel("Pollutant Average")
plt.ylabel("Frequency")
plt.title("Distribution of Pollutant Average")
plt.show()

# boxplot of pollutant average
plt.figure(figsize=(8, 5))
sns.boxplot(x=S1["pollutant_avg"])
plt.xlabel("Pollutant Average")
plt.title("Boxplot of Pollutant Average")
plt.show()

# correlation heatmap
plt.figure(figsize=(8, 6))

correlation = S1.select_dtypes(include="number").corr()

sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap")
plt.show()

# pollutant minimum vs maximum
plt.figure(figsize=(8, 5))

plt.scatter(S1["pollutant_min"], S1["pollutant_max"])

plt.xlabel("Pollutant Minimum")
plt.ylabel("Pollutant Maximum")
plt.title("Pollutant Minimum vs Maximum")
plt.show()


#linear regression



X = S1[["latitude", "longitude", "pollutant_min", "pollutant_max"]]
y = S1["pollutant_avg"]

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#train the model
model1 = LinearRegression()
model1.fit(X_train, y_train)

#predict values
y_pred = model1.predict(X_test)

#evaluating model performance
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, y_pred)
print("\nLINEAR REGRESSION RESULTS")
print("MAE:", MAE)
print("MSE:", MSE)
print("RMSE:", RMSE)
print("R2 Score:", R2)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Pollutant Average")
plt.ylabel("Predicted Pollutant Average")
plt.title("Linear Regression")
plt.show()


# polynomial regression

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model3 = LinearRegression()
model3.fit(X_train_poly, y_train)

y_pred3 = model3.predict(X_test_poly)

MAE3 = mean_absolute_error(y_test,y_pred3)
MSE3 = mean_squared_error(y_test,y_pred3)
RMSE3 = np.sqrt(MSE3)
R23 = r2_score(y_test, y_pred3)
print("\nPOLYNOMIAL REGRESSION RESULTS")
print("MAE:",MAE3)
print("MSE:",MSE3)
print("RMSE:",RMSE3)
print("R2 Score:",R23)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_test - y_pred3)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Actual Pollutant Average")
plt.ylabel("Prediction Error")
plt.title("Polynomial Regression - Residual Plot")
plt.show()

# logistic regression

# convert pollutant average into categories
y_class = (S1["pollutant_avg"] > S1["pollutant_avg"].median()).astype(int)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

model4 = LogisticRegression(max_iter=1000)
model4.fit(X_train2,y_train2)

y_pred4 = model4.predict(X_test2)

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", accuracy_score(y_test2,y_pred4))
print("Confusion Matrix:\n", confusion_matrix(y_test2,y_pred4))
print("Classification Report:\n", classification_report(y_test2,y_pred4))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test2, y_pred4),
            annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()


# KNN classification

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train2)
X_test_scaled = scaler.transform(X_test2)

model5 = KNeighborsClassifier(n_neighbors=5)
model5.fit(X_train_scaled, y_train2)

y_pred5 = model5.predict(X_test_scaled)
print("\nKNN RESULTS")

print("Accuracy:", accuracy_score(y_test2, y_pred5))
print("Confusion Matrix:\n", confusion_matrix(y_test2, y_pred5))
print("Classification Report:\n", classification_report(y_test2, y_pred5))

correct = (y_test2 == y_pred5).sum()
incorrect = (y_test2 != y_pred5).sum()

plt.figure(figsize=(6, 6))
plt.pie(
    [correct, incorrect],
    labels=["Correct Predictions", "Incorrect Predictions"],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"width": 0.4},
    pctdistance=0.8
)
plt.title("KNN Prediction Performance")
plt.tight_layout()
plt.show()

#svm classification

df = S1.copy()
df["target"] = (df["pollutant_avg"] > df["pollutant_avg"].median()).astype(int)

print(df.describe())
print(df.head())
print(df.info())

X = df[["latitude", "longitude"]].values
y = df["target"].values

svm = SVC(kernel="linear")
svm.fit(X, y)

DecisionBoundaryDisplay.from_estimator(
    svm,
    X,
    response_method="predict",
    alpha=0.8,
    cmap="Pastel1",
    xlabel="Latitude",
    ylabel="Longitude",
)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="yellow")
plt.show()
print("Accuracy:", svm.score(X, y))

#model comparison

# regression comparison
regression = pd.DataFrame({
    "Model": ["Linear", "Polynomial"],
    "MAE": [MAE, MAE3],
    "RMSE": [RMSE, RMSE3],
    "R2": [R2, R23]
})

print("\nRegression Comparison")
print(regression.round(3))


#regression graph
plt.figure(figsize=(7, 4))
plt.bar(regression["Model"], regression["R2"])

plt.title("Regression Model Comparison")
plt.xlabel("Models")
plt.ylabel("R2 Score")

plt.show()

#classification comparison
models = ["Logistic Regression", "KNN", "SVM"]
accuracies = [
    accuracy_score(y_test2, y_pred4),
    accuracy_score(y_test2, y_pred5),
    svm.score(X, y)
]

comparison = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracies
})

print("\nClassification Comparison")
print(comparison.round(3))

#classification graph

plt.figure(figsize=(7, 4))

plt.bar(comparison["Model"], comparison["Accuracy"])

plt.title("Classification Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.show()

