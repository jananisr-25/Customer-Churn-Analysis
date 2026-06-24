import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Make plots look clean
sns.set(style="whitegrid")

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Churn_Modelling.csv")

print("Shape:", df.shape)
df.head()

df.info()
df.describe()
df.isnull().sum()

df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# Gender: Male = 1, Female = 0
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
# Geography: One-hot encoding
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)
df.head()

churn_rate = df["Exited"].mean()
print(f"Overall Churn Rate: {churn_rate:.2%}")

plt.figure(figsize=(8,5))
sns.barplot(x="Tenure", y="Exited", data=df)
plt.title("Churn Rate by Customer Tenure")
plt.ylabel("Churn Rate")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="Exited", y="Balance", data=df)
plt.title("Balance Distribution by Churn")
plt.xlabel("Churned (1 = Yes)")
plt.show()

sns.barplot(x="IsActiveMember", y="Exited", data=df)
plt.title("Churn Rate by Activity Status")
plt.xlabel("Active Member (1 = Active)")
plt.ylabel("Churn Rate")
plt.show()

def tenure_group(t):
    if t <= 2:
        return "New"
    elif t <= 5:
        return "Mid"
    else:
        return "Long"

df["TenureGroup"] = df["Tenure"].apply(tenure_group)

sns.barplot(x="TenureGroup", y="Exited", data=df)
plt.title("Churn Rate by Tenure Group")
plt.show()

# Total balance held by churned customers
revenue_at_risk = df[df["Exited"] == 1]["Balance"].sum()
print(f"Total Balance at Risk due to Churn: ₹{revenue_at_risk:,.0f}")

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df.drop(columns=["Exited"])
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": X.columns,"Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

importance

df["ChurnProbability"] = model.predict_proba(X)[:,1]

df["RiskBucket"] = pd.cut(df["ChurnProbability"],
    bins=[0, 0.3, 0.6, 1], labels=["Low", "Medium", "High"])

df["RiskBucket"].value_counts()

df.to_csv("final_churn_analysis.csv", index=False)
























