import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)
import joblib
import json
import os

housing = pd.read_csv('housing.csv')
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

# ==========================
# Feature Engineering (Train)
# ==========================
strat_train_set["rooms_per_household"] = (
    strat_train_set["total_rooms"] / strat_train_set["households"]
)

strat_train_set["bedrooms_per_room"] = (
    strat_train_set["total_bedrooms"] / strat_train_set["total_rooms"]
)

strat_train_set["population_per_household"] = (
    strat_train_set["population"] / strat_train_set["households"]
)

# ==========================
# Feature Engineering (Test)
# ==========================
strat_test_set["rooms_per_household"] = (
    strat_test_set["total_rooms"] / strat_test_set["households"]
)

strat_test_set["bedrooms_per_room"] = (
    strat_test_set["total_bedrooms"] / strat_test_set["total_rooms"]
)

strat_test_set["population_per_household"] = (
    strat_test_set["population"] / strat_test_set["households"]
)

housing = strat_train_set.copy()


housing_labels = housing['median_house_value'].copy()
housing_features = housing.drop('median_house_value',axis=1)

num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. Pipelines
# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
 
# Categorical pipeline
cat_pipeline = Pipeline([
    # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
 
# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
 
# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing_features)
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"]
X_test_prepared = full_pipeline.transform(X_test)
 
# housing_prepared is now a NumPy array ready for training
print(housing_prepared.shape)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Training RMSE
lin_train_pred = lin_reg.predict(housing_prepared)
lin_train_rmse = root_mean_squared_error(housing_labels, lin_train_pred)

# Test RMSE
lin_test_pred = lin_reg.predict(X_test_prepared)
lin_test_rmse = root_mean_squared_error(y_test, lin_test_pred)

print("\n========== Linear Regression ==========")
print(f"Training RMSE : {lin_train_rmse:.2f}")
print(f"Test RMSE     : {lin_test_rmse:.2f}")

# Decision Tree
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

# Cross Validation
tree_rmses = -cross_val_score(
    tree_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

tree_cv_rmse = pd.Series(tree_rmses)

# Test Prediction
tree_test_pred = tree_reg.predict(X_test_prepared)
tree_test_rmse = root_mean_squared_error(y_test, tree_test_pred)

print("\n========== Decision Tree ==========")
print(f"Cross Validation Mean RMSE : {tree_cv_rmse.mean():.2f}")
print(f"Cross Validation Std Dev   : {tree_cv_rmse.std():.2f}")
print(f"Test RMSE                  : {tree_test_rmse:.2f}")

# Random Forest
forest_reg = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

forest_reg.fit(housing_prepared, housing_labels)

# Predictions
test_predictions = forest_reg.predict(X_test_prepared)

# Metrics
test_rmse = root_mean_squared_error(y_test, test_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("\n========== Random Forest ==========")
print(f"RMSE : {test_rmse:.2f}")
print(f"MAE  : {test_mae:.2f}")
print(f"R²   : {test_r2:.4f}")

print("\n========== Model Comparison ==========")
print(f"Linear Regression Test RMSE : {lin_test_rmse:.2f}")
print(f"Decision Tree Test RMSE     : {tree_test_rmse:.2f}")
print(f"Random Forest Test RMSE     : {test_rmse:.2f}")
print("======================================")

# Save the trained model and preprocessing pipeline
os.makedirs("models", exist_ok=True)

joblib.dump(forest_reg, "models/model.pkl")
joblib.dump(full_pipeline, "models/pipeline.pkl")

print("\n✅ Model saved as model.pkl")
print("✅ Pipeline saved as pipeline.pkl")

metrics = {
    "RMSE": round(test_rmse, 2),
    "MAE": round(test_mae, 2),
    "R2": round(test_r2, 4)
}

os.makedirs("artifacts", exist_ok=True)

with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Metrics saved.")
