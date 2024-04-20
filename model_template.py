import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Load the data
genotypes = pd.read_csv('Genotypic_data_maf10_min10_291acc.txt', index_col=0)
phenotype = pd.read_csv('phenodata_BLUP_2012.txt', sep='\t', index_col='ID')

# Function to calculate minor allele frequency
def calculate_maf(df):
    maf = df.apply(lambda x: min(x.mean(), 1 - x.mean()), axis=0)
    return maf

# Function for LD pruning
def ld_pruning(df, threshold=0.5):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))  # Updated to use bool
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(to_drop, axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    genotypes, phenotype['AVGROW97'], test_size=0.2, random_state=42)

# Apply MAF and LD pruning only to the training set
maf = calculate_maf(X_train)
X_train_filtered = X_train.loc[:, maf > 0.05]  # Filter out SNPs with MAF <= 5%
X_train_pruned = ld_pruning(X_train_filtered)

# Impute missing data and scale the data
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test.columns)

# Elastic Net for feature selection
elastic_net_cv = ElasticNetCV(cv=5, random_state=0, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
elastic_net_cv.fit(X_train_scaled, y_train)

# Filter features based on coefficients
selected_features = X_train_scaled.columns[elastic_net_cv.coef_ != 0]

# Export selected features
pd.Series(selected_features).to_csv('selected_features_elastic_net.csv', index=False)

# Using selected features for modeling
X_train_selected = X_train_scaled[selected_features]
X_test_selected = X_test_scaled[selected_features]

# Linear regression with cross-validation
regressor = LinearRegression()
cv_scores = cross_val_score(regressor, X_train_selected, y_train, cv=5, scoring='neg_mean_squared_error')
regressor.fit(X_train_selected, y_train)
y_pred = regressor.predict(X_test_selected)

# Calculate RMSE and R-squared
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mean_cv_rmse = np.sqrt(-cv_scores.mean())

print(f'Test RMSE: {rmse}')
print(f'Test R²: {r2}')
print(f'Cross-Validated RMSE: {mean_cv_rmse}')


# Visualization of predictions vs. actual
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Predictions vs. Actual Values')
plt.show()

# Visualization of feature importances from Elastic Net
plt.figure(figsize=(10, 5))
importances = elastic_net_cv.coef_[elastic_net_cv.coef_ != 0]
sorted_indices = np.argsort(importances)[::-1]
plt.title('Feature Importances from Elastic Net')
plt.bar(range(len(importances)), importances[sorted_indices], color="b", align="center")
plt.xticks(range(len(importances)), selected_features[sorted_indices], rotation=90)
plt.xlim([-1, len(importances)])
plt.show()
