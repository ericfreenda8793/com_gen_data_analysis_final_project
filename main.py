import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix

# load in the data
genotypes = pd.read_csv('Genotypic_data_maf10_min10_291acc.txt', index_col=0)
phenotype = pd.read_csv('phenodata_BLUP_2012.txt', sep='\t', index_col='ID')

# 'genotypes' and 'phenotype' are already loaded and aligned by their indices
# now we check for missing data
imputer = SimpleImputer(strategy='median')
genotypes_imputed = pd.DataFrame(imputer.fit_transform(genotypes), columns=genotypes.columns)
phenotype_imputed = pd.DataFrame(imputer.fit_transform(phenotype), columns=phenotype.columns)

# Scale the data
scaler = StandardScaler()
genotypes_scaled = pd.DataFrame(scaler.fit_transform(genotypes_imputed), columns=genotypes.columns)

# Check for any remaining NaNs or infinities
genotypes_scaled = genotypes_scaled.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
phenotype_scaled = pd.DataFrame(scaler.fit_transform(phenotype_imputed), columns=phenotype.columns)

# Extract the AVGROW97 column from the phenotype dataframe
y = phenotype_scaled['AVGROW97']
# Construct X from the genotype dataframe
X = genotypes_scaled

# Determine the number of samples (rows) and features (columns)
num_samples, num_features = X.shape

print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

# Perform PCA on the genotype data
pca = PCA(n_components=5)
principal_components = pca.fit_transform(genotypes_scaled)

# Plot the variance explained by each principal component
plt.figure(figsize=(8, 6))
# Adjust the x-range to match the number of PCA components
plt.plot(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Select components that explain at least 80% of the variance
cum_var_explained = np.cumsum(pca.explained_variance_ratio_)
num_components = np.where(cum_var_explained > 0.8)[0][0] + 1 if np.any(cum_var_explained > 0.8) else len(cum_var_explained)
pca_selected = principal_components[:, :num_components]

# Calculate the kinship matrix from PCA components
distances = euclidean_distances(principal_components)
# Threshold distances to create a sparser kinship matrix
threshold_distance = np.percentile(distances, 75)  # Keep distances below the 50th percentile
kinship_matrix = np.where(distances < threshold_distance, 1 - distances / np.max(distances), 0)
# If the exog_re (kinship matrix) can be made sparse,
# it might reduce memory usage significantly.
# Convert the kinship matrix to a sparse matrix
# Assume distances are computed from PCA as before

# Add a constant to the genotype data for the intercept
X_with_const = sm.add_constant(genotypes_scaled)  # Adds an intercept term to the predictors

# Create a DataFrame for the PCA components
pca_df = pd.DataFrame(pca_selected, columns=[f'PC_{i}' for i in range(pca_selected.shape[1])])

# Concatenate the PCA components with the genotype data
covariates = pd.concat([X_with_const, pca_df], axis=1)

# Check for NaN values in the covariates or the response
if covariates.isnull().any().any():
    raise ValueError("Covariates DataFrame contains NaN values")
if np.any(pd.isnull(y)):
    raise ValueError("y contains NaN or infinite values")

# Ensure the kinship matrix is correctly sized and formatted
kinship_matrix_shape = kinship_matrix.shape
if kinship_matrix_shape[0] != kinship_matrix_shape[1] or kinship_matrix_shape[0] != len(y):
    raise ValueError("Kinship matrix dimensions mismatch or do not match the number of observations in y")

# Create a model instance using all SNPs
groups = pd.Series(np.ones(len(y)))  # Assuming only one group for simplicity
model = MixedLM(y, covariates, groups=groups, exog_re=kinship_matrix)
result = model.fit(method='cg')

# Store and print the summary of the results
print(result.summary())
