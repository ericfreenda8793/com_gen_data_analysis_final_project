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

# load in the data
genotypes = pd.read_csv('tomatoes/Genotypic_data_maf10_min10_291acc.txt', index_col=0)
phenotype = pd.read_csv('tomatoes/phenodata_BLUP_2012.txt', sep='\t', index_col='ID')
# 'genotypes' and 'phenotype' are already loaded and aligned by their indices
# now we check for missing data
imputer = SimpleImputer(strategy='median')
genotypes_imputed = pd.DataFrame(imputer.fit_transform(genotypes), columns=genotypes.columns)
phenotype_imputed = pd.DataFrame(imputer.fit_transform(phenotype), columns=phenotype.columns)

# Scale the data
scaler = StandardScaler()
genotypes_scaled = pd.DataFrame(scaler.fit_transform(genotypes_imputed), columns=genotypes.columns)
phenotype_scaled = pd.DataFrame(scaler.fit_transform(phenotype_imputed), columns=phenotype.columns)


# Extract the AVGROW97 column from the phenotype dataframe
y = phenotype_scaled['AVGROW97']
# Construct X from the genotype dataframe
X = genotypes_scaled

# Determine the number of samples (rows) and features (columns)
num_samples, num_features = X.shape

print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

from sklearn.linear_model import LassoCV

# Lasso with built-in cross-validation to choose the best alpha
lasso = LassoCV(cv=5, random_state=0, max_iter=10000).fit(X, y)
selected_features_lasso = np.where(lasso.coef_ != 0)[0]
# save selected features from Lasso
np.savetxt("selected_features_lasso.txt", selected_features_lasso, fmt='%d')
# save Lasso coefficients
np.savetxt("lasso_coefficients.txt", lasso.coef_, fmt='%f')