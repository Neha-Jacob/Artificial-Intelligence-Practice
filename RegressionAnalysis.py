from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA

# Load the diabetes dataset
data = datasets.load_diabetes()

# Split the dataset into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(data.data, data.target, train_size=0.8,
                                                                    random_state=50)

# Scale the data
scaler = preprocessing.MinMaxScaler()
scaler.fit(data_train)
data_scaled_train = scaler.transform(data_train)
data_scaled_test = scaler.transform(data_test)

# Apply PCA
pca = PCA(n_components=5)
pca.fit(data_scaled_train)
data_lower_dim_train = pca.transform(data_scaled_train)
data_lower_dim_test = pca.transform(data_scaled_test)

# Print explained variance
print("PCA explained variance:", pca.explained_variance_)

# Fit the KNN Regressor
model = KNeighborsRegressor(n_neighbors=25)
model.fit(data_lower_dim_train, target_train)
result = model.predict(data_lower_dim_test)

# Calculate regression metrics
mse = mean_squared_error(target_test, result)
r2 = r2_score(target_test, result)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Print the first 20 predicted and true values
print("Predicted values:", result[:20])
print("True values:", target_test[:20])

# Find the best number of neighbors
best_mse = float('inf')
best_n_neighbors = 0

print("Varying n_neighbors")
for i in range(1, 50):
    model = KNeighborsRegressor(n_neighbors=i)
    model.fit(data_lower_dim_train, target_train)
    result = model.predict(data_lower_dim_test)

    mse = mean_squared_error(target_test, result)
    print(f"n_neighbors={i}, Mean Squared Error: {mse:.2f}")

    # Update the best MSE and the corresponding number of neighbors
    if mse < best_mse:
        best_mse = mse
        best_n_neighbors = i

# Print the best result
print(f"\nBest n_neighbors={best_n_neighbors} with Mean Squared Error: {best_mse:.2f}")
