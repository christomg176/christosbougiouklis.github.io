# 📦 Required Imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris  # Example dataset
from sklearn.metrics import accuracy_score, classification_report

# 📊 Load Sample Data (Iris for demo; replace with your own)
X, y = load_iris(return_X_y=True)

# 🔀 Split into Training and Test Sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ⚙️ Define Pipeline: Scaler → PCA → KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])

# 🔧 Hyperparameter Grid
param_grid = {
    'pca__n_components': [2, 3],
    'knn__n_neighbors': [3, 5, 7]
}

# 🔁 Cross-Validation Strategy (Stratified to keep class balance)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 🔍 Grid Search with Cross-Validation
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy'
)

# 🚂 Fit Model to Training Data
grid_search.fit(X_train, y_train)

# 🧠 Best Model Found
best_model = grid_search.best_estimator_

# 📈 Evaluate on Test Set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# 🖨️ Output Results
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
print("Test Set Accuracy with Best Parameters:", test_accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
