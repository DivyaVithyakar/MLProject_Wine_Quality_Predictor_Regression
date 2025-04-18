import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Load dataset
df = pd.read_csv("../data/winequality-red-cleaned.csv")

# Split features and target
X = df.drop("quality", axis=1)
y = (df["quality"] >= 6).astype(int)  # Binary classification: Good (>=6) vs Bad (<6)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search with Random Forest
param_grid = {
    'n_estimators': [100],
    'max_depth': [None],
    'min_samples_split': [5],
    'min_samples_leaf': [2]
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("✅ Best Parameters:", grid_search.best_params_)


# Feature Selection using SelectFromModel

selector = SelectFromModel(estimator=best_model, prefit=True, threshold='mean')

# Reduce train/test data
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_features = X_train.columns[selector.get_support()]
print(f"Number of Input Features (After Feature Selection): {X_train_selected.shape[1]}")
print("Selected Feature Names (After Feature Selection):\n", selected_features.tolist())


# Feature Importances
# Map importances back to selected features only
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
})
feature_importance_df = feature_importance_df[feature_importance_df['Feature'].isin(selected_features)]
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("📊 Feature Importances:\n", feature_importance_df)

# Plot top features
feature_importance_df.plot(kind='barh', x='Feature', y='Importance', legend=False)
plt.title("Top Feature Importances")
plt.gca().invert_yaxis()
plt.show()

# Train model with selected features
best_model.fit(X_train_selected, y_train)
y_pred = best_model.predict(X_test_selected)


# Evaluation

print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("📋 Classification Report:\n", classification_report(y_test, y_pred))
print("🌀 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model and selector using pickle
with open("../data/best_rf_model.sav", "wb") as f:
    pickle.dump(best_model, f)

with open("../data/feature_selector.sav", "wb") as f:
    pickle.dump(selector, f)

print("✅ Model and feature selector saved successfully!")