import pickle
import numpy as np

# Load model and feature selector
load_model = pickle.load(open("../data/best_rf_model.sav", 'rb'))
load_feature = pickle.load(open("../data/feature_selector.sav", 'rb'))

# Sample input data (4 selected features)
new_data_raw = np.array([[0.34, 63, 0.79, 14]])  # ['volatile acidity', 'total sulfur dioxide', 'sulphates', 'alcohol']

# Predict quality
prediction = load_model.predict(new_data_raw)

# output based on prediction
if prediction[0] == 1:
    print("🍷 Excellent choice! This wine has great potential – smooth, balanced, and full of character. Enjoy! 🥂")
else:
    print("🍷 Hmm, this one might not be the best on the shelf. Consider exploring a richer option for a better experience!")
