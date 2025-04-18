import pickle
import numpy as np
from django.shortcuts import render, redirect
from django.urls import reverse
from .forms import WineInputForm

# Load model and selector
model = pickle.load(open("../data/best_rf_model.sav", 'rb'))
selector = pickle.load(open("../data/feature_selector.sav", 'rb'))

def home(request):
    if request.method == 'POST':
        form = WineInputForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            input_data = np.array([[data['volatile_acidity'], data['total_sulfur_dioxide'], data['sulphates'], data['alcohol']]])
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                result = "🍷 Excellent choice! This wine has great potential – smooth, balanced, and full of character. Enjoy! 🥂"
            else:
                result = "🍷 Hmm, this one might not be the best on the shelf. Consider exploring a richer option for a better experience!"

            # Store result in session and redirect
            request.session['result'] = result
            return redirect(reverse('wine_result'))

    else:
        form = WineInputForm()
    return render(request, 'wine_app/home.html', {'form': form})

def result(request):
    result = request.session.get('result', None)
    return render(request, 'wine_app/result.html', {'result': result})
