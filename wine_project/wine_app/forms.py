from django import forms

class WineInputForm(forms.Form):
    volatile_acidity = forms.FloatField(label='Volatile Acidity')
    total_sulfur_dioxide = forms.FloatField(label='Total Sulfur Dioxide')
    sulphates = forms.FloatField(label='Sulphates')
    alcohol = forms.FloatField(label='Alcohol')
