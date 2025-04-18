from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='wine_home'),
    path('result/', views.result, name='wine_result'),
]