from django.urls import path
from . import views

urlpatterns = [
    path('ai_model/', views.your_view_function, name='display_sentiment_prediction'),
    # Add more URL patterns here
]
