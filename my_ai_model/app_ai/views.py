from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from django.http import HttpResponse
from .ml.base import predict_sentiment
from .ml.base import display_sentiment_prediction

def display_sentiment_prediction(request):
    tweet=request.GET['tweet']  
    prediction = predict_sentiment(tweet)
    if prediction == 1:
        return render(request, 'index.html', {'prediction_text':'Positive Sentiment'})
    else:
        return render(request, 'index.html', {'prediction_text':'Negative Sentiment'})
    return render(request, 'index.html')
    
    
    