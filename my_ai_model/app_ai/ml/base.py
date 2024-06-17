import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud


#nltk.download('stopwords')
#nltk.download('punkt')


# Path to the CSV file
path_file = r"D:\Sentiment_analysis\my_ai_model\data\sentimental.csv"


# Load the CSV file
df = pd.read_csv(path_file)

# Adjust display options to show the full content of the DataFrame
pd.set_option('display.max_colwidth', None)  # Do not truncate column width
pd.set_option('display.max_columns', None)   # Display all columns
pd.set_option('display.max_rows', None)      # Display all rows (or a large number if the dataset is large)


# Count the number of null values in each column


# Display the number of null values in each column
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# Apply the function to the 'message to examine' column
df['message to examine'] = df['message to examine'].apply(remove_urls)

# Display the updated DataFrame
# def remove_stopwords(text):
#     stop_words = set(stopwords.words('english'))
#     word_tokens = word_tokenize(text)
#     filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
#     return ' '.join(filtered_text)
# df['message to examine'] = df['message to examine'].apply(remove_stopwords)

def remove_names(text):
    return re.sub(r'@ \w+', '', text)

# Apply the function to the 'message to examine' column
df['message to examine'] = df['message to examine'].apply(remove_names)

def clean_text(text):
    # Remove mentions, hashtags, and URLs
    text = re.sub(r'@\w+|#\w+|https?://\S+', '', text)
    # Remove special characters, punctuation, and emojis
    text = re.sub(r'[^\w\s]|_+', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Handle contractions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
   
    text = re.sub(r'\s+', ' ', text).strip()
    return text
df['message to examine'] = df['message to examine'].apply(clean_text)

df['message to examine'] = df['message to examine'].apply(word_tokenize)

lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens):
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


df['message to examine'] = df['message to examine'].apply(lemmatize_tokens)

short_words = {
"aint": "am not",
"arent": "are not",
"cant": "cannot",
"'cause": "because",
"couldve": "could have",
"couldnt": "could not",
"didnt": "did not",
"doesnt": "does not",
"dont": "do not",
"hadnt": "had not",
"hasnt": "has not",
"havent": "have not",
"im": "I am",
"em": "them",
"ive": "I have",
"isnt": "is not",
"lets": "let us",
"theyre": "they are",
"theyve": "they have",
"wasnt": "was not",
"well": "we will",
"were": "we are",
"werent": "were not",
"you're": "you are",
"you've": "you have"
}

def replace_short_words(tokens):
    return [short_words[word] if word in short_words else word for word in tokens]

df["message to examine"] =df['message to examine'].apply(lambda text: replace_short_words(text))

df['message to examine'] = df['message to examine'].apply(lambda tokens: ' '.join(tokens))


# Rename columns
df.columns = ['Index', 'Tweets', 'Labels']


# Convert the tweets to a list of sentences
sentences = df['Tweets'].tolist()

# Join all sentences into a single string
joined_sentences = " ".join(sentences)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(joined_sentences)

positive_tweets = df[df['Labels'] == 0]
positive_sentences = positive_tweets['Tweets'].tolist()
positive_string = " ".join(positive_sentences)


negative_tweets = df[df['Labels'] == 1]
negative_sentences = negative_tweets['Tweets'].tolist()
negative_string = " ".join(negative_sentences)

# plt.figure(figsize = (12,8))
# plt.imshow(WordCloud().generate(negative_string));
# plt.show()

cv = TfidfVectorizer()
tfidf= cv.fit_transform(df['Tweets'])


# Splitting Dataset
from sklearn.model_selection import train_test_split

tfX_train, tfX_test, tfy_train, tfy_test = train_test_split(tfidf, df['Labels'], test_size = 0.2)

# Training the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(tfX_train, tfy_train)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predicting labels for testing data
tfy_pred = model.predict(tfX_test)

# Calculate accuracy
# accuracy = accuracy_score(tfy_test, tfy_pred)
# print("Accuracy:", accuracy)

# # Generate classification report
# print("Classification Report:")
# print(classification_report(tfy_test, tfy_pred))

# # Generate confusion matrix
# print("Confusion Matrix:")
# print(confusion_matrix(tfy_test, tfy_pred))


# preprossessing

def preprocess_tweet(tweet):
    # Apply the same preprocessing steps as done for training data
    cleaned_tweet = clean_text(tweet)
    tokens = word_tokenize(cleaned_tweet)
    tokens = replace_short_words(tokens)
    tokens = lemmatize_tokens(tokens)
    preprocessed_tweet = ' '.join(tokens)
    return preprocessed_tweet

def predict_sentiment(tweet):
    # Preprocess the tweet
    preprocessed_tweet = preprocess_tweet(tweet)
    
    # Transform the preprocessed tweet into TF-IDF vector
    tweet_tfidf = cv.transform([preprocessed_tweet])
    
    # Make prediction using the trained model
    prediction = model.predict(tweet_tfidf)
    
    # Return prediction label (0 for negative, 1 for positive)
    return prediction[0]

def display_sentiment_prediction(tweet):
    prediction = predict_sentiment(tweet)
    if prediction == 1:
        print("Positive sentiment detected!")
    else:
        print("Negative sentiment detected.")





