import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report    
import joblib

# Download nltk data (run once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)                      # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)                  # Remove non-alphabetic chars
    text = text.lower()                                      # Lowercase
    tokens = word_tokenize(text)                             # Tokenize
    tokens = [w for w in tokens if w not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]      # Lemmatize
    return ' '.join(tokens)

# Load dataset
df = pd.read_csv('NLP Incremental/IMDB Dataset.csv')

tqdm.pandas()
df['cleaned_review'] = df['review'].progress_apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])

# Label encoding
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SGDClassifier for incremental learning with logistic regression loss
model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)

# partial_fit needs classes specified for first call
model.partial_fit(X_train, y_train, classes=[0, 1])

# Evaluate on test set
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save vectorizer and model
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'sentiment_model.pkl')

print("Training complete and models saved.")
