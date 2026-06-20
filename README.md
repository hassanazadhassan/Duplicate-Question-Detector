Duplicate Question Detector
This project tackles a classic NLP problem from the Quora Question Pairs Kaggle competition: given two questions, determine whether they're asking the same thing, even if worded completely differently. For example, "What is the capital of Pakistan?" and "Do you know the capital of Pakistan?" should both be flagged as duplicates, despite sharing little surface structure.
Rather than relying on a single similarity metric, the model combines four categories of engineered features:

Token features — ratios of shared words and stopwords between the two questions, plus whether they start or end with the same word.
Length features — differences in question length and the longest common substring shared between them.
Fuzzy matching features — Levenshtein-distance-based scores (QRatio, partial ratio, token sort ratio, token set ratio) that catch rephrased or reordered questions.
Bag-of-Words vectors — a CountVectorizer-based representation capturing core vocabulary overlap between the two inputs.

All of these are concatenated into a single feature vector and passed into a Random Forest Classifier trained on labeled question pairs, which predicts whether the pair is duplicate or distinct.
The trained model is deployed in an interactive Streamlit web app. Users type two questions into side-by-side text boxes and click "Check Similarity" to get an instant prediction, along with a "How it works" tab explaining the underlying methodology for anyone curious about the mechanics.
One interesting edge case I worked through during development: very short questions made entirely of stopwords (like "how are you?") initially confused the non-stopword overlap features, since stopword removal left nothing to compare. I patched the feature extraction logic so that when both questions reduce to nothing after stopword removal, they're treated as a full match rather than a forced mismatch — a small but meaningful fix for short, conversational-style questions.
Tech stack: Python, scikit-learn (Random Forest, CountVectorizer), NLTK, FuzzyWuzzy, BeautifulSoup, NumPy, Streamlit
dataset: (https://www.kaggle.com/datasets/bruce0001/quora-question-pairs)
