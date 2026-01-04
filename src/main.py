import nltk
import spacy
import math
import random
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize

# Download NLTK tokenizer (only needed once)
nltk.download('punkt')

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")


# Sample dataset of customer complaints
complaints = [
    "On 12 August 2024, I Ali Khan placed order ID 2001 for a mobile phone. The delivery was late.",
    "Order 2002 was delayed without notice.",
    "Parcel for Islamabad has not been delivered.",
    "Laptop shipment was delayed multiple times.",
    "Order marked delivered but never arrived.",

    "Washing machine arrived damaged.",
    "Smartphone screen broken and charger missing.",
    "Defective refrigerator delivered.",
    "Wrong product delivered.",
    "Microwave oven damaged at delivery.",

    "Payment deducted twice.",
    "Amount deducted but order not confirmed.",
    "Refund not processed.",
    "Incorrect billing charged.",
    "Refund not credited after cancellation.",

    "Customer service was rude.",
    "Customer care call disconnected.",
    "Service request not resolved.",
    "Support ticket closed without solution.",

    "Late delivery and damaged product.",
    "Late delivery and double payment.",
    "Refund not processed and service unresponsive.",
    "Defective product and delayed delivery.",
    "Late delivery, faulty product, refund pending."
]


# Preprocess text: lowercase, tokenize, remove punctuation
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum()]


# Create vocabulary from all complaints
vocab = list(set(word for c in complaints for word in preprocess(c)))
N = len(complaints)

# Calculate document frequency for IDF
df = defaultdict(int)
for word in vocab:
    for c in complaints:
        if word in preprocess(c):
            df[word] += 1


# Compute TF-IDF vector manually
def tf_idf(text):
    words = preprocess(text)
    tf = Counter(words)
    vector = []

    for word in vocab:
        tf_val = tf[word] / len(words) if words else 0
        idf_val = math.log((N + 1) / (df[word] + 1)) + 1
        vector.append(tf_val * idf_val)

    return vector


# Complaint categories
cat_to_index = {
    "Delivery": 0,
    "Product": 1,
    "Payment": 2,
    "Service": 3
}
index_to_cat = {v: k for k, v in cat_to_index.items()}


# Labels for training data
y_labels = [
    0,0,0,0,0,
    1,1,1,1,1,
    2,2,2,2,2,
    3,3,3,3,
    0,2,3,1,0
]

# Convert complaints to TF-IDF vectors
X = [tf_idf(c) for c in complaints]

num_features = len(X[0])
num_classes = 4
learning_rate = 0.05
epochs = 100

# Initialize weights for Logistic Regression
weights = [
    [random.uniform(-0.01, 0.01) for _ in range(num_features)]
    for _ in range(num_classes)
]


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Predict class using Logistic Regression
def predict_lr(x):
    scores = []
    for w in weights:
        z = sum(a * b for a, b in zip(x, w))
        scores.append(sigmoid(z))
    return scores.index(max(scores))


# Train Logistic Regression using gradient descent
for _ in range(epochs):
    for xi, yi in zip(X, y_labels):
        for k in range(num_classes):
            pred = sigmoid(sum(a * b for a, b in zip(xi, weights[k])))
            y_true = 1 if k == yi else 0

            for j in range(num_features):
                weights[k][j] += learning_rate * (y_true - pred) * xi[j]


# Classify a new complaint
def classify_complaint(text):
    return index_to_cat[predict_lr(tf_idf(text))]


# Build word frequency for summarization
all_words = []
for c in complaints:
    all_words.extend(preprocess(c))

word_freq = Counter(all_words)


# Simple frequency-based text summarization
def summarize(text, max_words=12):
    words = preprocess(text)
    important = [w for w in words if word_freq[w] > 1]
    return " ".join(important[:max_words])


# Extract named entities using spaCy
def extract_entities(text):
    doc = nlp(text)
    entities = defaultdict(list)

    for ent in doc.ents:
        entities[ent.label_].append(ent.text)

    return dict(entities)


# Compute cosine similarity between two vectors
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0


# Find similar complaints using cosine similarity
def find_similar(text, top_n=3):
    query_vec = tf_idf(text)
    scores = []

    for c in complaints:
        sim = cosine_similarity(query_vec, tf_idf(c))
        scores.append((sim, c))

    scores.sort(reverse=True)
    return scores[:top_n]


# Test complaint
test_complaint = """
I ordered a mobile phone.
Delivery was late, product damaged,
service did not respond and refund pending.
"""

print("SUMMARY:", summarize(test_complaint))
print("CATEGORY:", classify_complaint(test_complaint))
print("ENTITIES:", extract_entities(test_complaint))

print("\nSIMILAR COMPLAINTS:")
for score, comp in find_similar(test_complaint):
    print(f"- {comp} (Similarity: {score:.2f})")


# Visualize word frequency using matplotlib
words = preprocess(test_complaint)
freq = Counter(words)
labels, values = zip(*freq.most_common(10))

plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.title("Top Word Frequencies")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()
