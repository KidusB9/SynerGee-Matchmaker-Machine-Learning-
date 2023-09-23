import random
from textblob import TextBlob
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template, redirect, url_for, jsonify

class User:
    def __init__(self, name, interests):
        self.name = name
        self.interests = interests
        self.past_matches = []  # Store past matches and their scores

    def __str__(self):
        return f"{self.name} is interested in {', '.join(self.interests)}"

class Match:
    def __init__(self, user1, user2, score):
        self.user1 = user1
        self.user2 = user2
        self.score = score

    def __str__(self):
        return f"{self.user1.name} and {self.user2.name} have a compatibility score of {self.score}%"

def calculate_compatibility(user1, user2):
    # Calculate compatibility based on shared interests
    shared_interests = len(set(user1.interests).intersection(set(user2.interests)))
    total_interests = len(set(user1.interests).union(set(user2.interests)))
    return (shared_interests / total_interests) * 100

def train_longevity_model(matches):
    X = []
    y = []
    for match in matches:
        shared_interests = len(set(match.user1.interests).intersection(set(match.user2.interests)))
        X.append([match.score, shared_interests])
        # Simulate longevity score (for demonstration purposes)
        y.append(random.randint(50, 100) if match.score > 70 else random.randint(0, 50))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Longevity model trained with R^2 score: {score}")
    return model

# Function to predict match longevity
def predict_longevity(match, model):
    shared_interests = len(set(match.user1.interests).intersection(set(match.user2.interests)))
    longevity = model.predict([[match.score, shared_interests]])[0]
    print(f"Predicted longevity for {match.user1.name} and {match.user2.name}: {longevity:.2f}")

# Train the longevity model
longevity_model = train_longevity_model(matches)

# Predict the longevity of each match
for match in matches:
    predict_longevity(match, longevity_model)

def notify_users(match):
    if match.score > 80:
        print(f"Notification: {match.user1.name} and {match.user2.name}, you are a high compatibility match!")

def super_match(user, all_users):
    # Use machine learning to predict future compatibility
    X = []
    y = []
    for past_match in user.past_matches:
        X.append([all_users.index(past_match[0])])
        y.append(past_match[1])
    
    if len(X) < 2:
        print(f"Insufficient data to predict super match for {user.name}.")
        return
    
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(np.array([[i] for i in range(len(all_users))]))
    best_match_index = np.argmax(predictions)
    
    if all_users[best_match_index] == user:
        print(f"{user.name}, you are your own best match!")
    else:
        print(f"SuperMatch: {user.name}, your future best match could be {all_users[best_match_index].name}.")

def interest_to_vector(interests, all_interests):
    return [1 if interest in interests else 0 for interest in all_interests]

# Function to calculate advanced compatibility using cosine similarity
def advanced_compatibility(user1_vector, user2_vector):
    similarity = cosine_similarity([user1_vector], [user2_vector])
    return similarity[0][0] * 100

# Function to recommend new interests to a user
def recommend_interests(user, all_users, all_interests):
    similar_users = []
    for other_user in all_users:
        if user != other_user:
            user_vector = interest_to_vector(user.interests, all_interests)
            other_user_vector = interest_to_vector(other_user.interests, all_interests)
            score = advanced_compatibility(user_vector, other_user_vector)
            if score > 70:
                similar_users.append(other_user)
def analyze_interest_sentiment(interests):
    sentiment_score = 0
    for interest in interests:
        blob = TextBlob(interest)
        sentiment_score += blob.sentiment.polarity
    return sentiment_score / len(interests) if interests else 0

# Function to calculate compatibility with sentiment analysis
def sentiment_adjusted_compatibility(user1, user2):
    base_score = calculate_compatibility(user1, user2)
    user1_sentiment = analyze_interest_sentiment(user1.interests)
    user2_sentiment = analyze_interest_sentiment(user2.interests)
    
    # Adjust the compatibility score based on sentiment
    adjustment = (user1_sentiment + user2_sentiment) * 10  # Scale factor
    return min(max(base_score + adjustment, 0), 100)  # Keep score between 0 and 100

# Update the matching logic with sentiment-adjusted compatibility
sentiment_adjusted_matches = []
for i in range(len(users)):
    for j in range(i+1, len(users)):
        score = sentiment_adjusted_compatibility(users[i], users[j])
        match = Match(users[i], users[j], score)
        sentiment_adjusted_matches.append(match)

        # Update past matches for SuperMatch feature
        users[i].past_matches.append((users[j], score))
        users[j].past_matches.append((users[i], score))

# Print the sentiment-adjusted matches
print("Sentiment-Adjusted Matches:")
for match in sentiment_adjusted_matches:
    print(match)
    
    recommended_interests = Counter()
    for similar_user in similar_users:
        for interest in similar_user.interests:
            if interest not in user.interests:
                recommended_interests[interest] += 1
    
    if recommended_interests:
        most_common_interests = recommended_interests.most_common(3)
        print(f"Recommended interests for {user.name}: {[x[0] for x in most_common_interests]}")
    else:
        print(f"No new interests to recommend for {user.name}.")

# Collect all unique interests
all_interests = list(set([interest for user in users for interest in user.interests]))

# Update the matching and recommendation logic
for i in range(len(users)):
    for j in range(i+1, len(users)):
        user1_vector = interest_to_vector(users[i].interests, all_interests)
        user2_vector = interest_to_vector(users[j].interests, all_interests)
        
        score = advanced_compatibility(user1_vector, user2_vector)
        match = Match(users[i], users[j], score)
        matches.append(match)
        
        # Update past matches for SuperMatch feature
        users[i].past_matches.append((users[j], score))
        users[j].past_matches.append((users[i], score))

# Recommend new interests to each user
for user in users:
    recommend_interests(user, users, all_interests)

# Create some users
users = [
    User("Alice", ["music", "sports", "movies"]),
    User("Bob", ["books", "movies", "hiking"]),
    User("Charlie", ["music", "pets", "hiking"]),
    User("Diane", ["books", "music", "sports"])
]

# Generate matches for the users
matches = []
for i in range(len(users)):
    for j in range(i+1, len(users)):
        score = calculate_compatibility(users[i], users[j])
        match = Match(users[i], users[j], score)
        matches.append(match)
        
        # Store past matches for SuperMatch feature
        users[i].past_matches.append((users[j], score))
        users[j].past_matches.append((users[i], score))

# Print and notify the matches
for match in matches:
    print(match)
    notify_users(match)

# Predict future best matches using SuperMatch
for user in users:
    super_match(user, users)

app = Flask(__name__)

registered_users = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register_user():
    name = request.form['name']
    interests = request.form['interests'].split(',')
    new_user = User(name, interests)
    registered_users.append(new_user)
    return redirect(url_for('index'))

@app.route('/generate_matches', methods=['GET'])
def generate_matches_route():
    matches = []
    for i in range(len(registered_users)):
        for j in range(i+1, len(registered_users)):
            score = calculate_compatibility(registered_users[i], registered_users[j])
            match = Match(registered_users[i], registered_users[j], score)
            matches.append(str(match))
    return jsonify({"matches": matches})

@app.route('/recommend_interests', methods=['GET'])
def recommend_interests_route():
    all_interests = list(set([interest for user in registered_users for interest in user.interests]))
    recommendations = {}
    for user in registered_users:
        recommend_interests(user, registered_users, all_interests)
        recommendations[user.name] = user.interests  # Replace this with actual recommendations
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
