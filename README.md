                          ###SynerGee
**SynerGee**: 
The Science of Connection".  combines "synergy" and "energy," creating meaningful, compatible connections between people based on various data-driven factors.

                           ** Overview **
SynerGee is an advanced social matching platform that leverages cutting-edge machine learning algorithms and natural language processing to create meaningful connections. Unlike traditional platforms that rely solely on shared interests, SynerGee incorporates multiple layers of compatibility metrics to ensure a more nuanced match.


# SynerGee: The Science of Connection

SynerGee is an advanced social matching platform that leverages cutting-edge machine learning algorithms and natural language processing to create meaningful connections. Unlike traditional platforms that rely solely on shared interests, SynerGee incorporates multiple layers of compatibility metrics to ensure a more nuanced match.

## SynerGee: combines "synergy" and "energy," creating meaningful, compatible connections between people based on various data-driven factors.

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository**

    ```bash
    https://github.com/KidusB9/SynerGee-Matchmaker-Machine-Learning.git
    ```

2. **Navigate to the directory**

    ```bash
    cd SynerGee
    ```

3. **Install the required packages**

    ```bash
    pip install Flask TextBlob numpy scikit-learn redis
    pip install Flask
    pip install TextBlob
    pip install numpy
    pip install scikit-learn
    pip install Flask-Session
or go the the requirement.txt file

    ```

## Features


### Basic Compatibility Scoring
At its core, SynerGee uses a shared-interest algorithm that calculates compatibility based on the number of interests two users have in common.

### Advanced Compatibility Scoring
Building on the basic model, the advanced compatibility feature uses cosine similarity to compare interest vectors, providing a more nuanced understanding of how well two users might get along.

### Sentiment-Adjusted Compatibility
SynerGee analyzes the sentiment behind each user's interests using NLP techniques. This allows the platform to adjust compatibility scores based on the emotional context of the interests, adding another layer of sophistication to the matching algorithm.

### Longevity Prediction
Using a neural network model, SynerGee predicts the longevity of a match. This feature provides an estimate of how long the connection between two users is likely to last, offering a unique perspective on the quality of the match.

### SuperMatch Prediction
This feature employs linear regression to predict future compatibility based on past interactions. It's like having a crystal ball that tells you who your best match could be in the future!

### Interest Recommendation System
SynerGee doesn't just stop at matching. It also recommends new interests to users based on their compatibility with other users, encouraging personal growth and exploration.

### High Compatibility Notifications
Users receive real-time notifications for high compatibility matches, ensuring they don't miss out on a potentially meaningful connection.


## Usage

1. **Initialize the Flask app**

    ```bash
    python app.py
    ```

2. **Open your web browser and go to**

    ```
    http://127.0.0.1:5000/
   you have to add this in the main section as 0.0.0.0 or whatever you want it to be
    ```

3. **Register or log in to start finding matches**

## Code Highlights

- **User and Match Classes**

    ```python
    class User:
        def __init__(self, name, interests):
            self.name = name
            self.interests = interests
    ```

    ```python
    class Match:
        def __init__(self, user1, user2, score):
            self.user1 = user1
            self.user2 = user2
            self.score = score
    ```

- **Compatibility Calculation**

    ```python
    def calculate_compatibility(user1, user2):
        shared_interests = len(set(user1.interests).intersection(set(user2.interests)))
        total_interests = len(set(user1.interests).union(set(user2.interests)))
        return (shared_interests / total_interests) * 100
    ```

- **Longevity Prediction**

    ```python
    def train_longevity_model(matches):
    X = []
    y = []
    for match in matches:
        shared_interests = len(set(match.user1.interests).intersection(set(match.user2.interests)))
        X.append([match.score, shared_interests])
        # Simulate longevity score (i have encontered errors here but this should work as expected)
        y.append(random.randint(50, 100) if match.score > 70 else random.randint(0, 50))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Longevity model trained with R^2 score: {score}")
    return model
    ```

- **Flask Routes**

    ```python
    @app.route('/')
    def index():
        return render_template('index.html')
    ```

## Contributing

Pull requests are welcome and appricated lets make this the next big thing please feel free to contributie. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License. See `LICENSE` for more information.

**By combining these features, SynerGee offers a multi-dimensional approach to social matching, going beyond surface-level similarities to create connections that are truly meaningful.**


