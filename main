import random

class User:
    def __init__(self, name, interests):
        self.name = name
        self.interests = interests

    def __str__(self):
        return f"{self.name} is interested in {', '.join(self.interests)}"

class Match:
    def __init__(self, user1, user2, score):
        self.user1 = user1
        self.user2 = user2
        self.score = score

    def __str__(self):
        return f"{self.user1.name} and {self.user2.name} have a compatibility score of {self.score}%"

def generate_matches(users):
    # Generate a random compatibility score for each pair of users
    matches = []
    for i in range(len(users)):
        for j in range(i+1, len(users)):
            score = random.randint(0, 100)
            matches.append(Match(users[i], users[j], score))
    return matches

# Create some users
users = [
    User("Alice", ["music", "sports", "movies"]),
    User("Bob", ["books", "movies", "hiking"]),
    User("Charlie", ["music", "pets", "hiking"]),
    User("Diane", ["books", "music", "sports"])
]

# Generate matches for the users
matches = generate_matches(users)

# Print the matches
for match in matches:
    print(match)
