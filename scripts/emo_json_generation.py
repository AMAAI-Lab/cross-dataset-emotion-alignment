import json
from collections import Counter

import pandas as pd

# Load the data from CSV file
csv_file_path = "./data/data.csv"
data = pd.read_csv(csv_file_path)
print(data.head())
print(data.columns)
data.columns = data.columns.str.strip()
# Define the list of emotion columns
emotion_columns = [
    "amazement",
    "solemnity",
    "tenderness",
    "nostalgia",
    "calmness",
    "power",
    "joyful_activation",
    "tension",
    "sadness",
]

# Initialize a dictionary to store the results
results = {}

# Group by track id and genre
grouped = data.groupby(["track id", "genre"])

# Perform majority voting for each group
for (track_id, genre), group in grouped:
    # Count the occurrences of each emotion
    emotion_counts = Counter()
    for emotion in emotion_columns:
        emotion_counts[emotion] += group[emotion].sum()

    # Find the top 3 emotions
    top_emotions = [emotion for emotion, count in emotion_counts.most_common(3)]

    # Construct the result entry
    result_entry = {
        "path": f"/alignment/data/emotifymusic/{genre}/{track_id}.mp3",  # Adjust the path as needed
        "mood/theme": top_emotions,
    }

    # Add the result entry to the results dictionary
    results[f"{track_id},{genre}"] = result_entry

# Save the results to a JSON file
json_file_path = "results.json"
with open(json_file_path, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"Results saved to {json_file_path}")
