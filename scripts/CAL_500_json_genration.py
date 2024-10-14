import json
import os
import re
from collections import defaultdict

# Folder where the files are located
folder_path = "/root/alignment/data/annotations"


# Function to parse the emotions from a file
def parse_emotions(file_path):
    emotions = {}
    with open(file_path) as file:
        for line in file:
            # Find lines that match the "Emotion-" pattern
            match = re.match(r'(Emotion-[\w/_]+)\s*=\s*"(\d+)"', line)
            if match:
                emotion = match.group(1)
                value = int(match.group(2))
                emotions[emotion] = value
    return emotions


# Function to calculate average emotion values across multiple files
def calculate_average_emotions(file_prefix, file_suffixes):
    emotion_sums = defaultdict(int)
    emotion_counts = defaultdict(int)

    # Iterate over all files with the same prefix
    for suffix in file_suffixes:
        file_name = f"{file_prefix}_{suffix}.txt"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            emotions = parse_emotions(file_path)

            for emotion, value in emotions.items():
                emotion_sums[emotion] += value
                emotion_counts[emotion] += 1

    # Calculate averages
    emotion_averages = {
        emotion: emotion_sums[emotion] / emotion_counts[emotion] for emotion in emotion_sums
    }
    return emotion_averages


# Function to collect emotions for each file prefix
def collect_emotions(file_prefix, emotion_averages):
    # Filter emotions with average values greater than 3
    filtered_emotions = [emotion for emotion, avg in emotion_averages.items() if avg > 4]
    if not filtered_emotions:
        max_emotion = max(emotion_averages, key=emotion_averages.get)
        filtered_emotions = [max_emotion]
    else:
        filtered_emotions = sorted(emotion_averages, key=emotion_averages.get, reverse=True)[:5]
    # Prepare the format
    result = {"path": os.path.join(folder_path, file_prefix), "mood/theme": filtered_emotions}

    return result


# Function to get unique file prefixes and corresponding suffixes
def get_file_prefixes_and_suffixes():
    prefixes = defaultdict(list)
    for file_name in os.listdir(folder_path):
        match = re.match(r"(.*)_(\d{2})\.txt", file_name)
        if match:
            prefix = match.group(1)
            suffix = match.group(2)
            prefixes[prefix].append(suffix)
    return prefixes


# Main function
def main():
    output_file = "all_emotions.json"  # Single output JSON file
    data = {}

    # Get all unique file prefixes and their suffixes
    file_prefixes_and_suffixes = get_file_prefixes_and_suffixes()

    # Process each file prefix and accumulate the results
    for file_prefix, file_suffixes in file_prefixes_and_suffixes.items():
        # Calculate the average emotions
        emotion_averages = calculate_average_emotions(file_prefix, file_suffixes)

        # Collect emotions for this prefix
        data[file_prefix] = collect_emotions(file_prefix, emotion_averages)

    # Write all results into a single JSON file
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    main()
