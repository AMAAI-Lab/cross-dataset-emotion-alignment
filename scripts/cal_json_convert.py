import json
import os


def update_json_keys_and_paths(input_json_path, output_json_path, new_base_path):
    # Load the JSON data
    with open(input_json_path) as f:
        data = json.load(f)

    # Function to convert mood/theme format
    def convert_mood_theme(mood_theme_list):
        converted_list = []
        for mood in mood_theme_list:
            # Remove "Emotion-" prefix and replace "_/_"
            mood = mood.replace("Emotion-", "").replace("_/_", ", ").lower()
            converted_list.append(mood)
        return converted_list

    # Create a new dictionary with updated keys, paths, and mood/theme
    updated_data = {}
    all_labels = set()
    for i, (key, value) in enumerate(data.items()):
        new_key = str(i)
        new_path = os.path.join(new_base_path, f"{key}.wav")
        if os.path.exists(new_path):
            new_mood_theme = convert_mood_theme(value["mood/theme"])
            updated_data[new_key] = {"path": new_path, "mood/theme": new_mood_theme}
            all_labels.update(new_mood_theme)
        else:
            print(f"No such song exists: {new_path}")

    # Save the updated JSON data to a new file
    with open(output_json_path, "w") as f:
        json.dump(updated_data, f, indent=4)

    # Print all collected labels
    print("All labels:", sorted(all_labels))


if __name__ == "__main__":
    input_json_path = "/renhangx/alignment/resources/CAL_500_all_emotions.json"
    output_json_path = "/renhangx/alignment/resources/CAL_500_all_emotions_updated.json"
    new_base_path = "/renhangx/alignment/datasets/CAL500_32kps_wav"

    update_json_keys_and_paths(input_json_path, output_json_path, new_base_path)
    print(f"Updated JSON saved to {output_json_path}")
