import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import process_mp3_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert MP3 files to WAV files while maintaining folder structure."
    )
    parser.add_argument("src_folder", help="Source folder containing MP3 files")
    parser.add_argument("target_folder", help="Target folder for WAV files")
    parser.add_argument(
        "--use_ffmpeg", action="store_true", help="Use ffmpeg instead of pydub for conversion"
    )
    args = parser.parse_args()

    src_folder = os.path.abspath(args.src_folder)
    target_folder = os.path.abspath(args.target_folder)

    if not os.path.exists(src_folder):
        print(f"Error: Source folder '{src_folder}' does not exist.")
        return

    os.makedirs(target_folder, exist_ok=True)

    process_mp3_files(src_folder, target_folder, use_ffmpeg=args.use_ffmpeg)


if __name__ == "__main__":
    main()
