import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import convert_track_metadata_to_wav


def main():
    parser = argparse.ArgumentParser(
        description="Convert MP3 files to WAV files while maintaining folder structure."
    )
    parser.add_argument("metadata_path", help="Path to the metadata file")
    parser.add_argument("metadata_wav_path", help="Path to the metadata file")
    args = parser.parse_args()

    convert_track_metadata_to_wav(args.metadata_path, args.metadata_wav_path)


if __name__ == "__main__":
    main()
