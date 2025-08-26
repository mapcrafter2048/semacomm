#!/usr/bin/env python3
import zipfile
import os
import argparse
from pathlib import Path

def is_image_file(filename):
    """Checks if a filename has a common image extension."""
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))

def extract_first_image_from_sequence_folders(zip_filepath, output_base_dir):
    """
    Extracts the first image from each sequence subfolder within a ZIP file.

    The expected structure inside the zip is:
    zip_file_root/
    ├── some_main_folder/
    │   ├── annotations/
    │   │   └── ...
    │   └── sequences/
    │       ├── sequence_folder_1/
    │       │   ├── image1.jpg
    │       │   ├── image1_variant.jpg
    │       │   └── ...
    │       ├── sequence_folder_2/
    │       │   ├── imageA.png
    │       │   └── ...
    │       └── ...
    """
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zf:
            print(f"Processing ZIP file: {zip_filepath}")
            processed_sequence_subfolders = set() # To keep track of which sequence subfolders we've taken an image from

            # Sort namelist to ensure consistent "first" file selection if order matters
            # and multiple images are directly in the sequence subfolder.
            # Zip order is usually fixed, but explicit sort doesn't hurt.
            member_list = sorted(zf.namelist())

            for member_path_str in member_list:
                member_path = Path(member_path_str)

                # Check if the path looks like it's inside a 'sequences' subfolder
                # and contains an image.
                # Example: root_folder_in_zip/sequences/actual_image_folder/image.jpg
                # parts will be ['root_folder_in_zip', 'sequences', 'actual_image_folder', 'image.jpg']
                parts = list(member_path.parts)

                try:
                    sequences_index = parts.index("sequences")
                except ValueError:
                    # "sequences" not in the path, skip
                    continue

                # We need at least one folder after "sequences" and then the image file
                # So, path should be like .../sequences/subfolder/image.ext
                if len(parts) > sequences_index + 2 and is_image_file(parts[-1]):
                    # The path to the specific sequence subfolder (e.g., "root_folder_in_zip/sequences/actual_image_folder")
                    sequence_subfolder_path_in_zip = Path(*parts[:sequences_index + 2])
                    sequence_subfolder_name = parts[sequences_index + 1] # e.g., "actual_image_folder"

                    if str(sequence_subfolder_path_in_zip) not in processed_sequence_subfolders:
                        # This is the first image file we've encountered in this particular sequence subfolder
                        original_image_filename = parts[-1]
                        zip_basename = Path(zip_filepath).stem # Get filename without .zip

                        # Create a more unique output filename
                        output_filename = f"{zip_basename}_{sequence_subfolder_name}_{original_image_filename}"
                        output_image_path = Path(output_base_dir) / output_filename

                        print(f"  Extracting: '{member_path_str}' from '{sequence_subfolder_name}'")
                        print(f"  Saving to: '{output_image_path}'")

                        # Extract the file content
                        image_data = zf.read(member_path_str)

                        # Write to the output directory
                        with open(output_image_path, 'wb') as outfile:
                            outfile.write(image_data)

                        processed_sequence_subfolders.add(str(sequence_subfolder_path_in_zip))
                    # else:
                        # print(f"  Skipping: '{member_path_str}' (already processed an image for '{sequence_subfolder_path_in_zip}')")

    except FileNotFoundError:
        print(f"Error: ZIP file not found at '{zip_filepath}'")
    except zipfile.BadZipFile:
        print(f"Error: Bad ZIP file or not a ZIP file: '{zip_filepath}'")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{zip_filepath}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Extracts the first image from each 'sequences' subfolder within specified ZIP files."
    )
    parser.add_argument(
        "zip_files",
        metavar="ZIP_FILE",
        type=str,
        nargs='+',
        help="Path to one or more ZIP files to process."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Directory where the extracted first images will be saved."
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {Path(output_dir).resolve()}")

    for zip_file_path in args.zip_files:
        extract_first_image_from_sequence_folders(zip_file_path, output_dir)
        print("-" * 30)

    print("Processing complete.")

if __name__ == "__main__":
    main()
