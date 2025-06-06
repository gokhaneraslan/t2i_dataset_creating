import os
import shutil
import argparse
from PIL import Image, UnidentifiedImageError


PILLOW_AVAILABLE = True

def get_supported_image_extensions():
    """Returns a list of supported image extensions."""
    
    common_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    return common_extensions

def rename_and_copy_images(source_folder, dest_folder, prefix="image", start_index=1, output_format=None):
    """
    Takes images from a source folder, renames them, and copies them to a destination folder.

    Args:
        source_folder (str): Source folder containing the images.
        dest_folder (str): Destination folder where renamed images will be saved.
        prefix (str): Prefix for the new filenames.
        start_index (int): Starting number for renaming.
        output_format (str, optional): Output image format (e.g., "jpg", "png").
                                       If None, the original format is preserved (only the extension might change if specified differently).
                                       If specified and Pillow is available, format conversion is performed.
    """
    
    if not os.path.isdir(source_folder):
        print(f"Error: Source folder not found -> {source_folder}")
        return 0

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Destination folder created: {dest_folder}")
        
    else:
        print(f"Images will be saved to this destination folder (already exists): {dest_folder}")

    supported_extensions = get_supported_image_extensions()
    image_files = []
    
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        
        if os.path.isfile(item_path) and os.path.splitext(item)[1].lower() in supported_extensions:
            image_files.append(item)

    if not image_files:
        print(f"No supported image formats found in the source folder ({source_folder}).")
        return 0

    image_files.sort()
    
    processed_count = 0
    for i, filename in enumerate(image_files, start=0): 
        original_filepath = os.path.join(source_folder, filename)
        original_ext = os.path.splitext(filename)[1].lower()


        if output_format:
            target_ext = f".{output_format.lower().lstrip('.')}"
        else:
            target_ext = original_ext
        

        new_filename = f"{prefix}_{start_index + i:06d}{target_ext}"
        new_filepath = os.path.join(dest_folder, new_filename)

        try:

            if output_format and PILLOW_AVAILABLE and target_ext != original_ext:
                
                try:
                    with Image.open(original_filepath) as img:
                        
                        if img.mode == 'RGBA' and target_ext.lower() in ['.jpg', '.jpeg']:
                            img = img.convert('RGB')
                            
                        elif img.mode == 'P' and target_ext.lower() in ['.jpg', '.jpeg']: 
                            img = img.convert('RGB')
                            
                        img.save(new_filepath)
                        
                    print(f"Copied and format converted: '{filename}' -> '{new_filename}'")
                    
                except UnidentifiedImageError:
                    
                    print(f"Error: '{filename}' could not be identified as a valid image. Skipping.")
                    continue
                
                except Exception as e_pil:
                    
                    print(f"Error: Pillow error while processing '{filename}': {e_pil}. Will only copy.")
                    shutil.copy2(original_filepath, new_filepath)
                    print(f"Copied (format could not be changed): '{filename}' -> '{new_filename}'")


            else:
                shutil.copy2(original_filepath, new_filepath)
                
                if output_format and original_ext != target_ext and not PILLOW_AVAILABLE:
                    print(f"Copied (Pillow not available, only extension changed): '{filename}' -> '{new_filename}'")
                    
                elif output_format and original_ext != target_ext and PILLOW_AVAILABLE: 
                     print(f"Copied (format was already the same or conversion not needed): '{filename}' -> '{new_filename}'")
                
                else:
                    print(f"Copied: '{filename}' -> '{new_filename}'")
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error: File '{filename}' could not be processed: {e}")

    print(f"\nA total of {processed_count} images were processed and saved to '{dest_folder}' folder.")
    return processed_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renames images in a folder and copies them to a new folder.")
    parser.add_argument("source_folder", help="Source folder containing the images.")
    parser.add_argument("dest_folder", help="Destination folder where renamed images will be saved.")
    parser.add_argument("--prefix", default="image", help="Prefix for new filenames (default: image).")
    parser.add_argument("--start_index", type=int, default=1, help="Starting number for renaming (default: 1).")
    parser.add_argument("--output_format", type=str, default=None, 
                        help="Output image format (e.g., jpg, png). If not specified, original format is preserved. "
                             "Pillow library is required for format conversion.")

    args = parser.parse_args()

    if args.output_format and not PILLOW_AVAILABLE:
        print("Warning: --output_format was specified, but Pillow library is not installed.")
        print("Images will only be copied, and file extensions will be changed; actual format conversion will not occur.")

    rename_and_copy_images(
        args.source_folder, 
        args.dest_folder, 
        args.prefix, 
        args.start_index, 
        args.output_format
    )