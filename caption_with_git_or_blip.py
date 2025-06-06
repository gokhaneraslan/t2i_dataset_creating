import os
from PIL import Image, UnidentifiedImageError
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import csv
import json


def generate_caption(image_path, model, processor, device, max_length=200):
    """
    Generates a caption for a given image.
    """
    
    try:
        image = Image.open(image_path).convert("RGB")
        
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file {image_path}. It might be corrupted or not a supported format.")
        return None
    
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values=pixel_values, max_length=max_length)

    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption.strip()

def process_images_in_folder(image_folder, output_file, model_name="microsoft/git-base-coco", custom_style_caption=" "):
    """
    Generates captions for all images in a folder and saves them to a file.
    Supports .jsonl, .csv, .tsv, or plain .txt output.
    """
    
    if not os.path.isdir(image_folder):
        print(f"Error: Image folder not found -> {image_folder}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        
        if model_name == "Salesforce/blip-image-captioning-base":
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name)
            
        if model_name == "microsoft/git-base-coco":
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        model.eval()
        
    except Exception as e:
        print(f"Error loading model or processor '{model_name}': {e}")
        print("Make sure you have a working internet connection if downloading for the first time.")
        print("You might also need to accept terms if it's a gated model on Hugging Face Hub.")
        return

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"No supported image files found in {image_folder}.")
        return

    print(f"Found {len(image_files)} images to process.")

    output_format_type = "txt"
    if output_file.lower().endswith(".jsonl"):
        output_format_type = "jsonl"
        
    elif output_file.lower().endswith(".csv"):
        output_format_type = "csv"
        
    elif output_file.lower().endswith(".tsv"):
        output_format_type = "tsv"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = None
        
        if output_format_type == "csv":
            writer = csv.writer(f, delimiter=',')
            writer.writerow(["file_name", "text"])
            
        elif output_format_type == "tsv":
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["file_name", "text"])
            
        elif output_format_type == "txt":
            f.write("file_name\ttext\n")

        for i, image_filename in enumerate(image_files):
            image_path = os.path.join(image_folder, image_filename)
            
            print(f"Processing image {i+1}/{len(image_files)}: {image_filename} ...")

            caption_text = generate_caption(image_path, model, processor, device)

            if caption_text:
                print(f"  Caption: {caption_text}")
            else:
                caption_text = ""
                print(f"  Could not generate caption for {image_filename}.")
                
            caption_text = custom_style_caption + " " + caption_text
            
            if output_format_type == "jsonl":     
                json_record = {"file_name": image_filename, "text": caption_text}
                f.write(json.dumps(json_record, ensure_ascii=False) + "\n")
                
            elif output_format_type in ["csv", "tsv"]:
                writer.writerow([image_filename, caption_text])
                
            else:
                f.write(f"{image_filename}\t{caption_text}\n")

            f.flush()

    print(f"\nAll captions processed and saved to {output_file} (format: {output_format_type.upper()})")

if __name__ == "__main__":
    
    MODEL_ID = "microsoft/git-base-coco"
    #MODEL_ID = "Salesforce/blip-image-captioning-base"

    IMAGE_INPUT_FOLDER = "/content/MyImgDataset/images"
    METADATA_OUTPUT_FILE = "/content/MyImgDataset/metadata.jsonl"
    custom_style_caption = "gaya_style"

    process_images_in_folder(
        IMAGE_INPUT_FOLDER, 
        METADATA_OUTPUT_FILE, 
        MODEL_ID, 
        custom_style_caption
    )