from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
import json
import time
import os
import torch


PROMPT_CONFIG = {
    "detailed_art_analyst": {
        "role": """You are an expert art and image analyst.
        Your task is to provide a comprehensive description of the visual elements in the presented image.
        Focus on the main subject(s), their attributes, the overall composition, dominant colors and color harmonies, lighting, and prominent stylistic characteristics (e.g., art movement, rendering style, brushwork, line quality).
        The description should be a coherent narrative.
        Do not add any text before or after the description itself.""",
        "instructions": """Generate a detailed, a few sentence description of the provided image, aiming for a length suitable for up to 200 tokens.
        **Begin your description directly with the most prominent subject, action, or atmospheric element. Avoid starting with phrases like "The image depicts...", "This scene shows...", or similar introductory statements.**
        Your description should thoroughly cover the following aspects:
        1.  **Main Subject(s) & Scene:** ...
        2.  **Color Palette & Lighting:** ...
        3.  **Composition & Form (Lines):** ...
        4.  **Artistic Style & Technique:** ...

        Combine these elements into a flowing, descriptive narrative.
        The caption should be suitable for a dataset where captions are typically under 200 tokens.
        Absolutely no conversational fillers, titles, external interpretations, or bullet points.
        Output only the descriptive text itself."""
    },
    "concise_captioner": {
        "role": """You are an image captioning assistant.
        Your task is to create a brief, factual caption for the image, highlighting only the most essential visual elements.
        The caption must be a single, short sentence.
        Do not add any text before or after the caption itself.""",
        "instructions": """Generate a very concise, single-sentence caption for the provided image.
        Focus on the main subject, its key attributes, and the immediate context or action.
        The caption should be suitable for a dataset where captions are typically under 200 tokens.
        Absolutely no conversational fillers, titles, or section headers.
        Output only the caption itself."""
    }
}

def initialize_gemma_model(model_id="google/gemma-3-12b-it"):
    """
    Initializes and returns the Gemma 3 model and processor.
    """
    
    print(f"Initializing Gemma model: {model_id}...")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    try:
        
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
        )
        model.eval()
        
        processor = AutoProcessor.from_pretrained(model_id)
        print("Gemma model and processor initialized successfully.")
        
        return model, processor
    
    except Exception as e:
        print(f"Error initializing Gemma model or processor: {e}")
        print("Please ensure you have enough GPU memory and a working internet connection for the first download.")
        return None, None

def generate_gemma_description(image_path, model, processor, prompt_style="concise_captioner", max_new_tokens=220):
    """
    Generates a detailed description for a given image using the Gemma model.
    """
    
    active_prompt = PROMPT_CONFIG.get(prompt_style)
    if not active_prompt:
        print(f"Warning: Unknown prompt style '{prompt_style}'. Using default style.")
        active_prompt = PROMPT_CONFIG["concise_captioner"]

    model_role = active_prompt["role"]
    model_instructions = active_prompt["instructions"]

    try:
        image = Image.open(image_path)
        image.load()
        
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    
    except (UnidentifiedImageError, OSError) as img_e:
        print(f"Error: Cannot open or load image file {image_path}. It might be corrupt or an unsupported format. Details: {img_e}")
        return None

    messages = [
        {"role": "system", "content": [{"type": "text", "text": model_role}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": model_instructions}
            ]
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded.strip()

    except RuntimeError as e:
        
        if "out of memory" in str(e).lower():
            print(f"CUDA out of memory while processing {image_path}. Skipping this image.")
            torch.cuda.empty_cache()
            return None
        
        else:
            print(f"Runtime error while processing {image_path}: {e}")
            return None
        
    except Exception as e:
        print(f"Error during description generation for {image_path}: {e}")
        return None

def create_metadata_jsonl(image_folder, output_file, gemma_model, gemma_processor, custom_style_caption):
    """
    Generates descriptions for all images in a folder using Gemma and saves them
    to a .jsonl file. Features a resume capability to continue where it left off.
    """
    
    if not os.path.isdir(image_folder):
        print(f"Error: Image folder not found -> {image_folder}")
        return

    if gemma_model is None or gemma_processor is None:
        print("Error: Gemma model or processor not initialized. Aborting.")
        return


    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    all_image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(supported_extensions)])

    if not all_image_files:
        print(f"No supported image files found in {image_folder}.")
        return


    processed_files = set()
    if os.path.exists(output_file):
        print(f"Existing output file found. Reading to resume: {output_file}")
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f_read:
                for line in f_read:
                    try:
                        record = json.loads(line)
                        if 'file_name' in record:
                            processed_files.add(record['file_name'])
                            
                    except json.JSONDecodeError:
                        print(f"Warning: Skipped a malformed line in existing file: {line.strip()}")
                        
            print(f"Resuming. Found {len(processed_files)} already processed images.")
            
        except Exception as e:
            print(f"Warning: Could not read resume data from {output_file}. Starting from scratch. Error: {e}")

    image_files_to_process = [f for f in all_image_files if f not in processed_files]

    if not image_files_to_process:
        print("All images have already been processed. Nothing to do.")
        return


    total_to_process = len(image_files_to_process)
    print(f"Found {total_to_process} new images to process in '{image_folder}'.")
    print(f"Output will be saved to: {output_file}")


    with open(output_file, 'a', encoding='utf-8') as f_jsonl:
        
        processed_count_this_run = 0
        for i, image_filename in enumerate(image_files_to_process):
            image_path = os.path.join(image_folder, image_filename)
            start_time = time.time()
            
            print(f"\nProcessing image {i+1}/{total_to_process}: {image_filename} ...")

            description_text = generate_gemma_description(image_path, gemma_model, gemma_processor)

            end_time = time.time()
            elapsed_time = end_time - start_time

            if description_text:
                
                print(f"  Description (took {elapsed_time:.2f}s): {description_text[:150]}...")
                
                description_text = custom_style_caption + ", " + description_text
                
                json_record = {"file_name": image_filename, "text": description_text}
                f_jsonl.write(json.dumps(json_record, ensure_ascii=False) + "\n")
                f_jsonl.flush()
                
                processed_count_this_run += 1
                
            else:
                print(f"  Could not generate description for {image_filename} (took {elapsed_time:.2f}s). Skipping.")

    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed {processed_count_this_run} images in this session.")
    print(f"Metadata saved to {output_file}")

if __name__ == "__main__":
    
    GEMMA_MODEL_ID = "google/gemma-3-12b-it"
    IMAGE_INPUT_FOLDER = "/content/MyImgDataset/images"
    METADATA_OUTPUT_FILE = "/content/MyImgDataset/metadata.jsonl"
    custom_style_caption = "your_style"

    gemma_model, gemma_processor = initialize_gemma_model(GEMMA_MODEL_ID)

    if gemma_model and gemma_processor:
        
        create_metadata_jsonl(
            IMAGE_INPUT_FOLDER, 
            METADATA_OUTPUT_FILE, 
            gemma_model, 
            gemma_processor, 
            custom_style_caption
        )
        
    else:
        print("Failed to initialize Gemma model. Exiting.")
