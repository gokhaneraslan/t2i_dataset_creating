# Text-to-Image Dataset Preparation Toolkit

A comprehensive collection of Python scripts for preparing high-quality datasets for text-to-image model training. This toolkit provides end-to-end functionality from extracting frames from videos to generating detailed image captions using state-of-the-art vision-language models.

## 🚀 Features

- **Video Frame Extraction**: Extract frames from videos at specified intervals
- **Image Processing**: Rename, resize, and format conversion utilities
- **Automated Captioning**: Generate detailed captions using multiple vision models
- **Advanced Description Generation**: Create comprehensive image descriptions with artistic analysis
- **Flexible Output Formats**: Support for JSONL, CSV, TSV, and TXT formats
- **Resume Capability**: Continue processing from where you left off
- **Custom Style Prefixes**: Add custom style tokens to captions
- **GPU Acceleration**: Optimized for CUDA-enabled systems

## 📦 Installation

### Prerequisites

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install transformers
pip install Pillow
pip install opencv-python

# For quantization (optional but recommended for large models)
pip install bitsandbytes accelerate
```

### Clone Repository

```bash
git clone https://github.com/yourusername/text-to-image-dataset-toolkit.git
cd text-to-image-dataset-toolkit
```

## 🛠️ Scripts Overview

### 1. Video Frame Extractor (`video_frame_extractor.py`)

Extract frames from videos at specified time or frame intervals.

**Features:**
- Extract frames by time intervals (seconds) or frame count
- Process single videos or entire folders
- Multiple output formats (JPG, PNG)
- Automatic folder organization by video name

**Usage:**
```bash
# Extract frame every 2 seconds
python video_frame_extractor.py input_video.mp4 output_folder -s 2.0

# Extract every 30th frame
python video_frame_extractor.py input_video.mp4 output_folder -f 30

# Process entire folder
python video_frame_extractor.py video_folder/ output_folder -s 1.5 --format png
```

### 2. Image Organizer (`image_organizer.py`)

Rename and organize images with sequential numbering and format conversion.

**Features:**
- Batch rename images with custom prefixes
- Format conversion between image types
- Automatic RGB conversion for JPEG compatibility
- Sequential numbering with zero-padding

**Usage:**
```bash
# Basic renaming
python image_organizer.py source_folder dest_folder --prefix "dataset_img"

# With format conversion
python image_organizer.py source_folder dest_folder --prefix "img" --output_format jpg --start_index 1000
```

### 3. Basic Image Captioner (`basic_captioner.py`)

Generate captions using lightweight vision models (BLIP, GIT).

**Features:**
- Support for Microsoft GIT and Salesforce BLIP models
- Multiple output formats (JSONL, CSV, TSV, TXT)
- Custom style prefix addition
- GPU acceleration support

**Usage:**
```python
# Configuration
MODEL_ID = "microsoft/git-base-coco"  # or "Salesforce/blip-image-captioning-base"
IMAGE_INPUT_FOLDER = "/path/to/images"
METADATA_OUTPUT_FILE = "/path/to/metadata.jsonl"
custom_style_caption = "your_style_token"

# Run the script
python basic_captioner.py
```

### 4. Advanced Description Generator (`advanced_captioner.py`)

Generate detailed, artistic descriptions using Google's Gemma-3 model.

**Features:**
- Two captioning modes: detailed analysis and concise captioning
- Advanced prompt engineering for artistic analysis
- 4-bit quantization for memory optimization
- Resume capability for large datasets
- Comprehensive error handling and memory management

**Caption Styles:**
- `detailed_art_analyst`: Comprehensive artistic analysis including composition, color theory, and style
- `concise_captioner`: Brief, factual descriptions for general use

**Usage:**
```python
# Configuration
GEMMA_MODEL_ID = "google/gemma-3-12b-it"
IMAGE_INPUT_FOLDER = "/path/to/images"
METADATA_OUTPUT_FILE = "/path/to/metadata.jsonl"
custom_style_caption = "your_style_token"

# Run the script
python advanced_captioner.py
```

## 📁 Recommended Workflow

### 1. Prepare Source Material
```bash
# Extract frames from videos
python video_frame_extractor.py videos/ raw_frames/ -s 2.0 --format jpg

# Organize and rename images
python image_organizer.py raw_frames/ organized_images/ --prefix "dataset" --output_format jpg
```

### 2. Generate Captions
```bash
# For basic captioning
python basic_captioner.py

# For detailed artistic descriptions
python advanced_captioner.py
```

### 3. Dataset Structure
Your final dataset should look like:
```
dataset/
├── images/
│   ├── dataset_000001.jpg
│   ├── dataset_000002.jpg
│   └── ...
└── metadata.jsonl
```

## 🎯 Output Formats

### JSONL Format (Recommended)
```json
{"file_name": "image_001.jpg", "text": "style_token A serene landscape featuring rolling hills..."}
{"file_name": "image_002.jpg", "text": "style_token Portrait of a person with dramatic lighting..."}
```

### CSV/TSV Format
```csv
file_name,text
image_001.jpg,"style_token A serene landscape featuring rolling hills..."
image_002.jpg,"style_token Portrait of a person with dramatic lighting..."
```

## ⚙️ Configuration Options

### Model Selection
- **Basic Models**: Fast, lightweight, good for general captions
  - `microsoft/git-base-coco`: General-purpose captioning
  - `Salesforce/blip-image-captioning-base`: Alternative captioning model

- **Advanced Models**: Detailed, artistic analysis
  - `google/gemma-3-12b-it`: Comprehensive visual analysis

### Memory Optimization
The advanced captioner includes several memory optimization features:
- 4-bit quantization using BitsAndBytesConfig
- Automatic CUDA memory cleanup
- Resume capability to handle interruptions
- Batch processing with memory monitoring

### Custom Style Tokens
Add custom style tokens to all captions for fine-tuning specific artistic styles:
```python
custom_style_caption = "your_style_name"
# Results in: "your_style_name [generated caption]"
```

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce batch size or use CPU-only processing
- Enable 4-bit quantization (already included in advanced captioner)
- Process smaller batches of images

**Model Download Issues:**
- Ensure stable internet connection
- Some models may require Hugging Face authentication
- Check available disk space for model downloads

**Image Processing Errors:**
- Verify image file integrity
- Check supported formats: PNG, JPG, JPEG, BMP, GIF, TIFF, WebP
- Ensure sufficient disk space for output

### Performance Tips

1. **GPU Utilization**: Use CUDA-enabled systems for faster processing
2. **Batch Processing**: Process images in batches to optimize memory usage
3. **Resume Feature**: Use the resume capability for large datasets
4. **Storage**: Use fast SSD storage for image datasets

## 📊 Dataset Quality Guidelines

### Image Quality
- Minimum resolution: 512x512 pixels (recommended)
- Clear, well-lit images produce better captions
- Diverse content improves model training

### Caption Quality
- **Detailed Mode**: 150-200 tokens per caption
- **Concise Mode**: 50-100 tokens per caption
- Consistent style and terminology
- Accurate visual descriptions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:
- Bug reports and feature requests
- Code improvements and optimizations
- Additional model integrations
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers library
- Google Gemma models
- Microsoft GIT model
- Salesforce BLIP model
- OpenCV and Pillow libraries

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the model documentation on Hugging Face

---

*Happy dataset preparation! 🎨📊*