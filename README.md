# WordScan

WordScan is a Python-based application designed to process document photos by enhancing the text within them. It extracts text from images and provides concise summaries, making it easier to understand and analyze the content.

## Features

- **Text Enhancement**: Improves the clarity and readability of text within document photos.
- **Text Extraction**: Utilizes Optical Character Recognition (OCR) to extract text from images.
- **Summarization**: Generates concise summaries of the extracted text for quick comprehension.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/This-is-khushwant/WordScan.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd WordScan
   ```

3. **Install Dependencies**:

   Ensure you have Python installed on your system. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: The `requirements.txt` file should list all necessary dependencies. If it's not present, you may need to install packages like `pytesseract` for OCR and `sumy` for text summarization individually.*

## Usage

1. **Prepare Your Document Photo**:

   Place the image you want to process into the `uploads` directory.

2. **Run the Application**:

   Execute the main script to process the image:

   ```bash
   python WordScan.py
   ```

3. **View Results**:

   The enhanced image will be saved in the `processed` directory, and the extracted text along with its summary will be displayed in the console or saved as a text file, depending on the implementation.

## Project Structure

- `WordScan.py`: The main script that orchestrates the processing of images.
- `image_processor.py`: Handles image enhancement functions.
- `text_extractor.py`: Contains functions for OCR to extract text from images.
- `text_summarizer.py`: Provides text summarization functionalities.
- `uploads/`: Directory to store input images.
- `processed/`: Directory where processed images are saved.
- `templates/`: Contains HTML templates (if a web interface is implemented).

## Acknowledgments

Special thanks to all contributors and the open-source community for their invaluable resources and support.

