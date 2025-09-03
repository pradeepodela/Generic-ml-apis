from enum import Enum
from pathlib import Path
from pydantic import BaseModel
import base64
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
import json 
from mistralai import Mistral
from mistralai.models import OCRResponse
from IPython.display import Markdown, display
import os
from dotenv import load_dotenv

load_dotenv()
client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))


def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """
    Replace image placeholders in markdown with base64-encoded images.

    Args:
        markdown_str: Markdown text containing image placeholders
        images_dict: Dictionary mapping image IDs to base64 strings

    Returns:
        Markdown text with images replaced by base64 data
    """
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """
    Combine OCR text and images into a single markdown document.

    Args:
        ocr_response: Response from OCR processing containing text and images

    Returns:
        Combined markdown string with embedded images
    """
    markdowns: list[str] = []
    # Extract images from page
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        # Replace image placeholders with actual images
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))

    return "\n\n".join(markdowns)

class StructuredOCR(BaseModel):
    file_name: str
    topics: list[str]
    languages: str
    ocr_contents: dict

def structured_ocr(URL: str) -> StructuredOCR:
    """
    Process an image using OCR and extract structured data.

    Args:
        image_path: Path to the image file to process

    Returns:
        StructuredOCR object containing the extracted data

    Raises:
        AssertionError: If the image file does not exist
    """

    # Process the image using OCR
    image_response = client.ocr.process(
        document=ImageURLChunk(image_url=URL),
        model="mistral-ocr-latest"
    )
    image_ocr_markdown = image_response.pages[0].markdown

    # Parse the OCR result into a structured JSON response
    chat_response = client.chat.parse(
        model="pixtral-12b-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    ImageURLChunk(image_url=URL),
                    TextChunk(text=(
                        f"This is the image's OCR in markdown:\n{image_ocr_markdown}\n.\n"
                        "Convert this into a structured JSON response "
                        "with the OCR contents in a sensible dictionnary."
                        )
                    )
                ]
            }
        ],
        response_format=StructuredOCR,
        temperature=0
    )
    res = chat_response.choices[0].message.parsed

    result =  res.model_dump_json(indent=4)
    return result
def ocr_docu(URL, TYPE):
    print(f'OCR Worker called with URL: {URL} and TYPE: {TYPE}')
    try:
        if TYPE == 'PDF':
            pdf_response = client.ocr.process(
                document=DocumentURLChunk(document_url=URL),
                model="mistral-ocr-latest",
                include_image_base64=True
            )

            # Get combined markdown directly from the OCRResponse object
            combined_markdown = get_combined_markdown(pdf_response)
            print(combined_markdown)
            return json.dumps({"markdown": combined_markdown}, indent=4)

        if TYPE == 'IMAGE':
            # Here's the fix: use image_url instead of document_url
            image_response = client.ocr.process(
                document=ImageURLChunk(image_url=URL),  # Changed from document_url to image_url
                model="mistral-ocr-latest",
                include_image_base64=True
            )

            # Get combined markdown directly from the OCRResponse object
            combined_markdown = get_combined_markdown(image_response)
            print(combined_markdown)
            return json.dumps({"markdown": combined_markdown}, indent=4)

    except Exception as e:
        print(f"Error in OCR processing: {e}")
        import traceback
        print(traceback.format_exc())
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Example usage
    url = "https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/receipt.png"  # Replace with your image URL
    # Structured OCR
    structured_data = structured_ocr(url)
    print(structured_data)
    # OCR Document
    # res = ocr_docu('https://www.zoho.com/inventory/features/ebook/zoho-inventory-features.pdf', "PDF")
    print('--'*20)
    # print(res)