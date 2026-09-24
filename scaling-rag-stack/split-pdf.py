import os
import requests
import io
from urllib.parse import urlparse
from PyPDF2 import PdfReader, PdfWriter

def get_pdf_reader(input_source):
    """
    Get a PdfReader object from a local file path or a URL.

    Args:
        input_source (str): The local file path or URL of the PDF.

    Returns:
        tuple: (PdfReader, base filename without extension, total page count).
    """
    if os.path.isfile(input_source):
        # If it's a local file, read it into memory so the reader outlives the file handle
        base_filename = os.path.splitext(os.path.basename(input_source))[0]
        with open(input_source, 'rb') as f:
            reader = PdfReader(io.BytesIO(f.read()))
        return reader, base_filename, len(reader.pages)
    else:
        # If it's a URL, download the PDF and read it
        base_filename = "output"
        response = requests.get(input_source, stream=True, timeout=30)
        response.raise_for_status()  # Raise an error for bad responses

        # Get filename from URL path
        parsed_url = urlparse(input_source)
        path_part = os.path.basename(parsed_url.path)
        if path_part and '.' in path_part:
            base_filename = os.path.splitext(path_part)[0]

        # Read content into memory
        pdf_content = io.BytesIO(response.content)
        reader = PdfReader(pdf_content)
        total_pages = len(reader.pages)
        return reader, base_filename, total_pages


def split_pdf(input_source, output_folder="split_output", pages_per_split=50):
    """
    Split a PDF (local path or URL) into smaller PDFs of up to pages_per_split pages.

    Args:
        input_source (str): The local file path or URL of the PDF.
        output_folder (str): Folder where the split PDFs are written.
        pages_per_split (int): Maximum number of pages per output file.

    Returns:
        list[str]: Paths of the written PDF files.
    """
    reader, base_filename, total_pages = get_pdf_reader(input_source)
    os.makedirs(output_folder, exist_ok=True)

    output_paths = []
    for start in range(0, total_pages, pages_per_split):
        end = min(start + pages_per_split, total_pages)
        writer = PdfWriter()
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])

        output_path = os.path.join(
            output_folder, f"{base_filename}_pages_{start + 1}-{end}.pdf"
        )
        with open(output_path, 'wb') as out_file:
            writer.write(out_file)
        output_paths.append(output_path)
        print(f"Wrote {output_path}")

    print(f"Split {total_pages} pages into {len(output_paths)} files in '{output_folder}'")
    return output_paths


if __name__ == "__main__":
    split_pdf(
        "https://web.stanford.edu/class/psych209/Readings/"
        "SuttonBartoIPRLBook2ndEd.pdf",
        output_folder="output-folder-name", pages_per_split=50
    )
