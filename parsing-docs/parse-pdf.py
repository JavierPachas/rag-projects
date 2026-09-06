"""
Extract text from PDFs at page and block level using PyMuPDF
Parse and extract table contents from PDF pages
Separate unstructured text from tabular data
Extract embedded images from PDF documents
"""

import json
import pymupdf
import pandas as pd
from IPython.display import Image, display
import io

# Extract all text on one page
with pymupdf.open("sample_data/sample_data.pdf") as doc:
    for page in doc:
        text: str = page.get_text()
#        print(text, end= "---")

# Extract text in blocks
with pymupdf.open("sample_data/sample_data.pdf") as doc:
  for page in doc:
    text: str = page.get_text("blocks")
    for block in text:
       pass
#      print(block[4], end="---\n")

# Extract table contents
with pymupdf.open("sample_data/sample_data.pdf") as doc:
  for page in doc: 
    tables = page.find_tables() # locate and extract any tables on page
    for i, table in enumerate(tables):
       pass
      #print (f"Table {i+1}")
      #print(table.extract(), end="\n\n")

# Extract non-table text
def extract_unstructured_text(pdf_path):
  """
  Extract unstructured text from PDF, excluding table content.
  Returns clean paragraph text without tabular data.
  """
  unstructured_text = []
  
  with pymupdf.open(pdf_path) as doc:
    for page in doc:
      # Get all text from the page
      page_text = page.get_text()
      
      # Find tables on the page to exclude their content
      tables = page.find_tables()
      
      # Get table text blocks to filter out
      table_text_blocks = []
      if tables.tables:
        for table in tables:
          # Get table bounding box
          bbox = table.bbox
          # Extract text within table bounds
          table_text = page.get_text(clip=bbox)
          table_text_blocks.append(table_text.strip())
      
      # Split page text into lines and filter out table content
      lines = page_text.split('\n')
      filtered_lines = []
      
      for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
          # Check if this line is part of any table
          is_table_content = False
          for table_text in table_text_blocks:
            if line in table_text:
              is_table_content = True
              break
          
          # Only include non-table content
          if not is_table_content:
            filtered_lines.append(line)
      
      # Join filtered lines back into paragraphs
      page_unstructured = '\n'.join(filtered_lines)
      if page_unstructured.strip():
        unstructured_text.append(page_unstructured)
  
  return '\n\n'.join(unstructured_text)

# Test the function
#unstructured_content = extract_unstructured_text("sample_data/sample_data.pdf")
#print("Unstructured text extracted:")
#print(unstructured_content)

# Extract images 
with pymupdf.open("sample_data/sample_data.pdf") as doc:
  for page in doc:
    # Get images from the page
    image_list = page.get_images()
    for image_index, img in enumerate(image_list):
      # Extract image data
      xref = img[0]
      base_image = doc.extract_image(xref)
      image_bytes = base_image["image"]
      # Display the image directly in Jupyter Notebook
      display(Image(data=image_bytes))

      # save image 
      image_format = base_image["ext"]
      with open(f"extracted_image_{page.number}_{image_index}.{image_format}", "wb") as img_file:
          img_file.write(image_bytes)