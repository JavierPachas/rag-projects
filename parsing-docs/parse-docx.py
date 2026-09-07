
import zipfile
import os
from docx import Document
from IPython.display import display, Image

document = Document("sample_data/sample_data.docx")

# Extract text
for para in document.paragraphs:
    print(para.text)

# Extract tables
for table in document.tables:
    print("\n-- Table --")
    for row in table.rows:
        row_text = [cell.text for cell in row.cells]
        print(row_text)

zipf = zipfile.ZipFile("sample_data/sample_data.docx")
filelist = zipf.namelist()

for fname in filelist:
    _, ext = os.path.splitext(fname)
    if ext in [".jpg", ".jpeg", ".png",".gif"]:
        # read image and display it
        with zipf.open(fname) as image_file:
            image_data = image_file.read()
            # in Jupyter:
            #display(Image(data=image_data)) 
            out_name = f"extracted_image_{os.path.basename(fname)}"
            with open(out_name, "wb") as f:
                f.write(image_data)
            print(f"Extracted image saved as: {out_name}")