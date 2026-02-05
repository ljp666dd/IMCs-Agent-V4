import fitz  # PyMuPDF
import os
import sys
import io
from PIL import Image

def extract_figures_from_pdf(pdf_path, output_dir, min_width=200, min_height=200):
    """
    Extracts images from a PDF file and saves them to the output directory.
    Filters out small images (logos, icons) based on min_width/height.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return []

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    save_dir = os.path.join(output_dir, pdf_name)
    os.makedirs(save_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    extracted_images = []

    print(f"Processing {pdf_path} ({len(doc)} pages)...")

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        
        if not image_list:
            continue
            
        print(f"  Page {page_index + 1}: Found {len(image_list)} images.")
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            
            try:
                # Load with Pillow to check dimensions/validity
                image = Image.open(io.BytesIO(image_bytes))
                width, height = image.size
                
                if width < min_width or height < min_height:
                    continue # Skip small images
                
                filename = f"p{page_index + 1}_img{img_index}.{ext}"
                filepath = os.path.join(save_dir, filename)
                
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                
                extracted_images.append(filepath)
                print(f"    Saved: {filename} ({width}x{height})")
                
            except Exception as e:
                print(f"    Error processing image {xref}: {e}")

    print(f"Done. Extracted {len(extracted_images)} figures to {save_dir}")
    return extracted_images

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default test path
        pdf = "data/literature/cache/10.1038_srep29700.pdf" 
        out = "data/literature/extracted_figures"
    else:
        pdf = sys.argv[1]
        out = "data/literature/extracted_figures"
        
    extract_figures_from_pdf(pdf, out)
