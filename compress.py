"""
used for compressing manga pdfs lol
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfWriter
from multiprocessing import Pool, cpu_count


def compress_pdf(input_path, output_path, image_quality=40):
    print(f"Compressing: {input_path}")
    writer = PdfWriter(clone_from=input_path)

    for page in writer.pages:
        for img in page.images:
            img.replace(img.image, quality=image_quality)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        writer.write(f)
    print(f"Compressed: {input_path} -> {output_path}")


def process_pdf(pdf_file, root, input_base, output_base):
    rel_path = os.path.relpath(root, input_base)
    input_path = os.path.join(root, pdf_file)
    output_path = os.path.join(output_base, rel_path, f"{pdf_file}")
    compress_pdf(input_path, output_path)


def process_directory(input_base, output_base):
    tasks = []
    for root, _, files in os.walk(input_base):
        for file in files:
            if file.lower().endswith(".pdf"):
                tasks.append((file, root, input_base, output_base))
    return tasks


def main():
    input_base_folder = "/Users/chilly/Documents/Mangas/"
    output_base_folder = "/Users/chilly/Documents/Compressed"

    os.makedirs(output_base_folder, exist_ok=True)
    all_tasks = process_directory(input_base_folder, output_base_folder)
    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_pdf, all_tasks)

    print("All PDFs have been compressed")


if __name__ == "__main__":
    main()
