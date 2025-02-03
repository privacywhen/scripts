import os
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf2image import convert_from_path
from tqdm import tqdm
from typing import Optional, Tuple
from PIL import Image, ImageOps
from io import BytesIO

# Disable decompression bomb protection for very large images.
Image.MAX_IMAGE_PIXELS = None

# Configure logging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Change to DEBUG for more verbose output.
)

# Default configuration parameters.
DEFAULT_MAX_FILESIZE_BYTES = 10000000  # Target: 10,000,000 bytes.
DEFAULT_INITIAL_DPI = 1200             # The full resolution from pdf2image.
DEFAULT_MIN_DPI = 300                  # Lower bound to preserve legibility.
# (min_scale = DEFAULT_MIN_DPI/DEFAULT_INITIAL_DPI)

def simulate_save(
    image: Image.Image,
    candidate_dpi: int,
    initial_dpi: int = DEFAULT_INITIAL_DPI,
    output_format: str = "PNG"
) -> Tuple[Optional[Image.Image], int]:
    """
    Resize the image (using a scaling factor derived from candidate_dpi/initial_dpi),
    convert it to 8‑bit grayscale, and boost its contrast. The processed image is then
    saved to an in‑memory buffer using optimized PNG settings.
    
    Args:
        image (Image.Image): The original image.
        candidate_dpi (int): The effective DPI to use (must be between min and initial DPI).
        initial_dpi (int): The reference DPI.
        output_format (str): Output image format (default: "PNG").
    
    Returns:
        Tuple[Optional[Image.Image], int]: A tuple with the processed image and its
                                             simulated file size in bytes.
    """
    scale = candidate_dpi / initial_dpi
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    try:
        # Resize using high-quality resampling.
        resized = image.resize(new_size, Image.LANCZOS)
        # Pre-process for Transkribus: convert to grayscale and autocontrast.
        processed = resized.convert("L")
        processed = ImageOps.autocontrast(processed)
        # Save to a buffer with PNG optimizations.
        buffer = BytesIO()
        processed.save(buffer, output_format, dpi=(candidate_dpi, candidate_dpi),
                       optimize=True, compress_level=9)
        size_bytes = len(buffer.getvalue())
        buffer.close()
        return processed, size_bytes
    except Exception as e:
        logging.error(f"simulate_save error at {candidate_dpi} DPI: {e}")
        return None, float('inf')

def get_optimal_scale(
    image: Image.Image,
    output_format: str = "PNG",
    initial_dpi: int = DEFAULT_INITIAL_DPI,
    min_dpi: int = DEFAULT_MIN_DPI,
    max_filesize: int = DEFAULT_MAX_FILESIZE_BYTES,
    iterations: int = 15  # Number of binary search iterations.
) -> Tuple[int, Image.Image, int]:
    """
    Performs a binary search on the scaling factor (which directly controls the effective DPI)
    to determine the highest resolution (largest scale factor) that yields a PNG file
    size ≤ max_filesize. The effective DPI is given by: candidate_dpi = round(initial_dpi * scale).
    
    Args:
        image (Image.Image): The original image.
        output_format (str): Output format (default: "PNG").
        initial_dpi (int): The full resolution DPI.
        min_dpi (int): The minimum acceptable DPI.
        max_filesize (int): Maximum allowed file size in bytes.
        iterations (int): How many iterations to perform in the binary search.
    
    Returns:
        Tuple[int, Image.Image, int]:
            - The optimal (effective) DPI.
            - The processed image (pre‐processed and scaled).
            - The simulated file size in bytes.
    """
    # The scale factor ranges from (min_dpi/initial_dpi) to 1.
    min_scale = min_dpi / initial_dpi
    low = min_scale
    high = 1.0
    best_scale = low
    best_dpi = int(round(initial_dpi * low))
    best_image, best_size = simulate_save(image, best_dpi, initial_dpi, output_format)

    for _ in range(iterations):
        mid = (low + high) / 2
        candidate_dpi = int(round(initial_dpi * mid))
        # Clamp candidate_dpi between min_dpi and initial_dpi.
        candidate_dpi = max(min_dpi, min(candidate_dpi, initial_dpi))
        processed, size = simulate_save(image, candidate_dpi, initial_dpi, output_format)
        # Debug logging can be enabled to trace the binary search.
        logging.debug(f"Testing scale {mid:.4f} (DPI {candidate_dpi}): file size {size} bytes")
        if size <= max_filesize:
            # Candidate is valid; try to push for higher resolution.
            best_scale = mid
            best_dpi = candidate_dpi
            best_image = processed
            best_size = size
            low = mid
        else:
            high = mid

    return best_dpi, best_image, best_size

def save_image(
    image: Image.Image,
    output_filename: str,
    output_format: str = "PNG",
    initial_dpi: int = DEFAULT_INITIAL_DPI,
    min_dpi: int = DEFAULT_MIN_DPI,
    max_filesize: int = DEFAULT_MAX_FILESIZE_BYTES,
) -> None:
    """
    Process and save an image such that its final PNG file is as close as possible
    to the max_filesize (10,000,000 bytes) without going over. The image is scaled,
    converted to grayscale, and autocontrasted (optimizing it for Transkribus).
    
    Args:
        image (Image.Image): The image to be saved.
        output_filename (str): Destination file path.
        output_format (str): Output image format.
        initial_dpi (int): Full resolution DPI (from pdf2image).
        min_dpi (int): Minimum acceptable DPI.
        max_filesize (int): Maximum file size allowed in bytes.
    """
    try:
        optimal_dpi, processed_image, file_size = get_optimal_scale(
            image, output_format, initial_dpi, min_dpi, max_filesize
        )

        # (At this point, the binary search has found the highest effective DPI
        # that yields a file size ≤ max_filesize.)
        logging.info(
            f"Optimal DPI: {optimal_dpi} yields file size: {file_size} bytes "
            f"with dimensions: {processed_image.size}"
        )
        # Finally, write the processed image to disk with the chosen DPI metadata.
        with open(output_filename, 'wb') as f:
            processed_image.save(f, output_format, dpi=(optimal_dpi, optimal_dpi),
                                 optimize=True, compress_level=9)
    except Exception as e:
        logging.error(f"Error saving {output_filename}: {e}")

def split_pdf_to_images(
    pdf_path: str,
    output_folder: str,
    output_format: str = "PNG",
    num_threads: int = 4,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
    poppler_path: Optional[str] = None,
    initial_dpi: int = DEFAULT_INITIAL_DPI,
    min_dpi: int = DEFAULT_MIN_DPI,
    max_filesize: int = DEFAULT_MAX_FILESIZE_BYTES,
) -> None:
    """
    Convert each page of a PDF into an individual PNG image, ensuring that each output
    is pre‑processed (grayscale with enhanced contrast) and scaled so that its file size
    is as close as possible to the target (10,000,000 bytes) without exceeding it.
    This processing is optimized for transcription in Transkribus.
    
    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Directory to store the output images.
        output_format (str): Output image format (default: "PNG").
        num_threads (int): Number of threads for parallel processing.
        first_page (Optional[int]): First page number to convert.
        last_page (Optional[int]): Last page number to convert.
        poppler_path (Optional[str]): Path to the Poppler binaries.
        initial_dpi (int): Full resolution DPI for conversion.
        min_dpi (int): Minimum acceptable DPI.
        max_filesize (int): Maximum allowed file size in bytes.
    """
    if not os.path.isfile(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        return

    os.makedirs(output_folder, exist_ok=True)
    logging.info(f"Loading PDF: {pdf_path}")

    # Build conversion arguments.
    convert_args = {"dpi": initial_dpi}
    if poppler_path:
        convert_args["poppler_path"] = poppler_path
    if first_page is not None:
        convert_args["first_page"] = first_page
    if last_page is not None:
        convert_args["last_page"] = last_page

    try:
        images = convert_from_path(pdf_path, **convert_args)
    except Exception as e:
        logging.error(f"Failed to convert PDF: {e}")
        return

    num_pages = len(images)
    logging.info(f"Total pages loaded: {num_pages}")
    start_page = first_page if first_page is not None else 1

    # Process and save pages concurrently.
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        with tqdm(total=num_pages, desc="Processing Pages") as pbar:
            for i, image in enumerate(images, start=start_page):
                filename = os.path.join(output_folder, f"page_{i:03d}.{output_format.lower()}")
                future = executor.submit(
                    save_image, image, filename, output_format,
                    initial_dpi, min_dpi, max_filesize
                )
                futures[future] = i

            for future in as_completed(futures):
                page_number = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing page {page_number}: {e}")
                pbar.update(1)

    logging.info("PDF conversion complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a multi-page PDF to high-resolution PNG images (grayscale and contrast-enhanced) "
                    "optimized for Transkribus. Each output is scaled to be as close as possible to a target "
                    "file size (10,000,000 bytes) without exceeding it."
    )
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file")
    parser.add_argument("output_folder", type=str, help="Folder to save output images")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for processing (default: 4)")
    parser.add_argument("--first_page", type=int, help="First page to convert (optional)")
    parser.add_argument("--last_page", type=int, help="Last page to convert (optional)")
    parser.add_argument("--poppler_path", type=str, help="Path to Poppler binaries (optional)")
    parser.add_argument(
        "--initial_dpi", type=int, default=DEFAULT_INITIAL_DPI,
        help=f"Initial DPI (default: {DEFAULT_INITIAL_DPI})"
    )
    parser.add_argument(
        "--min_dpi", type=int, default=DEFAULT_MIN_DPI,
        help=f"Minimum DPI (default: {DEFAULT_MIN_DPI})"
    )
    parser.add_argument(
        "--max_filesize", type=int, default=DEFAULT_MAX_FILESIZE_BYTES,
        help=f"Max file size in bytes (default: {DEFAULT_MAX_FILESIZE_BYTES})"
    )
    
    args = parser.parse_args()

    split_pdf_to_images(
        pdf_path=args.pdf_path,
        output_folder=args.output_folder,
        output_format="PNG",
        num_threads=args.threads,
        first_page=args.first_page,
        last_page=args.last_page,
        poppler_path=args.poppler_path,
        initial_dpi=args.initial_dpi,
        min_dpi=args.min_dpi,
        max_filesize=args.max_filesize,
    )

if __name__ == "__main__":
    main()
