import cv2
import numpy as np
import argparse
import os

def merge_ldr_to_hdr(ldr_image_paths, output_path, exposure_times):
    """
    Merges multiple LDR images into a single HDR image.

    Args:
        ldr_image_paths (list of str): List of paths to the LDR images.
        output_path (str): Path to save the merged HDR image.
        exposure_times (list of float): List of exposure times for each LDR image.
    """
    if len(ldr_image_paths) != len(exposure_times):
        raise ValueError("The number of LDR images and exposure times must be the same.")

    # Read LDR images
    images = []
    for img_path in ldr_image_paths:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        images.append(img)

    # Convert exposure times to a numpy array
    exposure_times_np = np.array(exposure_times, dtype=np.float32)

    # Merge LDR images into HDR
    print("Merging LDR images into HDR...")
    merge_debevec = cv2.createMergeDebevec()
    hdr_image = merge_debevec.process(images, times=exposure_times_np.copy()) # Add .copy() for C-contiguous array

    # Save the HDR image
    print(f"Saving HDR image to: {output_path}")
    if not cv2.imwrite(output_path, hdr_image):
        raise IOError(f"Could not save HDR image to {output_path}. Check file format and permissions.")
    print("HDR image saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple LDR images into a single HDR image.")
    parser.add_argument(
        "ldr_images",
        type=str,
        nargs=3,
        help="Paths to the three LDR input images (e.g., img1.jpg img2.jpg img3.jpg)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="merged_output.hdr",
        help="Path to save the merged HDR image (e.g., output.hdr or output.exr). Default: merged_output.hdr"
    )
    parser.add_argument(
        "--exposures",
        type=float,
        nargs=3,
        default=[1.0/16.0, 1.0/4.0, 1.0], # Example exposure times
        help="Exposure times for the LDR images (e.g., 0.0625 0.25 1.0). Default: 1/16, 1/4, 1"
    )

    args = parser.parse_args()

    # Basic validation for output file extension
    _, output_ext = os.path.splitext(args.output_path)
    if output_ext.lower() not in ['.hdr', '.exr']:
        print(f"Warning: Output file extension '{output_ext}' may not be ideal for HDR. Consider '.hdr' or '.exr'.")


    try:
        merge_ldr_to_hdr(args.ldr_images, args.output_path, args.exposures)
    except Exception as e:
        print(f"An error occurred: {e}") 