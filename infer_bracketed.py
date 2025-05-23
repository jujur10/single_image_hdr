import os
from argparse import ArgumentParser

import torch
from PIL import Image
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.ednet import EDNet

torch.autograd.set_grad_enabled(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)
torch.autograd.set_detect_anomaly(mode=False)


def getDisplayableImg(img):
    # [0, 255]
    img = (img * 255).clamp(min=0, max=255).round()
    # uint8
    img = img.type(torch.uint8)
    # [1, 3, H, W] -> [H, W, 3]
    img = img.squeeze(0).permute(1, 2, 0)
    # To numpy
    img = img.cpu().numpy()
    return img


def load_and_preprocess_image(image_path, device):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Pad the image to be divisible by 2^(nlayers-1), default nlayers=7 -> 2^6=64
    # This is crucial for U-Net architectures to avoid size mismatches during concat
    h, w = img.shape[:2]
    multiple = 64 # Assuming nlayers=7 as per EDNet defaults (n1_layers, n23_layers)
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    # Albumentations expects padding in (top, bottom, left, right)
    # For simplicity, we can use cv2.copyMakeBorder or A.PadIfNeeded
    # A.PadIfNeeded is more robust for albumentations pipeline
    # We need to calculate padding for top, bottom, left, right
    # To distribute padding somewhat evenly:
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    transform = A.Compose([
        A.PadIfNeeded(min_height=h + pad_h, min_width=w + pad_w, border_mode=cv2.BORDER_CONSTANT, value=0, position='top_left'),
        A.ToFloat(max_value=255.0),
        ToTensorV2() # HWC -> CHW, already a tensor
    ])
    # Store original dimensions to crop back later if needed, though for this task, output can remain padded.
    # For now, we don't crop back as the model output will be of the padded size.

    img_tensor = transform(image=img)['image']
    return img_tensor.unsqueeze(0).to(device) # Add batch dimension and send to device


def get_well_exposed_mask(img_tensor, clip_threshold, device):
    # Ensure img_tensor is on the correct device for calculations with rgb_to_y
    img_tensor_device = img_tensor.to(device)
    
    # Assuming img_tensor is [B, C, H, W] and in [0,1] range
    rgb_to_y = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, -1, 1, 1)
    
    Y = (img_tensor_device * rgb_to_y).sum(dim=1, keepdim=True) # Y is [B, 1, H, W]
    
    # Clamp Y to avoid issues if it's outside [0,1] due to minor float precision things
    Y = Y.clamp(min=0.0, max=1.0)

    # Calculate under and over exposed areas based on the threshold
    # tau is clip_threshold
    # Mask should be 1 for well-exposed, 0 for under/over exposed.
    # Original logic: under_exposed_mask = (1 - tau - Y).clip(min=0.0) / (1 - tau)
    #                 over_exposed_mask = (Y - tau).clip(min=0.0) / (1 - tau)
    #                 final_mask = 1 - torch.max(under_exposed_mask, over_exposed_mask)
    # Simplified: mark regions as bad if Y < (1-tau) (too dark) or Y > tau (too bright)
    # The original formulation seems to create a soft mask. Let's stick to it.

    # Check for edge case where clip_threshold is 1.0 or 0.0 to avoid division by zero
    # (1-tau) can be zero if tau is 1.0.
    # (Y - tau) can be problematic if tau is 0 and Y is 0.
    # The paper states tau is typically 0.95. Division by (1-tau) means division by 0.05.
    # If tau is 1, (1-tau) is 0. Denominator is (1-tau).
    # If tau is 0, (1-tau) is 1. Denominator is (1-tau). Okay.

    # The original formula for under_exposed_mask and over_exposed_mask is a bit confusing
    # regarding what represents fully saturated vs. fully dark in the mask values themselves.
    # Let's re-derive from the paper or assume the original code's intention is correct.
    # _get_well_exposed_masks_ from dataset.py:
    #   under_exposed_mask = (1 - self.tau - Y).clip(min=0.0) / (1 - self.tau)
    #   over_exposed_mask = (Y - self.tau).clip(min=0.0) / (1 - self.tau)
    #   return 1 - torch.max(under_exposed_mask, over_exposed_mask)
    # This produces values between 0 and 1, where 1 is well-exposed.

    # Denominators could be zero if clip_threshold is 1.0.
    # Let's add a small epsilon to denominators to prevent division by zero if clip_threshold is exactly 1.0 or 0.0,
    # though 0.95 is typical.
    denominator = (1.0 - clip_threshold) + 1e-6 

    under_exposed_regions = ((1.0 - clip_threshold) - Y).clamp(min=0.0) / denominator
    over_exposed_regions = (Y - clip_threshold).clamp(min=0.0) / denominator
    
    # The mask indicates badly exposed regions (0 for good, 1 for bad, or vice-versa depending on interpretation)
    # The original return is 1.0 - torch.max(under_exposed_mask, over_exposed_mask)
    # So, if either under_exposed_regions or over_exposed_regions is high, torch.max will be high, and (1-max) will be low.
    # This means the returned mask has: 1 for well-exposed, 0 for badly-exposed.
    mask = 1.0 - torch.max(under_exposed_regions, over_exposed_regions)
    
    return mask.to(device) # Ensure final mask is on the target device.


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--out_dir", required=True, help="Directory to save the output images.")
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--in_image", required=True, help="Path to the single input LDR image.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for inference.")
    parser.add_argument("--ev_steps", type=str, default="-2,0,2", help="Comma-separated string of EV steps to generate (e.g., '-2,0,2').")
    parser.add_argument("--clip_threshold", type=float, default=0.95, help="Clip threshold for well-exposed mask generation.")

    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device("cuda")

    model = EDNet.load_from_checkpoint(args.ckpt).to(device=device)
    model.eval()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    # Parse EV steps
    ev_values = [float(ev.strip()) for ev in args.ev_steps.split(',')]

    # Get base filename for output naming
    base_filename, ext = os.path.splitext(os.path.basename(args.in_image))

    input_img_tensor = load_and_preprocess_image(args.in_image, device)
    input_mask_tensor = get_well_exposed_mask(input_img_tensor, args.clip_threshold, device)

    print(f"Processing input image: {args.in_image}")
    print(f"Generating EV steps: {ev_values}")
    print(f"Output directory: {args.out_dir}")

    for ev in tqdm(ev_values, desc="Generating Exposures"):
        output_filename = os.path.join(args.out_dir, f"{base_filename}_EV{ev:+.1f}.png") # Using .1f for EV like +2.0

        if abs(ev) < 1e-5: # Check for EV = 0 with a small tolerance
            print(f"Saving EV 0.0 (processed input) to {output_filename}")
            # Save the processed input image (tensor converted back to displayable format)
            displayable_input_img = getDisplayableImg(input_img_tensor.clone()) # Clone to avoid in-place modifications if any
            Image.fromarray(displayable_input_img).save(output_filename)
        else:
            print(f"Generating EV {ev:+.1f} and saving to {output_filename}")
            exposure_ratio = 2.0 ** ev
            is_up_exposed = ev > 0
            # Model expects exposure ratio as a tensor [B, 1, 1, 1]
            exp_tensor = torch.tensor(exposure_ratio, device=device, dtype=input_img_tensor.dtype).view(1, 1, 1, 1)
            
            pred_tensor, _ = model(input_img_tensor, input_mask_tensor, exp_tensor, isUpExposed=is_up_exposed)
            
            final_image_data = getDisplayableImg(pred_tensor)
            Image.fromarray(final_image_data).save(output_filename)

    print("Processing complete.")
