# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from PIL import Image
import pydicom
from glob import glob
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_erosion

from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename

# --- Assume model.py exists and defines UNet ---
# If model.py is not available, you'll need to define the UNet architecture here or import it.
# For this example, I'll include a minimal placeholder if model.py isn't provided.
try:
    from model import UNet
except ImportError:
    print("Warning: model.py not found. Using a placeholder UNet definition.")
    # Placeholder UNet for demonstration if model.py is missing
    class UNet(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            # This is a dummy UNet, replace with your actual architecture
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        def forward(self, x):
            return self.conv(x)


# --- CONFIG ---
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
CHECKPOINT_PATH = 'checkpoints/best.pt' # Using best.pt as per context
ALLOWED_EXTENSIONS = {'nii', 'gz', 'png', 'jpg', 'jpeg', 'dcm'} # Added dcm for DICOM folders
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_ID = 1 # Tumor class for segmentation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Ensure upload and results directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# --- Global Model Instance ---
model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Functions from io_utils.py (adapted for single file/folder loading) ---
def load_nifti(file_path):
    nii = nib.load(file_path)
    volume = nii.get_fdata()
    header = nii.header
    spacing = header.get_zooms()
    return volume, spacing

def load_dicom(folder_path):
    dicom_files = sorted(glob(os.path.join(folder_path, "*.dcm")))
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {folder_path}")
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    volume = np.stack([s.pixel_array for s in slices], axis=-1)
    px_spacing = slices[0].PixelSpacing
    slice_thickness = getattr(slices[0], "SliceThickness", 1.0)
    spacing = (float(px_spacing[0]), float(px_spacing[1]), float(slice_thickness))
    return volume, spacing

def load_png(filepath):
    img = Image.open(filepath).convert("L")
    data = np.array(img)
    volume = data[:, :, np.newaxis] # Make it 3D with depth=1
    spacing = (1.0, 1.0, 1.0) # Dummy spacing for PNG
    return volume, spacing

def load_volume_for_app(filepath_or_folder):
    if os.path.isfile(filepath_or_folder):
        if filepath_or_folder.endswith((".nii", ".nii.gz")):
            return load_nifti(filepath_or_folder)
        elif filepath_or_folder.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            return load_png(filepath_or_folder)
        elif filepath_or_folder.endswith(".dcm"): # Handle single DICOM file if needed, though usually folders
            # For simplicity, if a single DCM is uploaded, treat it as a 1-slice volume
            # In a real app, you might want to instruct users to zip DICOM folders
            ds = pydicom.dcmread(filepath_or_folder)
            volume = ds.pixel_array[:, :, np.newaxis]
            px_spacing = ds.PixelSpacing
            slice_thickness = getattr(ds, "SliceThickness", 1.0)
            spacing = (float(px_spacing[0]), float(px_spacing[1]), float(slice_thickness))
            return volume, spacing
        else:
            raise ValueError("Unsupported file type. Use .nii/.nii.gz, .png, .jpg, or a DICOM folder.")
    elif os.path.isdir(filepath_or_folder):
        return load_dicom(filepath_or_folder)
    else:
        raise FileNotFoundError(f"Path not found: {filepath_or_folder}")

# --- Functions from metrics.py (adapted for single prediction) ---
def one_hot_encode(labels, num_classes):
    return F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()

def multiclass_dice(preds, targets, epsilon=1e-6):
    num_classes = preds.shape[1]
    targets_one_hot = one_hot_encode(targets, num_classes).to(preds.device)
    preds = F.softmax(preds, dim=1)
    dice_scores = []
    for c in range(num_classes):
        pred_flat = preds[:, c].contiguous().view(-1)
        target_flat = targets_one_hot[:, c].contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + epsilon) / (pred_flat.sum() + target_flat.sum() + epsilon)
        dice_scores.append(dice)
    return torch.stack(dice_scores).mean().item()

def surface_dice(pred_mask, gt_mask, tolerance=1.0):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    pred_surface = pred_mask ^ binary_erosion(pred_mask)
    gt_surface = gt_mask ^ binary_erosion(gt_mask)
    dt_pred = distance_transform_edt(~pred_mask)
    dt_gt = distance_transform_edt(~gt_mask)
    gt_close = dt_pred[gt_surface] <= tolerance
    pred_close = dt_gt[pred_surface] <= tolerance
    nsd = (np.sum(gt_close) + np.sum(pred_close)) / (np.sum(pred_surface) + np.sum(gt_surface) + 1e-6)
    return nsd

def specificity_score(preds, targets, epsilon=1e-6):
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    TN = ((preds == 0) & (targets == 0)).sum(dim=1).float()
    FP = ((preds == 1) & (targets == 0)).sum(dim=1).float()
    specificity = (TN + epsilon) / (TN + FP + epsilon)
    return specificity.mean().item()

def compute_tumor_size(mask, spacing=(1.0, 1.0, 1.0)):
    mask = mask.astype(bool)
    pixel_count = mask.sum()
    if mask.ndim == 2:
        dx, dy = spacing
        area_mm2 = pixel_count * dx * dy
        area_cm2 = area_mm2 / 100.0
        return {
            "pixel_count": int(pixel_count),
            "area_cm2": area_cm2
        }
    elif mask.ndim == 3:
        dx, dy, dz = spacing
        volume_mm3 = pixel_count * dx * dy * dz
        volume_cm3 = volume_mm3 / 1000.0
        return {
            "voxel_count": int(pixel_count),
            "volume_cm3": volume_cm3
        }
    else:
        raise ValueError("Mask must be 2D or 3D numpy array.")

# --- Functions from viz.py (adapted for web output) ---
def window_image(img, window_center, window_width):
    img_min = window_center - (window_width / 2)
    img_max = window_center + (window_width / 2)
    windowed_img = np.clip(img, img_min, img_max)
    return (windowed_img - img_min) / (img_max - img_min)

def overlay_prediction(img_tensor, gt_mask_tensor, pred_mask_tensor, filename="overlay.png"):
    """
    Generates an overlay image of the original scan and the predicted mask.
    Saves it to the static/results folder.
    """
    # Convert tensors to numpy arrays
    img_np = img_tensor.squeeze().cpu().numpy()
    pred_mask_np = pred_mask_tensor.squeeze().cpu().numpy()

    # For 3D volumes, pick a central slice for visualization
    if img_np.ndim == 3:
        z_slice = img_np.shape[2] // 2
        img_slice = img_np[:, :, z_slice]
        pred_mask_slice = pred_mask_np[:, :, z_slice]
    elif img_np.ndim == 2:
        img_slice = img_np
        pred_mask_slice = pred_mask_np
    else:
        raise ValueError("Image tensor must be 2D or 3D.")

    # Apply windowing (e.g., for CT scans, liver preset)
    # Assuming typical CT values, adjust window_center/width if your data is different
    windowed_slice = window_image(img_slice, window_center=40, window_width=400)

    # Create plot
    plt.figure(figsize=(8, 8))
    plt.imshow(windowed_slice, cmap="gray")
    # Overlay prediction mask (only where tumor is present)
    plt.imshow(np.ma.masked_where(pred_mask_slice == 0, pred_mask_slice), cmap="Reds", alpha=0.4)
    plt.title("Original Image with Predicted Tumor Overlay")
    plt.axis("off")

    # Save the plot
    save_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close() # Close the plot to free memory
    return url_for('static', filename=f'results/{filename}')


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result_image=None, tumor_vol=None)

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Load volume and spacing
            volume_np, spacing = load_volume_for_app(filepath)

            # Preprocess for UNet:
            # 1. Normalize (e.g., to 0-1 or -1 to 1, depending on model training)
            #    Assuming simple min-max normalization for now.
            #    For CT, often windowing is applied before normalization.
            #    Let's normalize to 0-1 for simplicity, assuming model expects this.
            volume_processed = (volume_np - volume_np.min()) / (volume_np.max() - volume_np.min() + 1e-8)

            # 2. Convert to PyTorch tensor (float32)
            #    Add batch dimension (B, C, H, W) or (B, C, D, H, W)
            #    UNet typically expects 2D or 3D input. Assuming 2D for simplicity in this web app.
            #    If your UNet is 3D, you'll need to process the whole volume or select a slice.
            #    For web display, we'll process a single slice if 3D.
            
            if volume_processed.ndim == 3:
                # For 3D, pick a central slice for 2D UNet inference
                # Or, if your UNet is 3D, you'd pass the whole volume
                # For this example, let's assume a 2D UNet and process a central slice.
                # If your model is truly 3D, you'd need to adapt the input shape and model.
                input_slice_np = volume_processed[:, :, volume_processed.shape[2] // 2]
            elif volume_processed.ndim == 2:
                input_slice_np = volume_processed
            else:
                raise ValueError("Unsupported image dimensions for processing.")

            # Ensure it's (C, H, W) -> (1, H, W) for grayscale
            input_tensor = torch.from_numpy(input_slice_np).float().unsqueeze(0).unsqueeze(0) # Add channel and batch dim

            # Move to device
            input_tensor = input_tensor.to(DEVICE)

            # Inference
            global model
            if model is None:
                # Initialize and load model only once
                model = UNet(in_channels=1, out_channels=3).to(DEVICE)
                if os.path.exists(CHECKPOINT_PATH):
                    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
                else:
                    return render_template('index.html', error=f"Model checkpoint not found at {CHECKPOINT_PATH}. Please ensure it's in the correct path.")
                model.eval()

            with torch.no_grad():
                preds = model(input_tensor)
                preds_softmax = F.softmax(preds, dim=1)
                preds_argmax = preds_softmax.argmax(dim=1)

            # Convert prediction to numpy for metric/viz
            pred_mask_np = (preds_argmax == CLASS_ID).cpu().numpy().astype(np.uint8)[0] # Get binary tumor mask

            # Calculate Tumor Volume (using the full 3D volume if available)
            if volume_np.ndim == 3:
                # If the original input was 3D, we need to run 3D inference or
                # process slice by slice to get a full 3D prediction mask for volume calculation.
                # For simplicity, if we only inferred on one slice, we can't calculate 3D volume accurately.
                # Let's assume for volume calculation, we'd have a full 3D prediction.
                # If your UNet is 2D, you'd loop through slices for full volume prediction.
                # For this example, I'll calculate volume based on the *single slice* prediction
                # if the input was 2D, or just use a dummy if the model is strictly 2D and input was 3D.
                # A more robust solution would involve 3D UNet or slice-by-slice 2D inference.

                # To make it work for 3D volume calculation, let's assume a dummy 3D pred_mask_full
                # or, if the model is 2D, we'd need to run it on all slices.
                # For now, let's just use the single slice prediction for volume if 2D input,
                # or a placeholder if 3D input was only partially processed.
                
                # If the model is 2D and input is 3D, you'd need to predict slice by slice:
                # full_pred_mask = np.zeros_like(volume_np, dtype=np.uint8)
                # for i in range(volume_np.shape[2]):
                #     slice_tensor = torch.from_numpy(volume_processed[:,:,i]).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                #     slice_preds = model(slice_tensor)
                #     slice_preds_argmax = F.softmax(slice_preds, dim=1).argmax(dim=1)
                #     full_pred_mask[:,:,i] = (slice_preds_argmax == CLASS_ID).cpu().numpy().astype(np.uint8)[0]
                # tumor_size_info = compute_tumor_size(full_pred_mask, spacing)

                # For simplicity, if we only processed one slice for 3D input,
                # we can't give a true 3D volume. Let's calculate volume for the *displayed slice*.
                tumor_size_info = compute_tumor_size(pred_mask_np, (spacing[0], spacing[1])) # Use 2D spacing for 2D slice
                tumor_vol_str = f"{tumor_size_info['area_cm2']:.2f} cm² (of displayed slice)"
            else: # 2D input
                tumor_size_info = compute_tumor_size(pred_mask_np, (spacing[0], spacing[1]))
                tumor_vol_str = f"{tumor_size_info['area_cm2']:.2f} cm²"


            # Generate and save overlay image
            result_image_url = overlay_prediction(input_tensor, None, preds_argmax, filename=f"overlay_{filename}.png")

            return render_template('index.html', result_image=result_image_url, tumor_vol=tumor_vol_str)

        except Exception as e:
            return render_template('index.html', error=f"Error processing file: {e}")
    else:
        return render_template('index.html', error='Invalid file type. Allowed: .nii, .nii.gz, .png, .jpg, .jpeg, .dcm (or DICOM folder).')

if __name__ == '__main__':
    # Load model on app startup
    # This ensures the model is loaded only once when the Flask app starts
    # and is available for all subsequent requests.
    print(f"Loading model from {CHECKPOINT_PATH} to {DEVICE}...")
    try:
        model = UNet(in_channels=1, out_channels=3).to(DEVICE)
        if os.path.exists(CHECKPOINT_PATH):
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
            model.eval()
            print("Model loaded successfully.")
        else:
            print(f"Error: Model checkpoint not found at {CHECKPOINT_PATH}. Please ensure it's in the correct path.")
            model = None # Set to None to indicate failure
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None # Set to None to indicate failure

    app.run(debug=True) # debug=True for development, set to False for production