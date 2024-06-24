'''Main file for prediction'''
import os
from dotenv import load_dotenv
import torch
import matplotlib.pyplot as plt

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    MapTransform
)

from model_initialization import get_model
from predict import  inference


load_dotenv()

directory = os.getenv("ASSETS_DIR")
file_path = os.path.join(directory, "BRATS_738.nii.gz")
model_path = os.path.join(directory,"best_metric_model_1.pth")

# Define validation transformations
def get_val_transform():
    """
    Trnsformation for pre-processing image.

    Parameters
    ----------
    None

    Return
    ------
    Transformation apply on image.

    """
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear",),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

def save_results_as_png(image, output, output_dir="output"):
    """
    Save the input and output of channels.

    Parameters
    ----------
    image
        Original channels enter as an input
    output
        Output of the input channels
    output_dir
        Directory to store  input and output channels.

    Returns
        None 

    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save image channels
    for i in range(4):
        plt.figure()
        plt.imshow(image[i, :, :, 70].detach().cpu(), cmap="gray")
        plt.title(f"image channel {i}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"image_channel_{i}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    # Save output channels
    for i in range(3):
        plt.figure()
        plt.imshow(output[i, :, :, 70].detach().cpu())
        plt.title(f"output channel {i}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"output_channel_{i}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
# Load and preprocess the image
def load_and_preprocess_image(file_name):
    """
    Load and Preprcess the image for inference.

    Parameters
    ----------
    file_name
        Name of the needs to infer.
    transformation
        Transformation of the image. 
    """
    data = {"image": file_name}
    transform = get_val_transform()
    data = transform(data)
    return data["image"].unsqueeze(0)

if __name__=="__main__":
    device = torch.device("cuda:0")
    inp_image = load_and_preprocess_image(file_name=file_path).to(device)
    model = get_model(device = device, model_path=model_path)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    with torch.no_grad():
        val_output = inference(inp_image, model, device)
        val_output = post_trans(val_output[0])
        save_results_as_png(inp_image[0], val_output)  # Store results in output directory

