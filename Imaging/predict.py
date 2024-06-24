'''Predict the segmentation'''
import torch
from monai.inferers import sliding_window_inference


# Define inference method
def inference(input, model, device, amp=True):
    """
    Infer the segmentation on nifti image.
    
    Paramters
    ---------
    input
        Input image .
    model
        Model on which segmentation inference.
    amp
        Automatically infer precission for better gpu performance.

    Return 
    ------
    Infered image.


    """
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if amp:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)
