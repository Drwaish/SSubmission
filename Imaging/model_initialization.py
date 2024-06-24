'''Initialize the model for segmentation'''
from monai.transforms import  MapTransform
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
import torch




# Define custom transform for converting labels
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d




# Create and load the model
def get_model(device, model_path):
    """
    Innitialize the model for segmentation.

    Paramters
    ---------
    device
        Specify working on CPU or GPU
    model_path
        Path of the model where present.

    Return
    ------
    Model
    """
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

