import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor


class FaceFinder:
    """This class wraps the capabilities of face detection models to detect and extract faces from images
    Parameters
    ----------
    model: nn.Module
        A pytorch module with a `detect` method that returns a list of bounding boxes for detected face along with a
        list of probabilities for each bounding box (the latter is ignored here). All models from facenet_pytorch
        include such a method.
    """

    def __init__(self, model):
        self.face_model = model

    def has_single_face(self, img):
        """
        Parameters
        ----------
        img: Image
        Returns
        -------
        bool
        """
        boxes, _ = self.face_model.detect(img)
        return (boxes is not None) and (len(boxes) == 1)

    def __call__(self, img, return_boxes=False):
        """Extract the image region containing the single face
        Parameters
        ----------
        img: Image
        return_boxes: bool, optional
            If True, then a the bounding box of each face is returned as a numpy array. The format of the bounding boxes
            are (x_min, y_min, x_max, y_max). If a single face is detected, then the array will have four elements.
            If multiple faces are detected then the array will have shape (n, 4). Default is False.
        Returns
        -------
        Image or list[Image]
            If there is one face, then a single Image is returned. If multiple faces are found, then a list of Image
            objects are returned. If return_boxes is True, then the bounding boxes of the faces are returned as well.
        """
        boxes, _ = self.face_model.detect(img)
        if boxes is not None:
            if len(boxes) == 1:
                face_img = img.crop(boxes[0])
                if return_boxes:
                    return face_img, boxes[0]
                return face_img

            faces = [img.crop(box) for box in boxes]
            if return_boxes:
                return faces, boxes
            return faces
        return None


class FacePredictor:
    """Predict the emotion of a detected face
    Parameters
    ----------
    model: nn.Module
        An pytorch model that classifies the emotion expressed in an image of a face
    emotions: list[str]
        A list of the emotions which is aligned with the output vector of the model
    img_size: tuple[int, int], optional
        The expected (width, height) dimensions for the model input.
        All images will be resized to this dimension (default=(224, 224))
    device: torch.device, optional
        The device to run the model on. Will run on the CPU if None (default=None)
    """

    def __init__(self, model, emotions, img_size=(224, 224), device=None):
        self.emotions = emotions
        self.transform = Compose([Resize(img_size), Grayscale(1), ToTensor()])
        self.device = torch.device('cpu') if device is None else device
        self.model = model.eval().to(self.device)

    def __call__(self, face):
        """Classify the expressed emotion
        Parameters
        ----------
        face: PIL.Image.Image
            An image of a face (should be zoomed in to just the face).
        Returns
        -------
        str, float
            Returns the predicted emotion along with the model's confidence
        """
        x = self.transform(face).to(self.device)
        with torch.no_grad():
            out = self.model(x.unsqueeze(0))
            probs = F.softmax(out, dim=1).squeeze()
            idx = torch.argmax(probs).item()
            return self.emotions[idx], probs[idx].item()


# ----------------------------------------------------------------------------------------------------------------------
#                               Utilities for initializing, loading, and saving models
# ----------------------------------------------------------------------------------------------------------------------

# using a pretrained (on imagenet) gray scale model improves our speed (reduced number of computations)
def init_grayscale_resnet(init_weights=None):
    """Initialize a pretrained Resnet-50 model and change the first layer to be a one-channel 2D convolution
    Returns
    -------
    gray_model: nn.Module
    """
    gray_model = resnet50(pretrained=True)
    if init_weights is None:
        w = torch.zeros((64, 1, 7, 7))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    else:
        w = init_weights
    gray_model.conv1.weight.data = w
    return gray_model


def load_model(model_state, num_emotions=5):
    """Redefine the model architecture and load the parameter state from a saved model
    Parameters
    ----------
    model_state: str
        The path to the saved model state dictionary
    num_emotions: int, optional
        The number of emotions to classify (default=5)
    Returns
    -------
    model: nn.Module
    """
    model = init_grayscale_resnet()
    conv_out_features = model.fc.in_features
    model.fc = nn.Linear(conv_out_features, num_emotions)
    model.load_state_dict(torch.load(model_state))
    return model
