from __future__ import annotations

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


NUM_CLASSES = 2
TREE_CLASS_ID = 1
TREE_CLASS_NAME = "Tree"


def build_model(num_classes: int = NUM_CLASSES):
    

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = fasterrcnn_resnet50_fpn(
        weights=weights
)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(weights_path, device: torch.device):
    model = build_model()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
#it starts from pretrained Faster R-CNN ResNet50 FPN
#then replaces the final prediction head so the model predicts your classes instead of the default COCO classes

#build_model() = make an empty notebook with labeled sections
#.pth file = the filled-in notes
#load_state_dict(...) = write those notes back into the correct sections

#build_model() creates the empty Faster R-CNN structure
#torch.load(...) reads the saved learned numbers from the .pth file
#model.load_state_dict(...) puts those learned numbers into the correct layers