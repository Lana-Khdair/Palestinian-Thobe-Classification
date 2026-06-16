import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn

IMG_SIZE = 224
NUM_CLASSES = 3
CLASSES = ['nablus', 'bethlehem', 'jaffa']
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_head():
    return nn.Sequential(
        nn.BatchNorm1d(1280), nn.Linear(1280, 256), nn.ReLU(inplace=True),
        nn.Dropout(0.5), nn.Linear(256, 128), nn.ReLU(inplace=True),
        nn.Dropout(0.4), nn.Linear(128, NUM_CLASSES),
    )

eff = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
eff.classifier = build_head()
eff.load_state_dict(torch.load('thobe_efficientnet_best.pt', map_location=DEVICE))
eff = eff.to(DEVICE)

mob = mobilenet_v2(weights=None)
mob.classifier = build_head()
mob.load_state_dict(torch.load('thobe_mobilenet_best.pt', map_location=DEVICE))
mob = mob.to(DEVICE)

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.gradients = None
        self.activations = None
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0]))
        layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o))

    def generate(self, x):
        self.model.eval()
        out = self.model(x)
        cls = out.argmax(1).item()
        self.model.zero_grad()
        out[:, cls].backward()
        w = self.gradients[0].mean(dim=[1,2])
        cam = (self.activations[0] * w[:,None,None]).mean(0).cpu().detach().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() != 0: cam /= cam.max()
        return cam, cls

def show_gradcam(model, layer, image_path, save_name):
    img = Image.open(image_path).convert('RGB')
    x = tf(img).unsqueeze(0).to(DEVICE)
    cam_gen = GradCAM(model, layer)
    heatmap, pred = cam_gen.generate(x)
    img_cv = cv2.resize(np.array(img), (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))), cv2.COLORMAP_JET)
    overlay = (heatmap * 0.4 + img_cv).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, im, t in zip(axes, [img_cv, heatmap, overlay],
                         ['Original', 'Grad-CAM Heatmap', f'Focus Area\nPrediction: {CLASSES[pred]}']):
        ax.imshow(im); ax.set_title(t); ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.close()
    print(f'Saved: {save_name}  →  predicted: {CLASSES[pred]}')

IMAGE = 'data_split_cropped/train/bethlehem/1778440923378.png'

show_gradcam(eff, eff.features[-1], IMAGE, 'gradcam_eff.png')
show_gradcam(mob, mob.features[-1], IMAGE, 'gradcam_mob.png')
