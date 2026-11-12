import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import colormaps

from models import explanation_layer


class GradCAM:
    def __init__(self, model, architecture):
        self.activations = None
        self.gradients = None
        self.architecture = architecture
        layer = explanation_layer(model, architecture)
        self.forward_hook = layer.register_forward_hook(self.save_activations)
        self.backward_hook = layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def heatmap(self, model, image, class_index):
        try:
            model.zero_grad(set_to_none=True)
            scores = model(image)
            scores[0, class_index].backward()
            activations = self.activations
            gradients = self.gradients
            if self.architecture == 'vit':
                tokens = activations.shape[1] - 1
                side = math.isqrt(tokens)
                if side * side != tokens:
                    raise ValueError('ViT patch tokens do not form a square')
                activations = activations[:, 1:].transpose(1, 2).reshape(1, -1, side, side)
                gradients = gradients[:, 1:].transpose(1, 2).reshape(1, -1, side, side)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            activation_map = (weights * activations).sum(dim=1, keepdim=True).relu()
            activation_map = F.interpolate(
                activation_map, size=image.shape[-2:], mode='bilinear', align_corners=False
            )
            activation_map = activation_map[0, 0]
            activation_map -= activation_map.min()
            activation_map /= activation_map.max().clamp(min=1e-8)
            return activation_map.cpu().numpy()
        finally:
            self.close()


def overlay_heatmap(image, heatmap):
    original = image.convert('RGB')
    intensity = Image.fromarray((heatmap * 255).astype('uint8')).resize(
        original.size, Image.Resampling.BILINEAR
    )
    array = np.asarray(intensity, dtype=np.float32) / 255.0
    heat = (colormaps['jet'](array)[..., :3] * 255).astype(np.uint8)
    colors = Image.fromarray(heat, mode='RGB')
    return Image.blend(original, colors, 0.45)
