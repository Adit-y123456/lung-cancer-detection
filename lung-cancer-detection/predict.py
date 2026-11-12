import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from explain import GradCAM, overlay_heatmap
from models import build_model


def load_checkpoint(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    architecture = saved['architecture']
    classes = saved['class_names']
    model = build_model(architecture, len(classes), pretrained=False)
    model.load_state_dict(saved['state_dict'])
    model = model.to(device).eval()
    return model, classes, architecture, device


def classify_image(image, model, classes, architecture, device):
    image = image.convert('RGB')
    tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        class_index = probabilities.argmax().item()
    heatmap = GradCAM(model, architecture).heatmap(model, tensor, class_index)
    overlay = overlay_heatmap(image, heatmap)
    return {
        'label': classes[class_index],
        'confidence': float(probabilities[class_index].item()),
        'probabilities': {name: float(probabilities[i].item()) for i, name in enumerate(classes)},
        'overlay': overlay
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/cnn.pt')
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', default='results/heatmap.png')
    args = parser.parse_args()
    model, classes, architecture, device = load_checkpoint(args.checkpoint)
    with Image.open(args.image) as image:
        prediction = classify_image(image, model, classes, architecture, device)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prediction['overlay'].save(destination)
    print(f"Prediction: {prediction['label']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f'Heatmap: {destination}')


if __name__ == '__main__':
    main()
