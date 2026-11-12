MODEL_CHOICES = ('cnn',)
EXPECTED_CLASSES = ['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma']

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(root, batch_size, image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    datasets_by_split = {
        'train': datasets.ImageFolder(root / 'train', transform=train_transform),
        'val': datasets.ImageFolder(root / 'val', transform=eval_transform),
        'test': datasets.ImageFolder(root / 'test', transform=eval_transform)
    }
    classes = datasets_by_split['train'].classes
    if any(item.classes != classes for item in datasets_by_split.values()):
        raise ValueError('Class folders must be identical in train, val, and test')
    loaders = {
        split: DataLoader(item, batch_size=batch_size, shuffle=split == 'train', num_workers=0)
        for split, item in datasets_by_split.items()
    }
    return loaders, classes


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(images)
        correct += (logits.argmax(1) == targets).sum().item()
        count += len(images)
    if count == 0:
        raise ValueError('Dataset split is empty')
    return {'loss': total_loss / count, 'accuracy': correct / count}


def test_model(model, loader, device, names):
    model.eval()
    actual = []
    predicted = []
    with torch.no_grad():
        for images, labels in loader:
            scores = model(images.to(device))
            actual.extend(labels.tolist())
            predicted.extend(scores.argmax(1).cpu().tolist())
    ids = list(range(len(names)))
    return {
        'accuracy': float(accuracy_score(actual, predicted)),
        'classification_report': classification_report(
            actual, predicted, labels=ids, target_names=names, output_dict=True, zero_division=0
        ),
        'confusion_matrix': confusion_matrix(actual, predicted, labels=ids).tolist(),
        'class_names': names,
        'test_images': len(actual)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODEL_CHOICES, default='cnn')
    parser.add_argument('--data', default='data')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--freeze-backbone', action='store_true')
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error('epochs, batch-size, and lr must be positive')
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data)
    loaders, classes = make_loaders(data_root, args.batch_size, 224)
    if EXPECTED_CLASSES is not None and classes != EXPECTED_CLASSES:
        raise ValueError(f'Expected class folders: {EXPECTED_CLASSES}, but found: {classes}')
    model = build_model(args.model, len(classes), pretrained=args.pretrained)
    if args.freeze_backbone and args.model != 'cnn':
        for parameter in model.parameters():
            parameter.requires_grad = False
        last_layer = model.fc if args.model == 'resnet50' else model.heads.head
        for parameter in last_layer.parameters():
            parameter.requires_grad = True
    model = model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    checkpoint_dir = Path('checkpoints')
    result_dir = Path('results')
    checkpoint_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    best_accuracy = -1.0
    save_to = checkpoint_dir / f'{args.model}.pt'
    history = []
    for epoch in range(args.epochs):
        train_scores = run_epoch(model, loaders['train'], loss_fn, device, optimizer)
        val_scores = run_epoch(model, loaders['val'], loss_fn, device)
        history.append({'epoch': epoch + 1, 'train': train_scores, 'val': val_scores})
        print(f"Epoch {epoch + 1}/{args.epochs}: train={train_scores['accuracy']:.4f}, val={val_scores['accuracy']:.4f}")
        if val_scores['accuracy'] > best_accuracy:
            best_accuracy = val_scores['accuracy']
            torch.save({
                'architecture': args.model,
                'class_names': classes,
                'state_dict': model.state_dict(),
                'image_size': 224,
                'validation_accuracy': best_accuracy
            }, save_to)
    saved = torch.load(save_to, map_location=device, weights_only=True)
    model.load_state_dict(saved['state_dict'])
    metrics = test_model(model, loaders['test'], device, classes)
    metrics['architecture'] = args.model
    metrics['best_validation_accuracy'] = best_accuracy
    metrics['history'] = history
    output = result_dir / f'{args.model}_metrics.json'
    output.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f'Checkpoint: {save_to}')
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
