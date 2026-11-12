PROJECT_TITLE = 'Lung CT Image Classification'
WARNING_TEXT = 'Research demo only. Not a medical device and not for clinical diagnosis.'

import argparse
import base64
import io
from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

from predict import classify_image, load_checkpoint


def encode_image(image):
    output = io.BytesIO()
    image.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode('ascii')


def create_app(checkpoint):
    model, classes, architecture, device = load_checkpoint(checkpoint)
    site = Flask(__name__)
    site.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

    @site.route('/', methods=['GET', 'POST'])
    def home():
        result = None
        error = None
        if request.method == 'POST':
            upload = request.files.get('image')
            if not upload or not upload.filename:
                error = 'Choose an image first.'
            else:
                try:
                    content = upload.read()
                    with Image.open(io.BytesIO(content)) as source:
                        source.load()
                        original = source.convert('RGB')
                    if original.width * original.height > 25_000_000:
                        raise ValueError('Image dimensions are too large.')
                    prediction = classify_image(original, model, classes, architecture, device)
                    result = {
                        'label': prediction['label'].replace('_', ' ').title(),
                        'confidence': f"{prediction['confidence']:.1%}",
                        'input_image': encode_image(original),
                        'heatmap_image': encode_image(prediction['overlay']),
                        'scores': [(key.replace('_', ' ').title(), f'{value:.1%}') for key, value in prediction['probabilities'].items()]
                    }
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    error = f'Could not read image: {exc}'
        return render_template('index.html', result=result, error=error, project=PROJECT_TITLE, warning=WARNING_TEXT)

    @site.errorhandler(413)
    def too_large(error):
        return 'Image exceeds 10 MB.', 413

    return site


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/cnn.pt')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    if not Path(args.checkpoint).exists():
        parser.error(f'Checkpoint not found: {args.checkpoint}. Train a model first.')
    create_app(args.checkpoint).run(host='127.0.0.1', port=args.port, debug=False)
