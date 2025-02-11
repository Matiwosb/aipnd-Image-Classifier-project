import torch
from torch import nn
from torchvision import models
import argparse
import json
from PIL import Image
import numpy as np
from tabulate import tabulate

# Load model checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if checkpoint['arch'] != 'mobilenet_v2':
        raise ValueError("Checkpoint architecture is not 'mobilenet_v2'.")
    model = models.mobilenet_v2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    print("Model loaded successfully!")
    return model

# Process the image
def process_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    if width < height:
        img = img.resize((256, int(256 * (height / width))))
    else:
        img = img.resize((int(256 * (width / height)), 256))
    width, height = img.size
    img = img.crop(((width - 224) / 2, (height - 224) / 2, (width + 224) / 2, (height + 224) / 2))
    np_image = np.array(img) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return torch.tensor(np_image, dtype=torch.float32)

# Predict
def predict(image_path, model, topk, device):
    image_tensor = process_image(image_path).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.exp(output)
    top_probabilities, top_indices = probabilities.topk(topk, dim=1)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]
    return top_probabilities[0].tolist(), top_classes

def display_results_as_table(top_probs, top_classes):
    """Create and display a table for the top classes and probabilities."""
    # Combine data into a table
    formatted_probs = [f"{prob:.4f}" for prob in top_probs]
    data = list(zip(top_classes, formatted_probs))
    table = tabulate(data, headers=["Class", "Probability"], tablefmt="pretty")
    print(table)

# Main
def main():
    parser = argparse.ArgumentParser(description='Predict flower name and class probability.')
    parser.add_argument('input', type=str, help='Path to the image.')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to real names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available.')
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    model.to(device)
    top_probs, top_classes = predict(args.input, model, args.top_k, device)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name.get(cls, cls) for cls in top_classes]

    print(f"Predictions for {args.input}:")
    display_results_as_table(top_probs, top_classes)

    # print(f"Predictions for {args.input}:")
    # print('Top Classes and Probabilities')
    # for i in range(len(top_probs)):
    #     print(f"{top_classes[i]}: {top_probs[i]:.4f}")

if __name__ == "__main__":
    main()