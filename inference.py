from AlexNet_pytorch import AlexNetTorch
import torch
import sys
from PIL import Image
from torchvision import transforms
import os
import matplotlib

 # Use non-interactive backend for scripts
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import numpy as np
import math

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS for inference")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for inference")
    else:
        device = torch.device("cpu")
        print("Using CPU for inference")
    return device

def load_model(model_path = './models/best_model.pth', num_classes = 10, device = None):
    full_state_dict = torch.load(model_path, map_location=device)
    model_state_dict = full_state_dict.get('model_state_dict', full_state_dict)
    model = AlexNetTorch(num_classes=num_classes)

    try:
        model.load_state_dict(model_state_dict)
    except Exception as e:
        print(f"An error occurred while loading model: {e}")
        sys.exit(1)

    if device is None:
        device = setup_device()
    model.to(device=device)
    return model

def preprocess_image(img_path):
    img_original = Image.open(img_path)
    img = img_original.convert("RGB")
    img = img.resize((227, 227))
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, img, img_original

def predict(model, image_tensor, device, class_name = class_names, visualize=False):

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_tensor.to(device)
        output, all_layers = model.forward(image_tensor, visualize=visualize)
        score, predicted = torch.max(output, 1)
        print(score, predicted)

    predicted_class = class_name[predicted.item()]

    return predicted_class, score.item(), all_layers

def plot_feature_map_details(visualization_items, save_dir='static', filename='layers_detail.png', max_maps_per_layer=16, cols=None):
    """
    Generates a visualization. Can handle two cases:
    1. A list of multiple images/tensors, which it will plot in a single row or a grid if cols is provided.
    2. A single 4D tensor (feature maps), which it will plot as a grid.

    Args:
        visualization_items (list): A list of dictionaries, each with 'data' and 'title'.
        save_dir (str): The directory to save the plot.
        filename (str): The filename for the saved plot.
        max_maps_per_layer (int): Max number of feature maps to show per layer.
        cols (int, optional): The number of columns for the grid. Defaults to None.
    """
    os.makedirs(save_dir, exist_ok=True)

    # == Case 1: A single item which is a 4D Tensor (feature maps) -> Plot as a grid ==
    if len(visualization_items) == 1 and isinstance(visualization_items[0]['data'], torch.Tensor) and visualization_items[0]['data'].ndim == 4:
        item_dict = visualization_items[0]
        data = item_dict['data']
        title = item_dict['title']
        
        num_channels = data.shape[1]
        maps_to_plot = min(num_channels, max_maps_per_layer)

        if maps_to_plot == 0:
            return None

        if cols is None:
            cols = int(math.ceil(math.sqrt(maps_to_plot)))
        rows = int(math.ceil(maps_to_plot / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        fig.suptitle(title, fontsize=16, weight='bold', y=1.02)
        
        axes = np.array(axes).flatten()

        for i in range(maps_to_plot):
            ax = axes[i]
            feature_map = data[0, i, :, :].detach().cpu().numpy()
            ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f"Map {i+1}", fontsize=10)
            ax.axis('off')

        # Hide any unused subplots
        for i in range(maps_to_plot, len(axes)):
            axes[i].axis('off')

    # == Case 2: Multiple items -> Plot as a single row or grid ==
    else:
        num_items = len(visualization_items)
        if num_items == 0:
            return None

        if cols is None:
            cols = num_items
        rows = int(math.ceil(num_items / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        if num_items == 1:
            axes = [axes] # make sure axes is always iterable
        axes = np.array(axes).flatten()

        for i, item_dict in enumerate(visualization_items):
            ax = axes[i]
            data = item_dict['data']
            title = item_dict['title']

            if isinstance(data, Image.Image):
                ax.imshow(data)
            elif isinstance(data, torch.Tensor) and data.ndim == 4 and data.shape[1] == 3:
                # Normalize for visualization if it's not already in [0,1] range
                plot_data = data.squeeze(0).permute(1, 2, 0).cpu().numpy()
                mean = np.array([0.4914, 0.4822, 0.4465])
                std = np.array([0.2023, 0.1994, 0.2010])
                plot_data = std * plot_data + mean
                plot_data = np.clip(plot_data, 0, 1)
                ax.imshow(plot_data)
            else:
                ax.text(0.5, 0.5, f"Cannot display type: {type(data)}", ha='center', va='center')
            
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        
        for i in range(num_items, len(axes)):
            axes[i].axis('off')

    # Adjust layout to make room for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Grid visualization saved to {plot_path}")
    return plot_path


def inference(image_tensor, class_names=class_names, visualize=False):
    device = setup_device() 
    model = load_model(device=device)
    predicted_class, score, all_layers = predict(model, image_tensor, device, class_names, visualize)
    return predicted_class, score, all_layers

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]

    # == 1. Setup & Preprocessing ==
    img_original = Image.open(image_path)
    img_resized = img_original.convert("RGB").resize((227, 227))
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(img_resized).unsqueeze(0) # Add batch dimension

    # == 2. Inference ==
    device = setup_device()
    model = load_model(device=device)
    
    predicted_class, score, all_layers = predict(model, image_tensor, device, class_names, visualize=True)
    print(f"Predicted class: {predicted_class} with score: {score:.4f}")

    # == 3. Visualization ==
    visualization_items = [
        {'data': img_original, 'title': f'Original Image ({img_original.size[0]}x{img_original.size[1]})'},
        {'data': image_tensor, 'title': 'Normalized Input Tensor'}, # Pass tensor to see preprocessed image
    ]

    layer_names = []
    if hasattr(model, 'features') and isinstance(model.features, torch.nn.Sequential):
        layer_names = [name for name, module in model.features.named_children() if isinstance(module, (torch.nn.Conv2d, torch.nn.ReLU, torch.nn.MaxPool2d))]

    for i, layer_output in enumerate(all_layers):
        layer_name = layer_names[i] if i < len(layer_names) else f'Layer {i+1}'
        title = f"{layer_name} | Output Shape: {list(layer_output.shape)}"
        visualization_items.append({'data': layer_output, 'title': title})

    plot_feature_map_details(visualization_items)