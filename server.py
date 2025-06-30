import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
from inference import inference, plot_feature_map_details, preprocess_image, class_names
import torch
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return render_template('index.html', error='No image file provided')
        image_file = request.files['image']
        if image_file.filename == '' or image_file.filename is None:
            return render_template('index.html', error='No selected file')
        
        os.makedirs('uploads', exist_ok=True)
        image_path = os.path.join('uploads', image_file.filename)
        image_file.save(image_path)
        
        image_tensor, img, img_original = preprocess_image(image_path)
        predicted_class, score, all_layers = inference(image_tensor, class_names=class_names, visualize=True)

        plot_paths = []

        # 1. Plot initial images together
        initial_images = [
            {'data': img_original, 'title': f'Original Image ({img_original.size[0]}x{img_original.size[1]})'},
            {'data': img, 'title': 'Resized Image (227x227)'},
            {'data': image_tensor, 'title': 'Normalized Input Tensor'}
        ]

        input_plot_path = plot_feature_map_details(initial_images, filename='input_images.png', cols=3)
        if input_plot_path:
            plot_paths.append(input_plot_path)

        # 2. Plot each convolutional layer separately
        conv_layer_idx = 1
        for i, layer_output in enumerate(all_layers):
            # Only plot tensors that have 4 dimensions (B, C, H, W), which are the conv layers
            if isinstance(layer_output, torch.Tensor) and layer_output.ndim == 4:
                title = f"Layer {conv_layer_idx} | Output Shape: {list(layer_output.shape)}"
                item_to_plot = [{'data': layer_output, 'title': title}]
                
                layer_plot_path = plot_feature_map_details(item_to_plot, filename=f'layer_{conv_layer_idx}_maps.png', cols=4)
                if layer_plot_path:
                    plot_paths.append(layer_plot_path)
                conv_layer_idx += 1
        
        print(f"Predicted class: {predicted_class} with score: {score:.4f}")
        
        os.remove(image_path)
        
        return render_template('index.html', 
                    file_name=image_file.filename,
                    predicted_class=predicted_class,
                    confidence=score,
                    all_layers=plot_paths,
                    success=True)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return render_template('index.html', error=f'Prediction failed: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)