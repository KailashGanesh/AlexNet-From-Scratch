from AlexNet_pytorch import AlexNetTorch
import torch
import sys
from PIL import Image
from torchvision import transforms

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

def Inference(model, image_tensor, device, class_name, visualize=False):

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_tensor.to(device)
        output, all_layers = model.forward(image_tensor, visualize=visualize)
        score, predicted = torch.max(output, 1)
        print(score, predicted)

    predicted_class = class_name[predicted.item()]

    return predicted_class, score.item(), all_layers

def plot_all_layers(all_layers):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for i in range(10):
        axes[i].imshow(all_layers[i], cmap='gray')
        axes[i].set_title(f'Layer {i+1}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


if  __name__ == "__main__":
    print(sys.argv)
    device = setup_device()
    image_tensor, img, img_original = preprocess_image(sys.argv[1])
    model = load_model(device=device)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10']
    predicted_class, score, all_layers = Inference(model, image_tensor, device, class_names, visualize=True)
    print(f"Predicted class: {predicted_class} with score: {score:.4f}")

    all_layers = [img_original, img, image_tensor, *[layer.squeeze().cpu().numpy() for layer in all_layers]] 

    plot_all_layers(all_layers)


# with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in test_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             del images, labels, outputs

#         print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))