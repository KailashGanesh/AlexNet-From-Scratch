from AlexNet_pytorch import AlexNetTorch
import torch
import sys

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

def load_model(model_path, num_classes, device):
    model = AlexNetTorch(num_classes=num_classes)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"An error occurred while loading model: {e}")
        sys.exit(1)
    
    model.to(device=device)
    return model


def preprocess_image(img_path):
    with open("r", img_path) as img:
        print(img)

def Inference(model, image_tensor, device, class_name):

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_tensor.to(device)
        output = model.forward(x)
        _, predicted = torch.max(output, 1)

    predicted_class = class_name[predicted.item()]

    return predicted_class


if  __name__ == "__main__":
    print(sys.argv)
    preprocess_image(sys.argv[1])



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