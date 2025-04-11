import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=False)
model.load_state_dict(torch.load('./best_model.pth', map_location=device))
model = model.to(device)
model.eval()

class_labels = ['Healthy', 'Multiple Diseases', 'Rust', 'Scab']

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    image = Image.fromarray(image).convert("RGB")  
    transformed_image = image_transform(image).unsqueeze(0).to(device) 
    with torch.no_grad():
        outputs = model(transformed_image)
        predicted_class = torch.argmax(outputs, dim=1).item()  
    return class_labels[predicted_class]

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload an Image"),  
    outputs=gr.Label(num_top_classes=4, label="Predicted Class"), 
    title="Plant Disease Classifier",
    description="Upload an image to classify it as Healthy, Multiple Diseases, Rust, or Scab."
)

if __name__ == "__main__":
    interface.launch()

