from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from app.config import MODEL_PATH, CLASS_NAMES, IMAGE_SIZE

class WoundPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f'Model not found: {model_path}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(CLASS_NAMES))
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image: Image.Image) -> dict:
        image = image.convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        index = int(torch.argmax(probabilities).item())
        return {
            'injury': CLASS_NAMES[index],
            'confidence': round(float(probabilities[index]) * 100, 2),
            'probabilities': {name: round(float(probabilities[i]) * 100, 2) for i, name in enumerate(CLASS_NAMES)},
        }
