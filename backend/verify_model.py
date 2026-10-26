import sys
import torch
import torch.nn as nn
from torchvision import models

path = sys.argv[1] if len(sys.argv) > 1 else 'models/best_wound_model_finetuned.pth'
classes = ['bruise', 'burn', 'cut']
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
state = torch.load(path, map_location='cpu', weights_only=True)
model.load_state_dict(state)
print('Model loaded successfully.')
print('Expected class-index mapping:')
for i, name in enumerate(classes): print(f'  {i} -> {name}')
