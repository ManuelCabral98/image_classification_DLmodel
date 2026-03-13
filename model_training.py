# LIBRARY FOR HANDLE DYNAMIC PATH
from pathlib import Path
# RESNET ECOSYSTEM EXPERT IN IMAGES
from torchvision import transforms, datasets
from torchvision.models import ResNet18_Weights, resnet18
# TOOLS FOR LOADING AND SPLITING IMAGES
from torch.utils.data import DataLoader, random_split
# PYTORCH CORE
import torch
import torch.nn as nn


# TRADITIONAL WAY TO TRANSFORM DATASETS INTO 'RESNET18' LEGIBLE DATASET
# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]

# images_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std)
# ])

# AUTOMATIC WAY INTRODUCED BY PYTORCH
my_transforms = ResNet18_Weights.DEFAULT.transforms()

# DETECTING IN WHAT FOLDER IS THE SCRIPT
code_dir = Path(__file__).resolve().parent

# CONNECTING ABSOLUTE PATH WITH DATASET PATH
dataset_path = code_dir / "animals"

# READ MY 'ANIMALS' FOLDER TO SEPARATE THE OBJECTS IN IT, BESIDES THAT IT 'TRANSFORM' THE IMAGES WITH THE CONFIGURATION 
# WE SET IN THE PREVIOUS VARIABLE (my_transforms)
dataset = datasets.ImageFolder(root=dataset_path, transform=my_transforms)

# SPLIT THE DATASET IN 2 PARTS, ONE FOR TRAINING AND THE OTHER ONE FOR VALIDATION
train_size = int(len(dataset) * 0.8)
validation_size = int(len(dataset) - train_size)

train_data, validation_data = random_split(dataset, [train_size, validation_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False)

# INSTANTIANTING THE MODEL
model = resnet18(ResNet18_Weights.DEFAULT)

# FREEZEING THE MODEL TO AVOID 'TRAINING' OR 'LEARNING'
# OUR GOAL ISN'T TO IMPROVE PARAMETERS' GRADIENTS, BECAUSE MODEL IS ALREADY TRAINED
for parameter in model.parameters():
    parameter.requires_grad = False

# REPLACING 'FULLY CONNECTED' (FC) LAYER FOR ONLY 5 OUT FEATURES
# DEFAULT RESNET18 MODEL HAS 1000 OUT FEATURES AND LOTS OF CATEGORIES (DOGS, CATS, ETC...)
# WE ONLY NEED 5 FOR OUR DATASET
in_features_qty = model.fc.in_features
model.fc = nn.Linear(in_features=in_features_qty, out_features=5)

# LOOKING FOR MACHINE COMPONENT TO WORK WITH
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): # FOR Macs WITH M1/M2/M3 CHIPS
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# UPLOADING MODEL UP TO SELECTED DEVICE
# GOLD RULE: DATASET AND MODEL MUST BE IN THE SAME DEVICE
model = model.to(device)
