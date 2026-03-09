from torchvision import transforms, datasets
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
from pathlib import Path

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

