import torch 
import torchvision
import os 
from pathlib import Path
import data_setup, engine , utils

# set the device to gpu if it is available
device='cuda' if torch.cuda.is_available() else 'cpu'

#get the ResNet50 weights
weights=torchvision.models.ResNet50_Weights.DEFAULT
#define the ResNet50 model with pretrained weights and send it to device
resnet_50=torchvision.models.resnet50(weights=weights).to(device)
#Set the requires_grad of all parameters to false for transfer learning
for param in resnet_50.parameters():
    param.requires_grad=False
#define a new classifier head for ResNet50 to suit our problem
resnet_50.fc=torch.nn.Linear(in_features=2048,out_features=101)

# Create Food101 training data transforms (only perform data augmentation on the training images)
food101_train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.TrivialAugmentWide(),
    torchvision.transforms.Resize(256,interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    torchvision.transforms.CenterCrop(size=224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
])
#Create Food101 testing data transforms
food101_test_transforms=torchvision.transforms.Compose([
    #torchvision.transforms.TrivialAugmentWide(),
    torchvision.transforms.Resize(256,interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    torchvision.transforms.CenterCrop(size=224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
])

#Download the food101 dataset from torchvision
data_dir=Path('data')
train_dir=torchvision.datasets.Food101(root=data_dir,split='train',transform=food101_train_transforms,download=True)
test_dir=torchvision.datasets.Food101(root=data_dir,split='test',transform=food101_test_transforms,download=True)

#define a function to split our data 
def split_dataset(dataset:torchvision.datasets, split_size:float=0.2, seed:int=42):
    """Randomly splits a given dataset into two proportions based on split_size and seed.

    Args:
        dataset (torchvision.datasets): A PyTorch Dataset, typically one from torchvision.datasets.
        split_size (float, optional): How much of the dataset should be split? 
            E.g. split_size=0.2 means there will be a 20% split and an 80% split. Defaults to 0.2.
        seed (int, optional): Seed for random generator. Defaults to 42.

    Returns:
        tuple: (random_split_1, random_split_2) where random_split_1 is of size split_size*len(dataset) and 
            random_split_2 is of size (1-split_size)*len(dataset).
    """
    # Create split lengths based on original dataset length
    length_1 = int(len(dataset) * split_size) # desired length
    length_2 = len(dataset) - length_1 # remaining length
        
    # Print out info
    print(f"[INFO] Splitting dataset of length {len(dataset)} into splits of size: {length_1} ({int(split_size*100)}%), {length_2} ({int((1-split_size)*100)}%)")
    
    # Create splits with given random seed
    random_split_1, random_split_2 = torch.utils.data.random_split(dataset, 
                                                                   lengths=[length_1, length_2],
                                                                   generator=torch.manual_seed(seed)) # set the random seed for reproducible splits
    return random_split_1, random_split_2

# Use only 40% of data for training and testing
train_data_half,_=split_dataset(dataset=train_dir,split_size=0.4)
test_data_half,_=split_dataset(dataset=test_dir,split_size=0.4)

# turn our dataset into DataLoader
train_dataloader=torch.utils.data.DataLoader(dataset=train_data_half,batch_size=32,shuffle=True,)
test_dataloader=torch.utils.data.DataLoader(dataset=test_data_half,batch_size=32,shuffle=False)

# Setup optimizer
optimizer = torch.optim.Adam(params=resnet_50.parameters(),
                             lr=1e-3)

# Setup loss function
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1) # throw in a little label smoothing because so many classes

# Train our model for 10 epochs
torch.manual_seed(42)
torch.cuda.manual_seed(42)  
resnet50_results = engine.train(model=resnet_50,
                                        train_dataloader=train_dataloader,
                                        test_dataloader=test_dataloader,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        epochs=10,
                                        device=device)

# create a path named models and make it a directory
save_model_path=Path("models")
save_model_path.mkdir(parents=True,exist_ok=True)

# Create a model path
resnet_50_model_path = "pretrained_resnet50_feature_extractor_food101_40_percent.pth" 

# Save FoodVision Big model
utils.save_model(model=resnet_50,
                 target_dir="models",
                 model_name=resnet_50_model_path)
