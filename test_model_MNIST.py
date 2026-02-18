import time
import pandas as pd
import torch
import torchvision
from modules import test
from modules import CustomResNet18, CustomResNet50, CustomVGG16, CustomVGG19, CustomAlexNet


def main(name_basemodel, num_classes, num_channels_img):

    # Set random seed for reproducibility
    torch.manual_seed(1000)

    # Define classification type and number of classes
    classification = 'multiclass' if num_classes > 2 else 'binary'

    # Define the model and select the appropriate size of images based on the model architecture
    if name_basemodel == 'ResNet18':
        model = CustomResNet18(num_classes=num_classes, num_channels=num_channels_img, weights=None).to('cuda')
        crop_size = 224
    elif name_basemodel == 'ResNet50':
        model = CustomResNet50(num_classes=num_classes, num_channels=num_channels_img, weights=None).to('cuda')
        crop_size = 224
    elif name_basemodel == 'VGG16':
        model = CustomVGG16(num_classes=num_classes, num_channels=num_channels_img, weights=None).to('cuda')
        crop_size = 224
    elif name_basemodel == 'VGG19': 
        model = CustomVGG19(num_classes=num_classes, num_channels=num_channels_img, weights=None).to('cuda')
        crop_size = 224
    elif name_basemodel == 'AlexNet':
        model = CustomAlexNet(num_classes=num_classes, num_channels=num_channels_img, weights=None).to('cuda')
        crop_size = 227
    else:
        raise ValueError(f"Invalid model name: {name_basemodel}. Choose from 'ResNet18', 'ResNet50', 'VGG16', 'VGG19', 'AlexNet'.")
    
    model.load_state_dict(torch.load('path-to-trained-model'))  # Load trained weights

    # Load training and validation datasets
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.ToTensor()
        ]),
        download=True
    )

    # Setup data loaders
    loader_args = dict(batch_size=16, num_workers=3, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, **loader_args)

    # Measure time for the test process
    torch.cuda.synchronize()
    start = time.time()  # Start timer

    # Test the model and obtain mean and std of predictions
    ouputs, labels = test(model, test_loader, classification, num_classes)

    torch.cuda.synchronize()
    elapsed_time = time.time() - start  # Calculate elapsed time
    print(f"Test completed in {elapsed_time:.2f} seconds.")

    # Save test results (mean and std) to CSV file
    results = pd.DataFrame({
        'label_estimated': ouputs,
        'label_correct': labels
    })
    results.to_csv('path-to-save-test-predictions', index=False)



if __name__ == "__main__":

    main(
        name_basemodel='ResNet18',  # Model architecture to use (e.g., 'ResNet18', 'ResNet50', 'VGG16', 'VGG19', 'AlexNet')
        num_classes=10,       # Number of classes for classification (2 for binary, >2 for multiclass)
        num_channels_img=1,  # Number of channels in the input images (e.g., 1 for grayscale, 3 for RGB)
    )

