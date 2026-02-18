import math
import time

import pandas as pd
import torch
import torchinfo
import torchvision
from modules import CustomAlexNet, CustomResNet18, CustomResNet50, CustomVGG16, CustomVGG19
from modules import train, train_eval, valid
from torch.optim.lr_scheduler import StepLR


def main(name_basemodel, num_classes, num_channels_img):

    # Set random seed for reproducibility
    torch.manual_seed(1000)

    # Define classification type and number of classes
    classification = 'multiclass' if num_classes > 2 else 'binary'

    # Define the model and select the appropriate size of images based on the model architecture
    if name_basemodel == 'ResNet18':
        model = CustomResNet18(num_classes=num_classes, num_channels=num_channels_img).to('cuda')
        crop_size = 224
    elif name_basemodel == 'ResNet50':
        model = CustomResNet50(num_classes=num_classes, num_channels=num_channels_img).to('cuda')
        crop_size = 224
    elif name_basemodel == 'VGG16':
        model = CustomVGG16(num_classes=num_classes, num_channels=num_channels_img).to('cuda')
        crop_size = 224
    elif name_basemodel == 'VGG19': 
        model = CustomVGG19(num_classes=num_classes, num_channels=num_channels_img).to('cuda')
        crop_size = 224
    elif name_basemodel == 'AlexNet':
        model = CustomAlexNet(num_classes=num_classes, num_channels=num_channels_img).to('cuda')
        crop_size = 227
    else:
        raise ValueError(f"Invalid model name: {name_basemodel}. Choose from 'ResNet18', 'ResNet50', 'VGG16', 'VGG19', 'AlexNet'.")

    # Load training and validation datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.ToTensor()
        ]),
        download=True
    )
    valid_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False,
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.ToTensor()
        ]),
        download=True
    )

    # Setup data loaders
    loader_args = dict(batch_size=16, num_workers=3, pin_memory=True, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_args)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, **loader_args)

    # Set up parameters for training
    epochs = 3  # Number of training epochs
    lr = 1e-3  # Initial learning rate
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)  # Adam optimizer
    scheduler = scheduler = StepLR(optimizer, step_size=10, gamma=math.pow(0.1, 1/10)) # Learning rate scheduler

    # History dictionary to store loss and error values for each epoch
    history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'valid_loss': [],
        'valid_accuracy': [],
        'valid_precision': [],
        'valid_recall': [],
        'valid_f1': []
    }

    # Print parameters for training
    print("Epochs: ", epochs)
    print("Learning rate: ", lr)
    print("Optimizer: Adam")
    print("Scheduler: StepLR")
    print("Model Summary:")
    # Display model architecture and parameter count
    torchinfo.summary(
        model,
        input_size=(1, num_channels_img, crop_size, crop_size)
    )

    print(
        '\n\n\n' +
        '~' * 50 +
        '\nLoss and error in each epoch\n' +
        '~' * 50 +
        '\n'
    )


    # Measure time for the training and validation process
    torch.cuda.synchronize()
    start = time.time()  # Start timer

    # Training and Validation loop
    for epoch in range(1, epochs+1):
        # Train the model for one epoch
        train(model, train_loader, optimizer)
        # Evaluate training performance and record metrics
        train_eval(model, train_loader, epoch, history, classification, num_classes)
        # Validate the model and record metrics
        valid(model, valid_loader, history, classification, num_classes)
        # Update learning rate
        scheduler.step()

    torch.cuda.synchronize()
    elapsed_time = time.time() - start  # Calculate elapsed time
    print(f"Training and validation completed in {elapsed_time:.2f} seconds.")


    # Save training history to CSV file
    pd.DataFrame(history).to_csv('path-to-save-history.csv', index=False)  #FIXME: Update with the actual path to save the CSV file of the history 
    # Save model weights
    torch.save(model.state_dict(), 'path-to-save-model')                   #FIXME: Update with the actual path to save the trained model weights



if __name__ == '__main__':

    main(
        name_basemodel='VGG16',  # Model architecture to use (e.g., 'ResNet18', 'ResNet50', 'VGG16', 'VGG19', 'AlexNet')
        num_classes=10,          # Number of classes for classification (2 for binary, >2 for multiclass)
        num_channels_img=1,      # Number of channels in the input images (e.g., 1 for grayscale, 3 for RGB)
    )

