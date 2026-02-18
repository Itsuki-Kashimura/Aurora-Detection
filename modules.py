import os

import pandas as pd
import PIL.Image as Image
import torch
import torch.nn as nn
import torchvision
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy, F1Score, Precision, Recall


# Custom dataset class for loading and preprocessing image data
class ImageDataSet(torch.utils.data.Dataset):


    def __init__(self, path_to_labels_csv, path_to_img_dir, crop_size):

        self.df_labels = pd.read_csv(path_to_labels_csv, index_col=None)
        self.filenames_img = self.df_labels['filename_img']
        self.labels_detection = self.df_labels['label_detection']
        self.path_to_img_dir = path_to_img_dir
        self.crop_size = crop_size
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),            # Resize the smaller edge of the image to 256-pixels while maintaining aspect ratio
            torchvision.transforms.CenterCrop(crop_size),  # Center crop the image to the specified crop size
            torchvision.transforms.ToTensor()              # Convert the image to a PyTorch tensor and normalize pixel values to [0, 1]
        ])
    

    def __len__(self):

        # Return the number of samples in the dataset
        return len(self.filenames_img)
    

    def __getitem__(self, idx):

        # Get the target value and normalized features for a given index
        img = Image.open(os.path.join(self.path_to_img_dir, self.filenames_img.iloc[idx]))
        img = self.transform(img)
        label = self.labels_detection.iloc[idx]

        return (img, label)



# Training loop for one epoch
def train(model, train_loader, optimizer):

    model.train()  # Set model to training mode

    for batch in train_loader: # batch = ()

        # Move data to GPU
        img, label = [x.to('cuda', non_blocking=True) for x in batch]

        optimizer.zero_grad()  # Reset gradients

        output = model(img)  # Forward pass
        loss = CrossEntropyLoss()(output, label)  # Compute Cross Entropy loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights



# Evaluate training loss and error for one epoch
def train_eval(model, train_loader, epoch, history, classification, num_classes):

    model.eval()  # Set model to evaluation mode

    loss_total = 0
    accuracy_total = 0
    precision_total = 0
    recall_total = 0
    f1_total = 0
    num_batches = 0

    with torch.no_grad():

        for batch in train_loader:

            # Move data to GPU
            img, label = [x.to('cuda', non_blocking=True) for x in batch]

            output = model(img)  # Forward pass

            Accuracy_metric = Accuracy(task=classification, num_classes=num_classes).to('cuda')
            Precision_metric = Precision(task=classification, num_classes=num_classes).to('cuda')
            Recall_metric = Recall(task=classification, num_classes=num_classes).to('cuda')
            F1_metric = F1Score(task=classification, num_classes=num_classes).to('cuda')

            loss_total += CrossEntropyLoss()(output, label)  # Accumulate Cross Entropy loss            
            accuracy_total += Accuracy_metric(output, label)  # Accumulate Accuracy    
            precision_total += Precision_metric(output, label)  # Accumulate precision
            recall_total += Recall_metric(output, label)  # Accumulate recall
            f1_total += F1_metric(output, label)  # Accumulate F1 score
            num_batches += 1

    loss_avg = loss_total / num_batches  # Average Cross Entropy loss
    accuracy_avg = accuracy_total / num_batches  # Average Accuracy
    precision_avg = precision_total / num_batches  # Average precision
    recall_avg = recall_total / num_batches  # Average recall
    f1_avg = f1_total / num_batches  # Average F1 score

    # Record metrics in history
    history['epoch'].append(epoch)
    history['train_loss'].append(loss_avg.item())
    history['train_accuracy'].append(accuracy_avg.item())
    history['train_precision'].append(precision_avg.item())
    history['train_recall'].append(recall_avg.item())
    history['train_f1'].append(f1_avg.item())

    print(
        f' --Validation-- '
        f'Loss: {loss_avg:.4f}  '
        f'Accuracy: {accuracy_avg:.4f}  '
        f'Precision: {precision_avg:.4f}  '
        f'Recall: {recall_avg:.4f}  '
        f'F1 Score: {f1_avg:.4f}\n'
    )



# Evaluate validation loss and error for one epoch
def valid(model, valid_loader, history, classification, num_classes):

    model.eval()  # Set model to evaluation mode

    loss_total = 0
    accuracy_total = 0
    precision_total = 0
    recall_total = 0
    f1_total = 0
    num_batches = 0

    with torch.no_grad():

        for batch in valid_loader:

            # Move data to GPU
            img, label = [x.to('cuda', non_blocking=True) for x in batch]

            output = model(img)  # Forward pass

            Accuracy_metric = Accuracy(task=classification, num_classes=num_classes).to('cuda')
            Precision_metric = Precision(task=classification, num_classes=num_classes).to('cuda')
            Recall_metric = Recall(task=classification, num_classes=num_classes).to('cuda')
            F1_metric = F1Score(task=classification, num_classes=num_classes).to('cuda')

            loss_total += CrossEntropyLoss()(output, label)  # Accumulate Cross Entropy loss
            accuracy_total += Accuracy_metric(output, label)  # Accumulate accuracy
            precision_total += Precision_metric(output, label)  # Accumulate precision
            recall_total += Recall_metric(output, label)  # Accumulate recall
            f1_total += F1_metric(output, label)  # Accumulate F1 score
            num_batches += 1

    loss_avg = loss_total / num_batches  # Average Cross Entropy loss
    accuracy_avg = accuracy_total / num_batches  # Average Accuracy
    precision_avg = precision_total / num_batches  # Average precision
    recall_avg = recall_total / num_batches  # Average recall
    f1_avg = f1_total / num_batches  # Average F1 score

    # Record metrics in history
    history['valid_loss'].append(loss_avg.item())
    history['valid_accuracy'].append(accuracy_avg.item())
    history['valid_precision'].append(precision_avg.item())
    history['valid_recall'].append(recall_avg.item())
    history['valid_f1'].append(f1_avg.item())

    print(
        f' --Validation-- '
        f'Loss: {loss_avg:.4f}  '
        f'Accuracy: {accuracy_avg:.4f}  '
        f'Precision: {precision_avg:.4f}  '
        f'Recall: {recall_avg:.4f}  '
        f'F1 Score: {f1_avg:.4f}\n'
    )



# Run model on test data and estimated labels and true labels for all test samples
def test(model, test_loader, classification, num_classes):

    model.eval()  # Set model to evaluation mode
    outputs = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            img, label = [x.to('cuda', non_blocking=True) for x in batch]
            output = model(img)
            outputs.append(output)
            labels.append(label)

    # Concatenate outputs and labels from all batches
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)

    # Calculate and print test metrics
    Accuracy_metric = Accuracy(task=classification, num_classes=num_classes).to('cuda')
    Precision_metric = Precision(task=classification, num_classes=num_classes).to('cuda')
    Recall_metric = Recall(task=classification, num_classes=num_classes).to('cuda')
    F1_metric = F1Score(task=classification, num_classes=num_classes).to('cuda')

    loss = CrossEntropyLoss()(outputs, labels)
    accuracy = Accuracy_metric(outputs, labels)
    precision = Precision_metric(outputs, labels)
    recall = Recall_metric(outputs, labels)
    f1 = F1_metric(outputs, labels)

    print(
        f' --Test-- '
        f'Loss: {loss:.4f}  '
        f'Accuracy: {accuracy:.4f}  '
        f'Precision: {precision:.4f}  '
        f'Recall: {recall:.4f}  '
        f'F1 Score: {f1:.4f}\n'
    )

    return outputs, labels



# Customed ResNet18 model
from torchvision.models import resnet18


class CustomResNet18(nn.Module):
    
    def __init__(self, num_classes, num_channels, weights):
        super().__init__()
        self.model = resnet18(weights=weights)

        # Replace the first conv layer if input image channels are different from 3(default for ResNet18)
        if num_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels=num_channels,
                out_channels=self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=self.model.conv1.bias is not None
            )

        # Replace the output layer to match the number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)


    def forward(self, x):

        return self.model(x)



# Customed ResNet50 model
from torchvision.models import resnet50


class CustomResNet50(nn.Module):

    def __init__(self, num_classes, num_channels, weights):
        super().__init__()
        self.model = resnet50(weights=weights)
        if num_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels=num_channels,
                out_channels=self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=self.model.conv1.bias is not None
            )

        # Replace the first conv layer if input image channels are different from 3(default for ResNet50)
        if num_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels=num_channels,
                out_channels=self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=self.model.conv1.bias is not None
            )

        # Replace the output layer to match the number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)


    def forward(self, x):

        return self.model(x)
    


# Customed VGG16 model
from torchvision.models import vgg16


class CustomVGG16(nn.Module):

    def __init__(self, num_classes, num_channels, weights):

        super().__init__()

        self.model = vgg16(weights=weights)

        # Replace the first conv layer if input image channels are different from 3(default for VGG16)
        if num_channels != 3:
            self.model.features[0] = nn.Conv2d(
                in_channels=num_channels,
                out_channels=self.model.features[0].out_channels,
                kernel_size=self.model.features[0].kernel_size,
                stride=self.model.features[0].stride,
                padding=self.model.features[0].padding,
                bias=self.model.features[0].bias is not None
            )

        # Replace the output layer to match the number of classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)


    def forward(self, x):

        return self.model(x)
    


# Customed VGG19 model
from torchvision.models import vgg19


class CustomVGG19(nn.Module):
    
    def __init__(self, num_classes, num_channels, weights):

        super().__init__()

        self.model = vgg19(weights=weights)

        # Replace the first conv layer if input image channels are different from 3(default for VGG19)
        if num_channels != 3:
            self.model.features[0] = nn.Conv2d(
                in_channels=num_channels,
                out_channels=self.model.features[0].out_channels,
                kernel_size=self.model.features[0].kernel_size,
                stride=self.model.features[0].stride,
                padding=self.model.features[0].padding,
                bias=self.model.features[0].bias is not None
            )

        # Replace the output layer to match the number of classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)


    def forward(self, x):

        return self.model(x)
    


# Customed AlexNet model
from torchvision.models import alexnet


class CustomAlexNet(nn.Module):
    
    def __init__(self, num_classes, num_channels, weights):

        super().__init__()

        self.model = alexnet(weights=weights)

        # Replace the first conv layer if input image channels are different from 3(default for AlexNet)
        if num_channels != 3:
            self.model.features[0] = nn.Conv2d(
                in_channels=num_channels,
                out_channels=self.model.features[0].out_channels,
                kernel_size=self.model.features[0].kernel_size,
                stride=self.model.features[0].stride,
                padding=self.model.features[0].padding,
                bias=self.model.features[0].bias is not None
            )

        # Replace the output layer to match the number of classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)


    def forward(self, x):

        return self.model(x)

