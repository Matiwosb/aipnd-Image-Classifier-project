import torch
from torch import nn, optim
from torchvision import transforms, models, datasets
import argparse
import os
import tqdm
def get_input_args():
  parser = argparse.ArgumentParser(description = 'Train a neural network on a dataset.')

  parser.add_argument('data_dir', type=str, help='Directory containing the dataset (train, valdation, test subfolders)')
  parser.add_argument('--save_dir', type=str, default='checkpoint', help='Directory to save the checkpoints')
  # parser.add_argument('--train_datasets')
  parser.add_argument('--arch', type=str, default='mobilenet_v2', choices=['vgg16', 'mobilenet_v2'], help='Choose model architecture')
  parser.add_argument('--learning_rate', type=float, default= 0.001, help='Learning Rate for training')
  parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
  parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
  parser.add_argument('--gpu', action='store_true', help='Use GPU if avaiable')

  return parser.parse_args()

def load_data(data_dir):
      data_training_transform = transforms.Compose([
      # transforms.RandomRotation(30),
      transforms.RandomHorizontalFlip(),
      transforms.RandomResizedCrop(224),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      ])

      data_validation_test_tranform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      ])

      # TODO: Load the datasets with ImageFolder
      train_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_training_transform)
      validation_datasets = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=data_validation_test_tranform)
      test_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_validation_test_tranform)

      batch_size=128

      # TODO: Using the image datasets and the trainforms, define the dataloaders
      training_dataloader = torch.utils.data.DataLoader(train_datasets,
                                                batch_size=batch_size,
                                                shuffle=True, pin_memory=True)

      validation_dataloader = torch.utils.data.DataLoader(validation_datasets,
                                                batch_size=batch_size,
                                                shuffle=False, pin_memory=True)

      test_dataloader = torch.utils.data.DataLoader(test_datasets,
                                                batch_size=batch_size,
                                                shuffle=False, pin_memory=True)

      return training_dataloader, validation_dataloader, train_datasets.class_to_idx

# Build the model
def build_model(arch, num_classes, hidden_units):
    """Builds the model based on the architecture choice"""
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_features = model.classifier[0].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )
    
    elif arch == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        input_features = model.last_channel
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_features, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )
    
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return model

# Training the model
def training_model(model, training_dataloader, validation_dataloader, criterion, optimizer, epochs, device):
  # Training and validation loop
  # epochs = 5
  train_losses, val_losses = [], []

  for epoch in range(epochs):
      # Training
      model.train()
      train_loss = 0
      for images, labels in training_dataloader:
          images, labels = images.to(device), labels.to(device)
          # print("labels" + str(labels))
          # break

          optimizer.zero_grad()
          outputs = model(images)
          # print(outputs)
          loss = criterion(outputs, labels)
          # print(f"Loss: {loss.item()}")
          loss.backward()
          optimizer.step()

          train_loss += loss.item()

          train_losses.append(train_loss / len(training_dataloader))

      # Validation
      model.eval()
      val_loss = 0
      accuracy = 0
      
      with torch.no_grad():
          for images, labels in validation_dataloader:
              images, labels = images.to(device), labels.to(device)
              outputs = model(images)
              loss = criterion(outputs, labels)
              val_loss += loss.item()

              # Calculate accuracy
              ps = torch.exp(outputs)
              top_p, top_class = ps.topk(1, dim=1)
              equals = top_class == labels.view(*top_class.shape)
              accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

      # Track losses
      # train_losses.append(train_loss / len(training_dataloader))
      val_losses.append(val_loss / len(validation_dataloader))

      print(f"Epoch {epoch+1}/{epochs}.. "
            # f"Loss: {loss:.3f}.. "
            f"Train loss: {train_losses[-1]:.4f}.. "
            f"Validation loss: {val_losses[-1]:.3f}.. "
            f"Validation accuracy: {accuracy / len(validation_dataloader):.3f}")


# def save_checkpoint(model, class_to_idx, save_dir, train_datasets, arch, optimizer, epochs):
def save_checkpoint(model, class_to_idx, save_dir, arch, optimizer, epochs):
  # model.class_to_idx = train_datasets.class_to_idx

  checkpoint = {
      'class_to_idx': class_to_idx,
      'state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'epochs': epochs,
      'arch': arch, # Specify the architecture used
      'classifier': model.classifier # Save the classifier architecture
  }

  os.makedirs(save_dir, exist_ok=True)

  torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

  print(f"Checkpoint saved to {os.path.join(save_dir, 'checkpoint.pth')} successfully.")

def main():
  # parse command-line arguments
  args = get_input_args()

  # Load data
  training_dataloader, validation_dataloader, class_to_idx = load_data(args.data_dir)

  # Build the model and set device
  model = build_model(args.arch, num_classes=102, hidden_units=args.hidden_units)
  device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
  print(device)
  model.to(device)

  # Define criterion and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

  # Train the model
  training_model(model, training_dataloader, validation_dataloader, criterion, optimizer, args.epochs, device)

  # Save the checkpoint
  save_checkpoint(model, class_to_idx, args.save_dir, args.arch, optimizer, args.epochs)

if __name__ == "__main__":
  main()



  
    