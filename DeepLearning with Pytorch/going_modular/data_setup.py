"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_dataloaders(
    batch_size: int, 
    transform: transforms.Compose = None
):
  """Creates training and testing DataLoaders.

  Returns a tuple of (train_dataloader, test_dataloader, class_names).
  """
  if transform is None:
      transform = transforms.ToTensor()

  # Use FashionMNIST
  train_data = datasets.FashionMNIST(
      root="data",
      train=True,
      download=True,
      transform=transform
  )

  test_data = datasets.FashionMNIST(
      root="data",
      train=False,
      download=True,
      transform=transform
  )

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=0, # Simple for local
      pin_memory=True,
  )

  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=0,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
