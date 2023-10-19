import torch
from torchvision import datasets
from torchvision import transforms

# Essa função é responsável por criar os loaders de treinamento, validação e teste
# a partir de um diretório fornecido através do parâmetro "dataset_dir"
def get_loaders(dataset_dir:str, batch_size:int=16):

    # Pré-processamento indicado para trabalhar com o modelo densenet
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_folder = datasets.ImageFolder(dataset_dir, preprocess)

    train_split, val_split = torch.utils.data.random_split(data_folder, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_split,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    return train_loader, val_loader