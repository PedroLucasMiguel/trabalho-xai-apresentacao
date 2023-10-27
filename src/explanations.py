import torch.nn as nn
import os
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from lime import lime_image
import cv2
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt

def get_grad_cam(model:nn.Module) -> None:
    device = "cpu"

    samples = os.listdir(os.path.join("..", "samples"))

    img_name = ""

    while True:
        print("\nSelecione uma das imagens: ")
        
        for i in range(len(samples)):
            print(f"[{i}] - {samples[i]}")

        img_i = int(input("Resposta: "))

        if img_i >= 0 and img_i < len(samples):
            img_name = samples[img_i]
            break
            
        else:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Imagem inválida!")
            input("Pressione ENTER para recomeçar...")
        
    os.system('cls' if os.name == 'nt' else 'clear')

    img = Image.open(os.path.join("..", "samples", img_name)).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = preprocess(img)
    input_batch = img.unsqueeze(0)
    input_batch = input_batch.to(device)

    model = model.to(device)

    model.eval()

    outputs = model(input_batch)

    probs = F.softmax(outputs).detach().cpu().numpy()

    print(f"Classificação do modelo: {probs.argmax()}")
    outputs[:, probs.argmax()].backward()

    gradients = model.get_activations_gradient()
    gradients = torch.mean(gradients, dim=[0, 2, 3])
    layer_output = model.get_activations(input_batch)

    for i in range(len(gradients)):
        layer_output[:, i, :, :] *= gradients[i]

    layer_output = layer_output[0, : , : , :]

    img = cv2.imread(os.path.join("..", "samples", img_name))

    heatmap = torch.mean(layer_output, dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) 
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join("..", "output", model.__class__.__name__, "gradient.png"), heatmap)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(os.path.join("..", "output", model.__class__.__name__, "map.png"), superimposed_img)

    print(f"\nGrad-cam salvo em: ../outputs/{model.__class__.__name__}/[map, gradient].jpg")
    input("Pressione ENTER para continuar")

    pass

def get_cam(model:nn.Module) -> None:
    device = "cpu"

    samples = os.listdir(os.path.join("..", "samples"))

    img_name = ""

    while True:
        print("\nSelecione uma das imagens: ")
        
        for i in range(len(samples)):
            print(f"[{i}] - {samples[i]}")

        img_i = int(input("Resposta: "))

        if img_i >= 0 and img_i < len(samples):
            img_name = samples[img_i]
            break
            
        else:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Imagem inválida!")
            input("Pressione ENTER para recomeçar...")
        
    os.system('cls' if os.name == 'nt' else 'clear')

    img = Image.open(os.path.join("..", "samples", img_name)).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = preprocess(img)
    input_batch = img.unsqueeze(0)
    input_batch = input_batch.to(device)

    model = model.to(device)

    model.eval()

    # Armazenando a saída da útlima camada convolucional
    conv_forward_output = []
    def hook(model, input, output):
        conv_forward_output.append(output) 
    model.modification.register_forward_hook(hook)

    outputs = model(input_batch)

    probs = F.softmax(outputs, dim=1).data.squeeze()

    print(f"Classificação do modelo: {probs.argmax()}")

    params = list(model.parameters())

    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    conv_out = conv_forward_output[0].detach().numpy()
    bz, nc, h, w = conv_out.shape
    conv_out = conv_out.reshape((nc, h*w))
    
    cam = weight_softmax[probs.argmax()].dot(conv_out)
    img = cv2.imread(os.path.join("../samples", img_name))

    cam = cam.reshape(h, w)

    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    
    cam_img = np.uint8(255* cam_img)
    #cam_img = np.uint8(255 * ((cam-np.min(cam))/(np.max(cam)-np.min(cam))))

    cam_img = cv2.resize(cam_img, (img.shape[1], img.shape[0])) 
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join("..", "output", model.__class__.__name__, "gradient.png"), cam_img)
    superimposed_img = cam_img * 0.4 + img
    cv2.imwrite(os.path.join("..", "output", model.__class__.__name__, "map.png"), superimposed_img)

    print(f"\nGrad-cam salvo em: ../outputs/{model.__class__.__name__}/[map, gradient].jpg")
    input("Pressione ENTER para continuar")

    pass

def get_lime(model:nn.Module) -> None:

    explainer = lime_image.LimeImageExplainer()

    def get_pil_transform(): 
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
        ])    

        return transf

    def get_preprocess_transform():
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])     
        transf = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]) 
        return transf

    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando: {device}")

    def get_image(path):
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB') 
            

    samples = os.listdir(os.path.join("..", "samples"))

    img_name = ""

    while True:
        print("\nSelecione uma das imagens: ")
        
        for i in range(len(samples)):
            print(f"[{i}] - {samples[i]}")

        img_i = int(input("Resposta: "))

        if img_i >= 0 and img_i < len(samples):
            img_name = samples[img_i]
            break
            
        else:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Imagem inválida!")
            input("Pressione ENTER para recomeçar...")

    os.system('cls' if os.name == 'nt' else 'clear')

    img = get_image(os.path.join("..", "samples", img_name))

    model = model.to(device)

    def classify_func(img):
        model.eval()
        batch = torch.stack(tuple(preprocess_transform(i) for i in img), dim=0)
        batch = batch.to(device)
        outputs = model(batch)

        return F.softmax(outputs, dim=1).detach().cpu().numpy()

    explanation = explainer.explain_instance(np.array(pill_transf(img)), classify_func, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    o_img = np.array(img)
    img_boundry1 = cv2.resize(img_boundry1, dsize=(o_img.shape[1], o_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    plt.imshow(img_boundry1)
    plt.savefig(os.path.join("..", "output", model.__class__.__name__, "lime.png"))

    print(f"LIME salvo em: ../outputs/{model.__class__.__name__}/lime.png")
    input("Pressione ENTER para continuar")

    pass