import os
from train import train
from models.densenet import *
from torchvision.models import densenet201
from explanations import *

# Hiper parâmetros
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001

def __show_invalid_message(message:str) -> None:
    os.system('cls' if os.name == 'nt' else 'clear')
    print(message)
    input("Pressione ENTER para recomeçar...")

def op_train() -> None:

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        batch_size = int(input("Batch size ( > 0): "))

        if batch_size > 0:
            epochs = int(input("Epochs ( > 0): "))

            if epochs > 0:
                lr = float(input("Learning rate ( > 0): "))

                if lr > 0:
                    print("\nSelecione o modelo para treinamento: ")
                    print("[1] - DenseNet201")
                    print("[2] - DenseNet201EncoderDecoder")

                    op = int(input("Resposta: "))

                    match op:
                        case 1:
                            os.system('cls' if os.name == 'nt' else 'clear')
                            train(
                                DenseNet201GradCam(
                                    densenet201(weights='IMAGENET1K_V1'), 
                                    2),
                                os.path.join("../datasets", "PetImages"),
                                batch_size,
                                epochs,
                                lr
                            )
                            return
                            
                        case 2:
                            os.system('cls' if os.name == 'nt' else 'clear')
                            train(
                                DenseNet201EncoderDecoder(
                                    densenet201(weights='IMAGENET1K_V1'), 
                                    2),
                                os.path.join("../datasets", "PetImages"),
                                batch_size,
                                epochs,
                                lr
                            )
                            return

                        case _:
                            __show_invalid_message("Modelo inválido!")
                else:
                    __show_invalid_message("Learning Rate inválida!")
            else:
                __show_invalid_message("Número inválido de epochs!")
        else:
            __show_invalid_message("Batch size inválido!")


def op_explain() -> None:

    while True:
        print("\nSelecione o modelo para treinamento: ")
        print("[1] - DenseNet201")
        print("[2] - DenseNet201EncoderDecoder")

        op = int(input("Resposta: "))

        if op == 1:
            model = DenseNet201GradCam(densenet201(weights='IMAGENET1K_V1'), 2)
        elif op == 2:
            model = DenseNet201EncoderDecoder(densenet201(weights='IMAGENET1K_V1'), 2)
        else:
            __show_invalid_message("Modelo inválido")
            continue
        
        try:
            for f in os.listdir(os.path.join("../output", model.__class__.__name__)):
                if f.endswith(".pt"):
                    model.load_state_dict(torch.load(os.path.join("../output", model.__class__.__name__, f)))
                    break
        except OSError as _:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("O modelo seleciona não possui nenhum arquivo de treinamento...")
            input("Pressione ENTER para ir a seção de treinamento")
            op_train()
        
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Selecione o método de explicação:")
        print("[1] - Grad-cam")
        print("[2] - LIME")
        
        op = int(input("Resposta: "))

        if op == 1:
            get_grad_cam(model)
            return
        
        elif op == 2:
            get_lime(model)
            return
        
        else:
            __show_invalid_message("Opção inválida!")
            continue

    pass

if __name__ == "__main__":

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Escolha uma das opções:")
        print("[1] - Treinar modelos")
        print("[2] - Gerar explicações")
        print("[3] - Finalizar execução")
        
        op = int(input("\nResposta: "))

        match op:
            case 1:
                op_train()
            case 2:
                op_explain()
            case 3:
                break
            case _:
                __show_invalid_message("Operação inválida!")

