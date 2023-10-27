import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from models.densenet import *
from tools.data_loaders import get_loaders

from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Accuracy, Precision, Recall, Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

def train(model:nn.Module, 
          dataset_path:str,
          batch_size:int,
          epochs:int,
          lr:float) -> None:

      final_json = {}

      train_loader, val_loader = get_loaders(dataset_path, batch_size)

      device = f"cuda" if torch.cuda.is_available() else "cpu"
      model = model.to(device)

      print(f"Treinando utilizando: {device}")

      # Definindo o otimizador e a loss-functions
      optimizer = optim.Adam(model.parameters(), lr=lr)
      criterion = nn.CrossEntropyLoss().to(device)

      val_metrics = {
        "accuracy": Accuracy(),
        "precision": Precision(average='weighted'),
        "recall": Recall(average='weighted'),
        "f1": (Precision(average='weighted') * Recall(average='weighted') * 2 / (Precision(average='weighted') + Recall(average='weighted'))),
        "loss": Loss(criterion)
    }

      # Criando os trainers para treinamento e validação
      trainer = create_supervised_trainer(model, optimizer, criterion, device)
      val_evaluator = create_supervised_evaluator(model, val_metrics, device)

      for name, metric in val_metrics.items():
            metric.attach(val_evaluator, name)

      train_bar = ProgressBar(desc="Treinando...")
      val_bar = ProgressBar(desc="Validando...")
      train_bar.attach(trainer)
      val_bar.attach(val_evaluator)

      # Após o termino de uma epoch do "trainer", execute o processo de validação
      @trainer.on(Events.EPOCH_COMPLETED)
      def log_validation_results(trainer):
            val_evaluator.run(val_loader)
            metrics = val_evaluator.state.metrics

            final_json[trainer.state.epoch] = metrics

            print(f"Resultados da Validação - Epoch[{trainer.state.epoch}] {final_json[trainer.state.epoch]}")

      # Definição da métrica para realizar o "checkpoint" do treinamento,
      # nesse caso, será utilizada a métrica F1.
      def score_function(engine):
            return engine.state.metrics["f1"]
      
      # Definindo e criando (se necessário) a pasta para armazenar os dados de saída
      # da aplicação
      output_folder = os.path.join("..", "output", model.__class__.__name__)

      try:
            #os.mkdir("../output")
            os.mkdir(os.path.join("..", "output"))
            os.mkdir(output_folder)
      except OSError as _:
            files_on_dir = os.listdir(output_folder)

            for file in files_on_dir:
                  if file.endswith(".pt"):
                        while True:
                              print(f"Arquivo de checkpoint encontrado! ({file})")
                              print("Deseja excluí-lo?")
                              print("[S]im || [N]ão")
                              op = input("Resposta: ")

                              if op.upper() == 'S':
                                    os.remove(os.path.join(output_folder, file))
                                    break
                              elif op.upper() == 'N':
                                    break
                              else:
                                    print("\nOperação inválida!\n")
      
      # Definindo o processo de checkpoint do modelo
      model_checkpoint = ModelCheckpoint(
            output_folder,
            require_empty=False,
            n_saved=1,
            filename_prefix=f"train",
            score_function=score_function,
            score_name="f1",
            global_step_transform=global_step_from_engine(trainer),
      )
            
      val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

      print(f"\nTreinando o modelo {model.__class__.__name__}...")

      trainer.run(train_loader, max_epochs=epochs)

      print(f"\nTrain finished for model {model.__class__.__name__}")

      # Salvando as métricas em um arquivo .json
      with open(f"{output_folder}/training_results.json", "w") as f:
            json.dump(final_json, f)

      print("\nTreinamento finalizado!")
      print(f"Os resultados do treinamento podem ser encontrado em ../output/{model.__class__.__name__}")
      input("Pressione ENTER para continuar")