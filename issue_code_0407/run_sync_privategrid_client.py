from syft.workers.node_client import NodeClient
import logging
import sys
import asyncio
import torch.nn as nn
import torch.nn.functional as F

from syft.workers import websocket_client
from syft.frameworks.torch.fl import utils

from multiprocessing import Process
import argparse
import os
import syft as sy
import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms

LOG_INTERVAL = 25
logger = logging.getLogger("run_websocket_client")

@torch.jit.script
def loss_fn(pred, target):
    return F.nll_loss(input=pred, target=target)


# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def fit_model_on_worker (worker, traced_model, batch_size, curr_round, max_nr_batches, lr):
    
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        epochs=1,
        optimizer="SGD",
        optimizer_args={"lr": lr},
    )
    
    train_config.send(worker)
    loss = worker.fit(dataset_key="mnist", return_ids=[0])
    model = train_config.model_ptr.get().obj
    return worker.id, model, loss

def evaluate_model_on_worker(
    model_identifier, 
    worker, 
    dataset_key, model, 
    nr_bins, 
    batch_size, 
    device, 
    print_target_hist=False):
    
    model.eval()
    train_config = sy.TrainConfig(
        batch_size=batch_size, model=model, loss_fn=loss_fn, optimizer_args=None, epochs=1
    )
    train_config.send(worker)
    
    result = worker.evaluate(
        dataset_key=dataset_key,
        return_histograms=True,
        nr_bins=nr_bins,
        return_loss=True,
        return_raw_accuracy=True,
        device=device,
    )
    
    test_loss = result["loss"]
    correct = result["nr_correct_predictions"]
    len_dataset = result["nr_predictions"]
    hist_pred = result["histogram_predictions"]
    hist_target = result["histogram_target"]
    
    if print_target_hist:
        logger.info("Target histogram: %s", hist_target)
    percentage_0_3 = int(100 * sum(hist_pred[0:4]) / len_dataset)
    percentage_4_6 = int(100 * sum(hist_pred[4:7]) / len_dataset)
    percentage_7_9 = int(100 * sum(hist_pred[7:10]) / len_dataset)
    logger.info(
        "%s: Percentage numbers 0-3: %s%%, 4-6: %s%%, 7-9: %s%%",
        model_identifier,
        percentage_0_3,
        percentage_4_6,
        percentage_7_9,
    )

    logger.info(
        "%s: Average loss: %s, Accuracy: %s/%s (%s%%)",
        model_identifier,
        f"{test_loss:.4f}",
        correct,
        len_dataset,
        f"{100.0 * correct / len_dataset:.2f}",
    )
    
def main():
    hook = sy.TorchHook(torch)
    device = torch.device("cpu")
    model = Net().to(device)
    traced_model = torch.jit.trace(model, torch.zeros([1, 1, 28, 28], dtype=torch.float).to(device))
    batch_size = 64
    optimizer_args = {"lr" : 0.1}
    train_config = sy.TrainConfig(model=traced_model,
                      loss_fn=loss_fn,
                      optimizer="SGD",
                      batch_size=batch_size,
                      optimizer_args=optimizer_args,
                      epochs=1,
                      shuffle=True)

    alice = NodeClient(hook, "ws://localhost:6666" , id="alice")
    bob = NodeClient(hook, "ws://localhost:6667" , id="bob")
    charlie = NodeClient(hook, "ws://localhost:6668", id="charlie")
    testing = NodeClient(hook, "ws://localhost:6669" , id="testing")
    
    worker_list = [alice, bob, charlie]
    
    for epoch in range(50):
        
        models = {}
        loss_values = {}
        
        for worker in worker_list:
            worker_id, model, loss = fit_model_on_worker(worker, traced_model, 64, epoch, 0, 0.1)
            models[worker_id] = model
            loss_values[worker_id] = loss
#             traced_model = model
            print("-" * 50)
            print("Iteration %s: %s loss: %s" % (epoch, worker.id, loss))
        
        ## FedAvg
        traced_model = utils.federated_avg(models)
        
        if epoch%5 == 0 or epoch == 49:
            evaluate_model_on_worker(
                model_identifier="Federated model",
                worker=testing,
                dataset_key="mnist_testing",
                model=traced_model,
                nr_bins=10,
                batch_size=64,
                device=device,
                print_target_hist=False,
            )
            
    
     
if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)
    
    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())
    
    main()
    