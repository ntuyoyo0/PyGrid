from syft.workers.node_client import NodeClient
import logging
import sys
import asyncio
import torch.nn as nn
import torch.nn.functional as F
import pdb

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
    
    
async def fit_model_on_worker(
    worker: websocket_client.WebsocketClientWorker,
    traced_model: torch.jit.ScriptModule,
    batch_size: int,
    curr_round: int,
    max_nr_batches: int,
    lr: float,
):
    """Send the model to the worker and fit the model on the worker's training data.

    Args:
        worker: Remote location, where the model shall be trained.
        traced_model: Model which shall be trained.
        batch_size: Batch size of each training step.
        curr_round: Index of the current training round (for logging purposes).
        max_nr_batches: If > 0, training on worker will stop at min(max_nr_batches, nr_available_batches).
        lr: Learning rate of each training step.

    Returns:
        A tuple containing:
            * worker_id: Union[int, str], id of the worker.
            * improved model: torch.jit.ScriptModule, model after training at the worker.
            * loss: Loss on last training batch, torch.tensor.
    """
    print("Enter Iteration %s: %s" % (curr_round, worker.id))
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        max_nr_batches=max_nr_batches,
        epochs=1,
        optimizer="SGD",
        optimizer_args={"lr": lr},
    )
    train_config.send(worker)
    
    print("Start Iteration %s: %s" % (curr_round, worker.id))
#     pdb.set_trace()
    
    loss = await worker.async_fit(dataset_key="mnist", return_ids=[0])
    
    print("Iteration %s: %s loss: %s" % (curr_round, worker.id, loss))
#     pdb.set_trace()
    
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
    
async def main():
    hook = sy.TorchHook(torch)
    device = torch.device("cpu")
    model = Net().to(device)
    traced_model = torch.jit.trace(model, torch.zeros([1, 1, 28, 28], dtype=torch.float).to(device))
    batch_size = 64
    lr = 0.1
    learning_rate = lr
    optimizer_args = {"lr" : lr}
    train_config = sy.TrainConfig(model=traced_model,
                      loss_fn=loss_fn,
                      optimizer="SGD",
                      batch_size=batch_size,
                      optimizer_args=optimizer_args,
                      epochs=1,
                      shuffle=True)

#    alice = NodeClient(hook, "ws://localhost:6666" , id="alice")
#    bob = NodeClient(hook, "ws://localhost:6667" , id="bob")
#    charlie = NodeClient(hook, "ws://localhost:6668", id="charlie")
#    testing = NodeClient(hook, "ws://localhost:6669" , id="testing")
    
    kwargs_websocket = {"hook": hook, "verbose": True, "host": "0.0.0.0"}
    alice = websocket_client.WebsocketClientWorker(id="alice", port=6666, **kwargs_websocket)
    bob = websocket_client.WebsocketClientWorker(id="bob", port=6667, **kwargs_websocket)
    charlie = websocket_client.WebsocketClientWorker(id="charlie", port=6668, **kwargs_websocket)
    testing = websocket_client.WebsocketClientWorker(id="testing", port=6669, **kwargs_websocket)
    
    worker_list = [alice, bob, charlie]
    
    for epoch in range(50):
        
#         models = {}
#         loss_values = {}
        logger.info("Training round %s/%s", epoch, 50)
        
        results = await asyncio.gather(
            *[
                fit_model_on_worker(
                    worker=worker,
                    traced_model=traced_model,
                    batch_size=32,
                    curr_round=epoch,
                    max_nr_batches=10,
                    lr=0.1,
                )
                for worker in worker_list
            ]
        )
        
        models = {}
        loss_values = {}

        
        for worker_id, worker_model, worker_loss in results:
            if worker_model is not None:
                models[worker_id] = worker_model
                loss_values[worker_id] = worker_loss
#         for worker in worker_list:
#             worker_id, model, loss = fit_model_on_worker(worker, traced_model, 64, epoch, 0, 0.1)
#             models[worker_id] = model
#             loss_values[worker_id] = loss
#             print("-" * 50)
#             print("Iteration %s: %s loss: %s" % (epoch, worker.id, loss))
        
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
            
        # decay learning rate
        learning_rate = max(0.98 * learning_rate, lr * 0.01)
    
     
if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)
    
    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())
    
    asyncio.get_event_loop().run_until_complete(main())
#     main()
    
