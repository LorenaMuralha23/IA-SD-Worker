import json
import socket
import os
from queue import Queue
from multiprocessing import Process, active_children
from cnn import CNN
import torch
from torchvision import datasets
from torchvision.transforms import v2
import time
from main import Main


task_queue = Queue()
main = Main()

data_transforms = main.define_transforms(224, 224)
train_data, validation_data, test_data = main.read_images(data_transforms)
cnn = CNN(train_data, validation_data, test_data, 8)


def receiveTask(receivedJson):
    if 'data' in receivedJson and isinstance(receivedJson.get('data'), list):
        for combination in receivedJson.get('data'):
            processTask(combination)
    else:
        print("O JSON recebido não contém um campo 'data' válido.")


def sendJson(jsonToSend):
    return jsonToSend


def createJson(status, machine_id):
    try:

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]

        num_cores = os.cpu_count()

        createdJson = {
            "machine_id": machine_id,
            "ip_address": ip_address,
            "port": 5000,
            "status": status,
            "num_cores": num_cores
        }

        createdJson = json.dumps(createdJson, indent=4)

        return createdJson

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=4)


def process_task_wrapper(cnn, repl, mn, epochs, lr, wd):
    from workerServer import sendToGroup
    main = Main()

    taskResult = main.processTask(cnn, repl, mn, epochs, lr, wd)

    sendToGroup(taskResult)


def processTask(combination):

    repl = combination.get('replications')
    mn = combination.get('model_name')
    epochs = combination.get('epochs')
    lr = combination.get('learning_rate')
    wd = combination.get('weight_decay')
    task = Process(target=process_task_wrapper,
                   args=(cnn, repl, mn, epochs, lr, wd))
    task.start()
    print(f"Processo iniciado: PID={task.pid}, Nome={task.name}")
    print(f"Processos ativos no momento: {len(active_children())}")


# if __name__ == "__main__":
#     main()
