import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import models


class CNN:
    def __init__(self, train_data, validation_data, test_data, batch_size):
        self.train_loader = data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
        self.validation_loader = data.DataLoader(
            validation_data, batch_size=batch_size, shuffle=False)
        self.test_loader = data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False)
        self.device = torch.device("cpu")

    def create_and_train_cnn(self, model_name, num_epochs, learning_rate, weight_decay, replications):
        soma = 0
        acc_max = 0
        for i in range(0, replications):
            model = self.create_model(model_name)
            optimizerSGD = self.create_optimizer(
                model, learning_rate, weight_decay)
            criterionCEL = self.create_criterion()
            self.train_model(model, self.train_loader, optimizerSGD, criterionCEL,
                             model_name, num_epochs, learning_rate, weight_decay, i)
            acc = self.evaluate_model(model, self.validation_loader)
            soma = soma + acc
            if acc > acc_max:
                acc_max = acc
                iter_acc_max = i
        return soma / replications, iter_acc_max

    def create_model(self, model_name):
        match model_name:
            case 'alexnet':
                model = models.alexnet(weights='DEFAULT')
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[6] = nn.Linear(
                    model.classifier[6].in_features, 2)
                return model

            case 'mobilenet_v3_large':
                model = models.mobilenet_v3_large(weights='DEFAULT')
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[3] = nn.Linear(
                    model.classifier[3].in_features, 2)
                return model

            case 'mobilenet_v3_small':
                model = models.mobilenet_v3_small(weights='DEFAULT')
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[3] = nn.Linear(
                    model.classifier[3].in_features, 2)
                return model

            case 'resnet18':
                model = models.resnet18(weights='DEFAULT')
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, 2)
                return model

            case 'resnet101':
                model = models.resnet101(weights='DEFAULT')
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, 2)
                return model

            case 'vgg11':
                model = models.vgg11(weights='DEFAULT')
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[6] = nn.Linear(
                    model.classifier[6].in_features, 2)
                return model

            case 'vgg19':
                model = models.vgg19(weights='DEFAULT')
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[6] = nn.Linear(
                    model.classifier[6].in_features, 2)
                return model

            case _:
                raise ValueError(f"Model {model_name} not suported.")

    def create_optimizer(self, model, learning_rate, weight_decay):
        update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                update.append(param)
        optimizerSGD = optim.SGD(
            update, lr=learning_rate, weight_decay=weight_decay)
        return optimizerSGD

    def create_criterion(self):
        criterionCEL = nn.CrossEntropyLoss()
        return criterionCEL

    def train_model(self, model, train_loader, optimizer, criterion, model_name, num_epochs, learning_rate, weight_decay, replicacao):
        model.to(self.device)
        min_loss = 100
        e_measures = []
        for i in (range(1, num_epochs+1)):
            train_loss = self.train_epoch(
                model, train_loader, optimizer, criterion)
            if (train_loss < min_loss):
                min_loss = train_loss
                nome_arquivo = f"modelos/{model_name}_{num_epochs}_{learning_rate}_{weight_decay}_{replicacao}.pth"
                torch.save(model.state_dict(), nome_arquivo)

    def train_epoch(self, model, trainLoader, optimizer, criterion):
        model.train()
        losses = []
        for X, y in trainLoader:
            X = X.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        return np.mean(losses)

    def evaluate_model(self, model, loader):
        total = 0
        correct = 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            output = model(X)
            _, y_pred = torch.max(output, 1)
            total += len(y)
            correct += (y_pred == y).sum().cpu().data.numpy()
        acc = correct/total
        return acc
