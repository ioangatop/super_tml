import copy
import numpy as np
import torch
import torch.nn as nn

from src.utils import args, load_data, logging
from src.models import load_model


@torch.no_grad()
def valid_step(model, criterion, val_loader):
    model.eval()
    avg_loss, avg_acc = 0.0, 0.0
    for i, (x_imgs, labels) in enumerate(val_loader):
        # forward pass
        x_imgs, labels = x_imgs.to(args.device), labels.to(args.device)
        outputs = model(x_imgs)
        loss = criterion(outputs, labels)
        # gather statistics
        avg_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        avg_acc += torch.sum(preds == labels.data).item()
    return {'loss': avg_loss / len(val_loader), 'accuracy': avg_acc / len(val_loader.dataset)}


def train_step(model, criterion, optimizer, train_loader):
    model.train()
    avg_loss, avg_acc = 0.0, 0.0
    for i, (x_imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # forward pass
        x_imgs, labels = x_imgs.to(args.device), labels.to(args.device)
        probs = model(x_imgs)
        loss = criterion(probs, labels)
        # back-prop
        loss.backward()
        optimizer.step()
        # gather statistics
        avg_loss += loss.item()
        _, preds = torch.max(probs, 1)
        avg_acc += torch.sum(preds == labels.data).item()
    return {'loss': avg_loss / len(train_loader), 'accuracy': avg_acc / len(train_loader.dataset)}


def opt_selection(model, opt=args.opt):
    if opt=='Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=0.0001)
    elif opt=='Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-5)
    elif opt=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    else:
        raise NotImplementedError
    return optimizer



def train_model(model_name='densenet121', opt='Adagrad', dataset='iris', writer=None):
    train_loader, val_loader, test_loader = load_data(dataset)

    # Model selection
    model = load_model(model_name)

    # Optimizer
    optimizer = opt_selection(model, opt)

    # Loss Criterion
    criterion = nn.CrossEntropyLoss()

    best_train, best_val = 0.0, 0.0
    for epoch in range(1, args.epochs+1):
        # Train and Validate
        train_stats = train_step(model, criterion, optimizer, train_loader)
        valid_stats = valid_step(model, criterion, val_loader)

        # Logging
        logging(epoch, train_stats, valid_stats, writer)

        # Keep best model
        if valid_stats['accuracy'] > best_val or (valid_stats['accuracy']==best_val and train_stats['accuracy']>=best_train):
            best_train  = train_stats['accuracy']
            best_val    = valid_stats['accuracy']
            best_model_weights = copy.deepcopy(model.state_dict())

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_weights)
    test_stats = valid_step(model, criterion, test_loader)

    print('\nBests Model Accuracies: Train: {:4.2f} | Val: {:4.2f} | Test: {:4.2f}'.format(best_train, best_val, test_stats['accuracy']))

    return model


if __name__ == "__main__":
    pass
