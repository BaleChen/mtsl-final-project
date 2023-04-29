# This script is used for training ensemble models.

import argparse
from models import Model

from dataset import get_train_trans, get_val_trans, get_train_val_loader, test_image_loader, get_classes_indices_mapping
from experiment import generate_prediction
from utils import EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
from copy import deepcopy
import pickle
from tqdm import tqdm

def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
#   scheduler,
  device,
  n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for inputs, targets in tqdm(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        _,preds = torch.max(outputs, dim = 1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        # print(f'Iteration loss: {loss.item()}')
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0
    output_all = []
    target_all = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            output_all.append(outputs)
            target_all.append(targets)
            
            _, preds = torch.max(outputs, dim=1)
            # prediction_error += torch.sum(torch.abs(targets - outputs))
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses), (output_all, target_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ensemble-list", type=str, required=True)
    parser.add_argument("--small-sample", dest="small_sample", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    data_dir = "../data/"
    
    model_names = args.ensemble_list.split(",")
    criterion = torch.nn.CrossEntropyLoss()
    
    train_trans = get_train_trans(image_size=224, data_aug=True)
    val_trans = get_val_trans(image_size=224)
    
    train_loader, val_loader = get_train_val_loader(data_dir,
                                                    val_ratio=0.2,
                                                    train_trans=train_trans,
                                                    val_trans=val_trans,
                                                    batch_size=32,
                                                    small_sample=args.small_sample,
                                                    augment_size=0.0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on", device)
    
    model_stack = []
    
    for model_name in model_names:
        model = Model(model_name, 12)
        print(f"INFO: Training {model_name}")
        model.to(device)
        
        if model_name == "wide_resnet101_2" :
            optimizer = optim.Adam(model.parameters(), lr=1.65e-5)
        elif model_name == "wide_resnet50_2":
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
        else:
            optimizer = optim.Adam(model.parameters(), lr=3e-5)
        
        early_stopper = EarlyStopping(5, 0.0)
        
        best_accuracy = 0

        print('=======Sanity Test=======')
        val_acc, val_loss, _ = eval_model(
            model,
            val_loader,
            criterion,
            device,
            len(val_loader.dataset)
        )
        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()
        print('========================')

        
        for epoch in range(EPOCHS):

            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)

            train_acc, train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                # scheduler,
                device,
                len(train_loader.dataset)
            )

            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss, _ = eval_model(
                model,
                val_loader,
                criterion,
                device,
                len(val_loader.dataset)
            )

            print(f'Val loss {val_loss} accuracy {val_acc}')
            print()

            # # For visualization
            # history['train_acc'].append(train_acc)
            # history['train_loss'].append(train_loss)
            # history['val_acc'].append(val_acc)
            # history['val_loss'].append(val_loss)

            # Save model
            if val_acc > best_accuracy:
                model_copy = deepcopy(model)
                torch.save(model.state_dict(), f"../results/ensemble/best{model_name}.pt") 
                best_accuracy = val_acc
            
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print("INFO: Early stopping criteria is met. Stop training now...")
                break
    
        model_stack.append((model_name, model_copy))
    
    y_hats = []
    for name, model in model_stack:
        val_acc, val_loss, output_target = eval_model(
            model,
            val_loader,
            criterion,
            device,
            len(val_loader.dataset)
        )
        
        print(f"INFO: Loaded trained {name}, val_loss {val_loss}, val_acc {val_acc}")
        
        outputs, targets = output_target
        output_all = torch.cat(outputs, dim=0)
        target_all = torch.cat(targets, dim=0)
        
        y_hats.append(output_all)
    
    y_hats = torch.cat(y_hats, dim=1).detach().cpu().numpy()
    y_true = target_all.detach().cpu().numpy()
    
    logistic_model = LogisticRegression()
    logistic_model.fit(y_hats, y_true)
    
    y_stack_predictions = logistic_model.predict(y_hats)
    acc = np.sum(y_stack_predictions == y_true) / y_true.shape[0]
    
    print(f"INFO: Stacked prediction acc {acc}")
    
    filename = 'stack_head.pt'
    pickle.dump(logistic_model, open(filename, 'wb'))
    print("INFO: Stacking head saved.")
    
    generate_prediction(model_stack,
                        device=device,
                        data_dir = "../data/",
                        save_dir = "../results/ensemble/results.csv",
                        ensemble=True,
                        head=logistic_model)