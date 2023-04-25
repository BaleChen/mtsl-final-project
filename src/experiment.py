import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd


import models
from models import Model
from dataset import get_train_trans, get_val_trans, get_train_val_loader, test_image_loader, get_classes_indices_mapping

from utils import EarlyStopping
from torch.nn.utils import clip_grad_norm_

from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch

import time
import os
import json
from tqdm import tqdm


class Experiment:
    # Class for all experiment-level implementations
    
    def __init__(self, args):
        self.args = args
        
        self.classes, _, _ = get_classes_indices_mapping(self.args.data_dir)
        self.model = Model(self.args.model, len(self.classes))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        self.train_trans = get_train_trans(image_size=224, data_aug=self.args.data_aug)
        self.val_trans = get_val_trans(image_size=224)
        self.train_loader, self.val_loader = get_train_val_loader(self.args.data_dir,
                                                        train_trans=self.train_trans,
                                                        val_trans=self.val_trans,
                                                        batch_size=self.args.batch_size,
                                                        small_sample=self.args.small_sample)
        self.test_df = pd.read_csv(self.args.data_dir + "sample_submission.csv")
        
        
    def _init_early_stopper(self):
        """
        Initiate early stopper instance.
        """
        return EarlyStopping(self.args.patience, self.args.es_delta)
    
    def _reset(self):
        pass
    
    def _train_epoch(self):
        """
        Training for one epoch. 
        """
        self.model = self.model.train()

        losses = []
        correct_predictions = 0

        for inputs, targets in tqdm(self.train_loader):
            inputs = inputs.to(self.args.device)
            targets = targets.to(self.args.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            _,preds = torch.max(outputs, dim = 1)
            loss = self.criterion(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            
            # print(f'Iteration loss: {loss.item()}')
            losses.append(loss.item())

            loss.backward()
            
            # Potentially remove it
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            

        return correct_predictions.double() / len(self.train_loader.dataset), np.mean(losses)

    
    def _eval_model(self):
        """
        Evaluate the model on the validation set.
        
        Return the accuracy and average loss
        """
        self.model = self.model.eval()

        losses = []
        correct_predictions = 0
        
        with torch.no_grad():
            cnt, acc = [0 for c in range(len(self.classes))], [0 for c in range(len(self.classes))]
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, dim=1)
                # prediction_error += torch.sum(torch.abs(targets - outputs))
                loss = self.criterion(outputs, targets)

                correct_predictions += torch.sum(preds == targets)
                
                
                for c in range(len(self.classes)):
                    acc[c] += ((preds == targets).float() @ (targets == c).float()).float().item()
                    cnt[c] += (targets == c).sum().item()
                losses.append(loss.item())
    
        for c in range(len(self.classes)):
            acc[c] /= max(cnt[c], 1)

        return correct_predictions.double() / len(self.val_loader.dataset), np.mean(losses), acc
    
    def load_checkpoint(self, path):
        """
        Load a specific checkpoint.
        """
        device = torch.device(self.args.device)
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
    def fit(self):
        """
        Full Training function (Not for hyperparameter tuning).
        """
        ######
        # TODO: Save training log (per class accuracy)
        ######
        
        if self.args.early_stop:
            early_stopper = self._init_early_stopper()
        
        best_accuracy = 0
        history = {
            "train_acc" : [],
            "train_loss" : [],
            "val_acc" : [],
            "val_loss" : [],
            "val_acc_per_class": []
        }
        
        self.model.to(self.args.device)
        
        print('=======Sanity Test=======')
        print()
        val_acc, val_loss, _ = self._eval_model()
        print()
        print(f'Val loss {val_loss} accuracy {val_acc}')
        print('=========================')
        
        print("INFO: Training Starts...")
        start = time.time()
        
        for epoch in range(self.args.epochs):

            print(f'Epoch {epoch + 1}/{self.args.epochs}')
            print('-' * 10)

            train_acc, train_loss = self._train_epoch()

            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss, acc_per_class = self._eval_model()

            print(f'Val loss {val_loss} accuracy {val_acc}')
            print()

            # # For visualization & record
            # TODO: SAVE THE HISTORY TO A JSON
            history['train_acc'].append(train_acc.item())
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc.item())
            history['val_acc_per_class'].append(acc_per_class)
            history['val_loss'].append(val_loss)
            
            if self.args.hyper_tune:
                session.report({"accuracy":val_acc.item(), "loss":val_loss.item()})

            # Save model
            if val_acc > best_accuracy:
                torch.save(self.model.state_dict(), self.args.save_dir+"best_model.pt") 
                best_accuracy = val_acc
                
            if self.args.early_stop:
                early_stopper(val_loss)
                if early_stopper.early_stop:
                    print("INFO: Early stopping criteria is met. Stop training now...")
                    break
        
        end = time.time()
        print(f"INFO: Training time: {(end-start)/60:.3f} minutes")
        
        print(f"INFO: Saving training log to {self.args.save_dir}training-log.json")
        with open(os.path.join(self.args.save_dir, "training-log.json"), "w") as f:
            json.dump(history, f, indent=4)
    
    def pred_on_test(self):
        """
        Load the best model in the training session. Use it to predict results on test set.
        Return the predictions in a DataFrame.
        """
        
        _,_,idx_to_class = get_classes_indices_mapping(self.args.data_dir)
        
        best_model = self.model
        
        best_model.to(self.args.device)
        best_model.eval()
        
        pred_list = []

        for idx, row in tqdm(self.test_df.iterrows(), total=len(self.test_df)):
            im_dir = os.path.join(self.args.data_dir, "test", row["file"])
            if os.path.isfile(im_dir):
                test_im = test_image_loader(im_dir, self.val_trans)
                test_im = test_im.to(self.args.device)

            with torch.no_grad():
                predicted = best_model(test_im).data.max(1)[1].cpu().numpy().item()
                pred_list.append({"file": row["file"], "species": idx_to_class[predicted]})
        
        
        df_pred = pd.DataFrame(pred_list)
        
        return df_pred

    def plot():
        pass
    
    
def tuning_session(config, args):
    
    args.lr = config["lr"]
    args.batch_size = config["batch_size"]
    args.save_dir = args.save_dir + f"lr{args.lr}-b{args.batch_size}/"
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    exp = Experiment(args)
    
    exp.fit()