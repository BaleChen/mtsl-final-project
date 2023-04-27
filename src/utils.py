import os

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=3, min_delta=0.0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def prepare_save_folder(args):
    ht = "ht" if args.hyper_tune else ""
    es = "es-" if args.early_stop else ""
    ss = "ss-" if args.small_sample else ""
    aug = "aug-" if args.data_aug else ""
    ft = "ft-" if args.fine_tune else ""
    
    if not args.hyper_tune:
        directory = f"../results/{ss}{ft}{args.model}-{aug}as{args.aug_size}-ep{args.epochs}-b{args.batch_size}-lr{args.lr}-{es}p{args.patience}-dl{args.es_delta}/"
        
    else:
        abs_path = os.path.abspath("../results")
        directory = f"{abs_path}/{ss}{ht}-{args.model}-{aug}as{args.aug_size}-ep{args.epochs}-{es}p{args.patience}-dl{args.es_delta}/"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory