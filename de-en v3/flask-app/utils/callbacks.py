import tensorflow as tf
import time
import os
import shutil




class My_Callback(tf.keras.callbacks.Callback):
    
    def __init__(self, timeout, num_best_models=1, monitor='loss', mode='min', 
                 log_file='logs.txt', stop_file='stop_training.txt', checkpoint_path=None, 
                 save_weights_only=False, reload_best_weights=True, report_freq=50, verbose=1):
        super(My_Callback, self).__init__()
        self.timeout = timeout                          # stop training after 'timeout' seconds
        self.num_best_epochs = num_best_models          # how many models to keep
        self.checkpoint_path = checkpoint_path          # path to the checkpoints
        self.log_file = log_file                        # path to the log file
        self.stop_file = stop_file                      # path to the file containing instruction to stop training
        self.monitor = monitor                          # metric to monitor for saving 
        self.mode = mode                                # 'min' or 'max', whatever is better for the metric to monitor
        self.save_weights_only = save_weights_only      # save only weights or entire model
        self.reload_best_weights = reload_best_weights  # whether or not to reload the best weights after training
        self.report_freq = report_freq                  # the number of training steps/batches after which to report the metrics  
        self.verbose = verbose                          # 0: no reports, 
                                                        # 1: report after every epoch and after 'report_freq' batches
                                                        # >=2: only report after every epoch
        
        
    
    def on_train_begin(self, logs):
        
        # initializing the 'History'
        self.model.History = {'loss': [], 'w_loss': [], 'accuracy': [], 'w_accuracy': [], 'factor': [], 'lr': []} 
        
        # set the training start time
        self.training_start_time = time.time()
        
        #  a sorted list to store the best 5 epochs, 
        # e.g. [(epoch=2, w_acc=0.2), ... , (epoch=6, w_acc=0.45)] 
        self.model.top_epochs = self.num_best_epochs*[(0,0)] 
        
        # remove old checkpoints
        shutil.rmtree(self.checkpoint_path, ignore_errors=True)  
        
        # store the path to the checkpoints
        self.model.checkpoint_path = self.checkpoint_path            
        
        # creating the log file and save the starting time
        with open(self.log_file, 'w') as f:
            f.write('Training begins on ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + '\n')
                
        # initiating a text file allowing us to stop the training 
        # by changing its content from 'False' to 'True'
        with open(self.stop_file, 'w') as f:              
            f.write('False')                                   
        
        
    def on_epoch_begin(self, epoch, logs):
        
        # reset the epoch start time
        self.start = time.time()
        
        # saving the epoch number in the log file
        with open(self.log_file, 'a') as f:
            f.write(f'\nEpoch {epoch+1}\n')

        
        
    def on_train_batch_begin(self, batch, logs):
        pass
        

        
    def on_train_batch_end(self, batch, logs):
        
        # check if it's time to report (every 'report_freq' batches)
        if batch % self.report_freq == 0:            
            # save the metrics to the 'History' list of the model
            self.model.History['loss'].append(logs['loss'])
            self.model.History['w_loss'].append(logs['w_loss'])
            self.model.History['accuracy'].append(logs['accuracy'])
            self.model.History['w_accuracy'].append(logs['w_accuracy'])
            self.model.History['factor'].append(logs['factor'])
            #self.model.History['lr'].append(optimizer._fallback_apply_state.values()[0])
            
            # if verbose == 1, save the metrics in the log file
            if  self.verbose == 1:                
                with open(self.log_file, 'a') as f:
                    f.write(f"Batch {batch:6d}{'':6}Loss {logs['loss']:.4f}{'':6}w_Loss {logs['w_loss']:.4f}{'':6}Accuracy {logs['accuracy']:.4f}{'':6}w_Accuracy {logs['w_accuracy']:.4f}{'':6}Factor {logs['factor']*100:.2f}%\n")
    
        
    
    def on_epoch_end(self, epoch, logs):
        
        # compute the time passed during the epoch
        time_since_start = int(time.time() - self.training_start_time)
        days = time_since_start // (24*60*60)
        hours = (time_since_start - 24*60*60*days) // (60*60)
        minutes = (time_since_start - 24*60*60*days - 60*60*hours) // 60
        seconds = time_since_start - 24*60*60*days - 60*60*hours - 60*minutes
        
        # if verbose >= 1, save the metrics in the log file
        if self.verbose >= 1:
            with open(self.log_file, 'a') as f:
                f.write('Training results:\n')
                f.write(f"Epoch {epoch+1:5d}{'':6}Loss {logs['loss']:.4f}{'':6}w_Loss {logs['w_loss']:.4f}{'':6}Accuracy {logs['accuracy']:.4f}{'':6}w_Accuracy {logs['w_accuracy']:.4f}{'':6}Factor {logs['factor']*100:.2f}%\n")
                f.write("Validation results:\n")
                f.write(f"Epoch {epoch+1:5d}{'':6}Loss {logs['val_loss']:.4f}{'':6}w_Loss {logs['val_w_loss']:.4f}{'':6}Accuracy {logs['val_accuracy']:.4f}{'':6}w_Accuracy {logs['val_w_accuracy']:.4f}{'':6}Factor {logs['val_factor']*100:.2f}%\n")
                f.write(f"Time taken for 1 Epoch: {time.time() - self.start:.2f} secs\n")
                f.write(f"Time since start {days} days {hours} hours {minutes} mins {seconds} secs\n")
        
        # checking if the current epoch has a better validation 'metric' than the worst stored epoch 
        metric = logs[self.monitor]
        if (self.mode=='max' and metric > self.model.top_epochs[0][1]) or (self.mode=='min' and metric < self.model.top_epochs[0][1]):
            
            # if so, replace the worst stored epoch and sort the list again
            (old_epoch, old_metric) = self.model.top_epochs[0]
            self.model.top_epochs[0] = (epoch+1, metric)
            self.model.top_epochs = sorted(self.model.top_epochs, key=lambda x: x[1])
            if self.mode=='min':
                self.model.top_epochs.reverse()
            
            # also delete the checkpoint of the worst stored epoch and save a new checkpoint           
            try:
                shutil.rmtree(os.path.join(self.checkpoint_path, f"{old_epoch:04d}-{old_metric:.3f}"))
                if self.verbose >= 1:
                    with open(self.log_file, 'a') as f:
                        f.write(f'\nDeleted checkpoint {old_epoch:04d}-{old_metric:.3f}\n')
            except:
                pass
            if self.save_weights_only:
                self.model.save_weights(os.path.join(self.checkpoint_path, f"{epoch+1:04d}-{metric:.3f}"))
            else:
                self.model.save(os.path.join(self.checkpoint_path, f"{epoch+1:04d}-{metric:.3f}"))
          
        # stop training after self.timeout seconds       
        if time_since_start > self.timeout : 
            self.model.stop_training = True

        # stop training if the stop file contains 'True'
        with open(self.stop_file, 'r') as f:
            stop_training = f.read()
            if stop_training.lower().strip() == 'true':
                self.model.stop_training = True
                
                
            
    def on_train_end(self, logs):
        
        # if set, reload the best weights
        if self.reload_best_weights:            
            epoch, metric = self.model.top_epochs[-1]
            path = os.path.join(self.checkpoint_path, f"{epoch:04d}-{metric:.3f}", 'variables/variables')
            self.model.load_weights(path)
            with open(self.log_file, 'a') as f:
                f.write(f'\nLoaded weights of epoch {epoch}.\n')
                f.write('Training ends on ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + '\n')
 

