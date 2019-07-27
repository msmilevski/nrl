import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time
from sklearn.metrics import average_precision_score

from storage_utils import save_statistics


class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 test_data, learning_rate, weight_decay_coefficient, device, continue_from_epoch=-1):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()

        self.experiment_name = experiment_name
        self.model = network_model
        self.model.reset_parameters()
        self.device = device

        if torch.cuda.device_count() > 1:
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
        else:
            self.model.to(self.device)  # sends the model from the cpu to the gpu
        # re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optim.Adam(self.parameters(), amsgrad=False, lr=learning_rate,
                                    weight_decay=weight_decay_coefficient)
        # Generate the directory names
        self.experiment_folder = os.path.abspath('experiments/' + experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs)
        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_aps = 0.
        self.no_improvement_counter = 0

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = num_epochs
        self.criterion = nn.BCELoss().to(self.device)  # send the loss computation to the GPU
        if continue_from_epoch == -2:
            try:
                self.best_val_model_idx, self.best_val_model_aps, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  # reload existing model from epoch and return best val model index
                # and the best val acc of that model
                self.starting_epoch = self.state['current_epoch_idx']
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.starting_epoch = 0
                self.state = dict()

        elif continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_aps, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']
        else:
            self.starting_epoch = 0
            self.state = dict()

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def run_train_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels

        # print(type(x))

        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
                device=self.device)  # send data to device as torch tensors

        # x = x.to(self.device)
        y = y.to(self.device)

        out = self.model.forward(x)  # forward the data in the model
        out = out.type(torch.float)
        loss = self.criterion(input=out, target=y)  # compute loss

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        # _, predicted = torch.max(out.data, 1) # get argmax of predictions
        y_true = y.cpu().numpy()
        predicted = out.detach().cpu().numpy()
        average_precision = average_precision_score(y_true=y_true, y_score=predicted)
        # accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        return loss.data.detach().cpu().numpy(), average_precision

    def run_evaluation_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
                device=self.device)  # convert data to pytorch tensors and send to the computation device

        # x = x.to(self.device)
        y = y.to(self.device)

        out = self.model(x)  # forward the data in the model
        out = out.type(torch.float)
        loss = self.criterion(out, y)  # compute loss
        # _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        predicted = out.detach().cpu().numpy()
        y_true = y.cpu().numpy()
        average_precision = average_precision_score(y_true=y_true, y_score=predicted)
        # accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        return loss.data.detach().cpu().numpy(), average_precision

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        state['network'] = self.state_dict()  # save network parameter and other variables.
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx'], state['best_val_model_aps'], state

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"train_aps": [], "train_loss": [], "val_aps": [],
                        "val_loss": [], "curr_epoch": []}  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_aps": [], "train_loss": [], "val_aps": [], "val_loss": []}

            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for idx, batch in enumerate(self.train_data):  # get data batches
                    desc_1_batch = batch['desc1'].to(self.device)
                    desc_2_batch = batch['desc2'].to(self.device)
                    img_1_batch = batch['image_1'].to(self.device)
                    img_2_batch = batch['image_2'].to(self.device)
                    y = batch['target']
                    x = [desc_1_batch, img_1_batch, desc_2_batch, img_2_batch]
                    loss, aps = self.run_train_iter(x=x, y=y)  # take a training iter step
                    current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                    current_epoch_losses["train_aps"].append(aps)  # add current iter acc to the train acc list
                    pbar_train.update(1)
                    pbar_train.set_description("loss: {:.4f}, average precision score: {:.4f}".format(loss, aps))

            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  # create a progress bar for validation
                for idx, batch in enumerate(self.val_data):  # get data batches
                    desc_1_batch = batch['desc1'].to(self.device)
                    desc_2_batch = batch['desc2'].to(self.device)
                    img_1_batch = batch['image_1'].to(self.device)
                    img_2_batch = batch['image_2'].to(self.device)
                    y = batch['target']
                    x = [desc_1_batch, img_1_batch, desc_2_batch, img_2_batch]
                    loss, aps = self.run_evaluation_iter(x=x, y=y)  # run a validation iter
                    current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                    current_epoch_losses["val_aps"].append(aps)  # add current iter acc to val acc lst.
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("loss: {:.4f}, average precision score: {:.4f}".format(loss, aps))
            val_mean_aps = np.mean(current_epoch_losses['val_aps'])
            if val_mean_aps > self.best_val_model_aps:  # if current epoch's mean val acc is greater than the saved best val acc then
                self.best_val_model_aps = val_mean_aps  # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx
                self.no_improvement_counter = 0
            else:
                self.no_improvement_counter += 1

            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(
                    value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            total_losses['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (
                                    self.starting_epoch != 0 or i > 0) else False)  # save statistics to stats file.

            train_batch_losses = {"train_aps": [], "train_loss": []}
            val_batch_losses = {"val_aps": [], "val_loss": []}

            train_batch_losses["train_loss"] = current_epoch_losses["train_loss"]
            train_batch_losses["train_aps"] = current_epoch_losses["train_aps"]

            val_batch_losses["val_loss"] = current_epoch_losses["val_loss"]
            val_batch_losses["val_aps"] = current_epoch_losses["val_aps"]

            save_statistics(experiment_log_dir=self.experiment_logs, filename='train_summary.csv',
                            stats_dict=train_batch_losses, current_epoch=i, save_full_dict=True,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False)

            save_statistics(experiment_log_dir=self.experiment_logs, filename='val_summary.csv',
                            stats_dict=val_batch_losses, current_epoch=i, save_full_dict=True,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False)
            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_aps'] = self.best_val_model_aps
            self.state['best_val_model_idx'] = self.best_val_model_idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx, state=self.state)
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx='latest', state=self.state)

            # Early stopping
            if self.no_improvement_counter == 20:
                break

        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        # load best validation model
                        model_save_name="train_model")
        current_epoch_losses = {"test_aps": [], "test_loss": []}  # initialize a statistics dict
        with tqdm.tqdm(total=len(self.test_data)) as pbar_test:  # ini a progress bar
            for idx, batch in enumerate(self.test_data):  # sample batch
                desc_1_batch = batch['desc1'].to(self.device)
                desc_2_batch = batch['desc2'].to(self.device)
                img_1_batch = batch['image_1'].to(self.device)
                img_2_batch = batch['image_2'].to(self.device)
                y = batch['target']
                x = [desc_1_batch, img_1_batch, desc_2_batch, img_2_batch]
                loss, aps = self.run_evaluation_iter(x=x,
                                                     y=y)  # compute loss and accuracy by running an evaluation step
                current_epoch_losses["test_loss"].append(loss)  # save test loss
                current_epoch_losses["test_aps"].append(aps)  # save test accuracy
                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description(
                    "loss: {:.4f}, average precision score: {:.4f}".format(loss,
                                                                           aps))  # update progress bar string output

        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)

        return total_losses, test_losses
