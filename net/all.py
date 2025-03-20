import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from net.lidar_feature_extractor_houston2018 import lidar_e
from net.hsi_feature_extractor_houston2018 import hsi_e
from net.classify import classify

from net.routing_modules import RoutingModule

from parameter import *
from utility import output_metric
from sklearn.metrics import classification_report, accuracy_score


class DCMNet(object):
    def __init__(self):
        # Build Models
        self.args = args
        self.lidar_enc = lidar_e()
        self.hsi_enc = hsi_e()
        self.cla = classify()
        self.routing_module = RoutingModule(args)
        self.criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.lidar_enc = self.lidar_enc.to(device)
            self.hsi_enc = self.hsi_enc.to(device)
            self.routing_module = self.routing_module.to(device)
            self.cla = self.cla.to(device)



        params = list(self.hsi_enc.parameters())
        params += list(self.lidar_enc.parameters())
        params += list(self.routing_module.parameters())
        params += list(self.cla.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.99, threshold=0.0002,
                                                              verbose=True, patience=2)

    def state_dict(self):
        state_dict = [self.lidar_enc.state_dict(), self.hsi_enc.state_dict(), self.routing_module.state_dict(),
                      self.cla.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.lidar_enc.load_state_dict(state_dict[0])
        self.hsi_enc.load_state_dict(state_dict[1])
        self.routing_module.load_state_dict(state_dict[2])
        self.cla.load_state_dict(state_dict[3])

    def train_start(self):
        self.lidar_enc.train()
        self.hsi_enc.train()
        self.routing_module.train()
        self.cla.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.lidar_enc.eval()
        self.hsi_enc.eval()
        self.routing_module.eval()
        self.cla.eval()

    def enc(self, hsi, lidar):
        """Compute the image and caption embeddings
        """
        # Forward
        e_lidar = self.lidar_enc(lidar)
        e_hsi = self.hsi_enc(hsi)
        return e_hsi, e_lidar

    def cla(self, f):
        """Compute the image and caption embeddings
        """
        # Forward
        c = self.cla(f)
        return c

    def cal_acc(self, epoch):
        count = 0
        self.val_start()
        total_loss = 0
        n = 0
        with torch.no_grad():
            for hsi, lidar, tr_labels in test_loader:
                hsi = hsi.to(device)
                lidar = lidar.to(device)
                tr_labels = tr_labels.to(device)
                e_hsi, e_lidar = self.enc(hsi, lidar)
                f, paths = self.routing_module(e_hsi, e_lidar)
                results = self.cla(f[0])
                loss = self.criterion(results, tr_labels)
                outputs = np.argmax(results.detach().cpu().numpy(), axis=1)
                tr_labels = tr_labels.detach().cpu().numpy()
                n += 1
                total_loss += loss.item()
                if count == 0:
                    y_pred_test = outputs
                    gty = tr_labels
                    count = 1
                else:
                    y_pred_test = np.concatenate((y_pred_test, outputs))  #
                    gty = np.concatenate((gty, tr_labels))

        acc1 = accuracy_score(gty, y_pred_test)
        OA2, AA_mean2, Kappa2, AA2 = output_metric(gty, y_pred_test)
        classification = classification_report(gty, y_pred_test, digits=4)
        print(classification)
        print("OA2=", OA2)
        print("AA_mean2=", AA_mean2)
        print("Kappa2=", Kappa2)
        print("AA2=", AA2)
        print(
            'Testing stage: [Epoch: %d] [loss avg: %.4f]   [current loss: %.4f]' % (
            epoch + 1, total_loss / n, loss.item()),
            ' acc: ', acc1)
        self.scheduler.step(total_loss / n)
        return acc1


    def train(self, epoch):
        """One training step given images and captions.
        """
        n = 0
        total_loss = 0
        count = 0
        for i, (hsi, lidar, tr_labels) in enumerate(train_loader):
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            tr_labels = tr_labels.to(device)

            self.optimizer.zero_grad()
            e_hsi, e_lidar = self.enc(hsi, lidar)
            n += 1

            f, paths = self.routing_module(e_hsi, e_lidar)
            results = self.cla(f[0])

            loss = self.criterion(results, tr_labels)

            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            outputs = np.argmax(results.detach().cpu().numpy(), axis=1)
            tr_labels = tr_labels.detach().cpu().numpy()
            n += 1
            total_loss += loss.item()
            if count == 0:
                y_pred_test = outputs
                gty = tr_labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))  #
                gty = np.concatenate((gty, tr_labels))
        acc1 = accuracy_score(gty, y_pred_test)
        print('Training stage: [Epoch: %d] [loss avg: %.4f]   [current loss: %.4f]' % (
            epoch + 1, total_loss / n, loss.item()), ' acc: ', acc1)

        return loss.item()
