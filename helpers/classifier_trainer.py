import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import abc
import matplotlib.pyplot as plt
import SemesterProject2.helpers.pytorch_utils as ptu

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from SemesterProject2.helpers.utils import create_param_dir

CIFAR10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_data_loader(batch_size=16, root_dir="data/cifar10", **kwargs):
    """
    This function should be called at root directory of the project.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    num_workers = kwargs.get("num_workers", 0)
    train_set = CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_set = CIFAR10(root=root_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


class ClassifierTrainer(abc.ABC):
    def __init__(self, model, train_loader, test_loader, train_params, log_params):
        """
        train_params: batch_size, epochs, opt_name, (opt_params: lr), loss, grad_clamp_val, num_classes, if_notebook
        log_params: log_dir, param_save_dir (should be created in main)
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_params = train_params
        self.log_params = log_params

        # tags: train_loss, eval_loss, (epoch level): train_acc, eval_acc
        self.writer = SummaryWriter(self.log_params["log_dir"])

        self.opt = self.train_params["opt_name"](self.model.parameters(), **self.train_params["opt_params"])
        self.loss = self.train_params["loss"]()
        self.timestep_counters = {"train": 0, "eval": 0, "epoch": 0}

    def train_(self):
        self.model.train()

        if self.train_params["if_notebook"]:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(self.train_loader, total=len(self.train_loader), desc="training", leave=False)

        loss_avg = 0
        acc = 0
        num_samples = 0
        for i, (X, y) in enumerate(pbar):
            # ###
            # # Debug only
            # if i > 1:
            #     break
            # ###
            X = X.float().to(ptu.device)
            y = y.long().to(ptu.device)  # (B,)
            y_pred = self.model(X)  # (B, num_classes)

            self.opt.zero_grad()
            loss = self.loss(y_pred, y)
            loss.backward()
            if "grad_clamp_val" in self.train_params:
                nn.utils.clip_grad_value_(self.model.parameters(), self.train_params["grad_clamp_val"])
            self.opt.step()

            # logging
            with torch.no_grad():
                loss_avg += loss.item() * X.shape[0]
                y_pred_labels = torch.argmax(y_pred, dim=-1)
                acc += (y_pred_labels == y).sum()
                num_samples += X.shape[0]

            self.writer.add_scalar("train_loss", loss.item(), self.timestep_counters["train"])
            self.timestep_counters["train"] += 1
            pbar.set_description(f"train_loss: {loss.item():.3f}")

        loss_avg = loss_avg / num_samples
        acc = acc / num_samples
        pbar.close()

        return loss_avg, acc

    @torch.no_grad()
    def eval_(self):
        self.model.eval()

        if self.train_params["if_notebook"]:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(self.test_loader, total=len(self.test_loader), desc="eval", leave=False)

        loss_avg = 0
        acc = 0
        num_samples = 0
        for i, (X, y) in enumerate(pbar):
            # ###
            # # Debug only
            # if i > 1:
            #     break
            # ###
            X = X.float().to(ptu.device)
            y = y.long().to(ptu.device)  # (B,)
            y_pred = self.model(X)  # (B, num_classes)
            loss = self.loss(y_pred, y)

            # logging
            loss_avg += loss.item() * X.shape[0]
            y_pred_labels = torch.argmax(y_pred, dim=-1)
            acc += (y_pred_labels == y).sum()
            num_samples += X.shape[0]

            self.writer.add_scalar("eval_loss", loss.item(), self.timestep_counters["eval"])
            self.timestep_counters["eval"] += 1
            pbar.set_description(f"eval_loss: {loss.item():.3f}")

        loss_avg = loss_avg / num_samples
        acc = acc / num_samples
        pbar.close()

        return loss_avg, acc

    def train(self):
        if self.train_params["if_notebook"]:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.train_params["epochs"], desc="epoch")
        best_acc = -1

        for epoch in pbar:
            train_loss, train_acc = self.train_()
            eval_loss, eval_acc = self.eval_()

            # logging
            self.writer.add_scalar("train_acc", train_acc, self.timestep_counters["epoch"])
            self.writer.add_scalar("eval_acc", eval_acc, self.timestep_counters["epoch"])
            self.timestep_counters["epoch"] += 1
            pbar.set_description(f"loss: train: {train_loss:.3f}, eval: {eval_loss:.3f}, acc: train:{train_acc:.3f}, "
                                 f"eval: {eval_acc:.3f}")

            # save model
            if eval_acc > best_acc:
                best_acc = eval_acc
                model_filename = "cls.pt"
                model_savename = create_param_dir(self.log_params["param_save_dir"], model_filename)
                torch.save(self.model.state_dict(), model_savename)  # automatic rewriting

            # (optional) end-of-epoch visualization
            self.end_epoch_eval_(epoch)

    @torch.no_grad()
    def end_epoch_eval_(self, epoch):
        self.model.eval()
        index = np.random.randint(len(self.test_loader.dataset))
        img_test, label = self.test_loader.dataset[index]  # (C, H, W)
        y_pred = self.model(img_test.unsqueeze(0).to(ptu.device))  # (1, C, H, W) -> (1, num_classes)
        y_pred_label = torch.argmax(y_pred, dim=-1)  # (1,)
        fig, axis = plt.subplots()
        axis.imshow(img_test.permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axis.set_title(f"label: {CIFAR10_classes[label]}, pred: {CIFAR10_classes[y_pred_label[0]]}")
        if self.train_params["if_notebook"]:
            plt.show()
        plt.close()

        return fig


class ResNetTrainer(ClassifierTrainer):
    def __init__(self, model, train_loader, test_loader, train_params, log_params):
        super(ResNetTrainer, self).__init__(model, train_loader, test_loader, train_params, log_params)

    @torch.no_grad()
    def end_epoch_eval_(self, epoch):
        fig = super().end_epoch_eval_(epoch)

        return fig
