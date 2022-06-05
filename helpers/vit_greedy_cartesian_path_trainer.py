import torch
import torch.nn as nn
import SemesterProject2.helpers.pytorch_utils as ptu

from SemesterProject2.helpers.modules.vit_agent_modules import EncoderGreedy, MLPHead
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from SemesterProject2.helpers.deterministic_path_sampler import DeterministicPathSampler
from SemesterProject2.helpers.data_processing import VolumetricDataset
from SemesterProject2.helpers.utils import create_param_dir


class CartesianTrainer(object):
    def __init__(self, params):
        """
        params:
        configs_ac.volumetric_env_params
        As keys: configs_networks:
            encoder_params, *_head_params,
            encoder_opt_args, *_head_opt_args (add clf_head_params & clf_head_opt_args)
        bash:
            num_updates, hdf5_filename, train_test_filename, grid_size, if_clip_grad, log_dir, params_save_dir, if_notebook
        """
        self.params = params
        self.encoder = EncoderGreedy(self.params["encoder_params"]).to(ptu.device)
        self.patch_pred_head = MLPHead(self.params["patch_pred_head_params"]).to(ptu.device)
        self.clf_head = MLPHead(self.params["clf_head_params"]).to(ptu.device)
        self.encoder_opt = self.params["encoder_opt_args"]["class"](self.encoder.parameters(),
                                                                    **self.params["encoder_opt_args"]["args"])
        self.patch_pred_head_opt = self.params["patch_pred_head_opt_args"]["class"](self.patch_pred_head.parameters(),
                                                                                    **self.params[
                                                                                        "patch_pred_head_opt_args"][
                                                                                        "args"])
        self.clf_head_opt = self.params["clf_head_opt_args"]["class"](self.clf_head.parameters(),
                                                                      **self.params["clf_head_opt_args"]["args"])
        train_ds = VolumetricDataset(self.params["hdf5_filename"], self.params["train_test_filename"], "train")
        test_ds = VolumetricDataset(self.params["hdf5_filename"], self.params["train_test_filename"], "test")
        self.sampler_train = DeterministicPathSampler(train_ds, self.params)
        self.sampler_test = DeterministicPathSampler(test_ds, self.params)
        self.ce_loss = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(self.params["log_dir"])
        self.global_steps = {"train": 0, "eval": 0, "epoch": 0}

    def train_(self):
        self.encoder.train()
        self.patch_pred_head.train()
        self.clf_head.train()

        log_dicts = []
        samples = self.sampler_train.sample()
        for sample in samples:
            log_dict = self.compute_loss_and_update_(sample, if_train=True)
            log_dicts.append(log_dict)
            for key, val in log_dict.items():
                self.writer.add_scalar(key, val, self.global_steps["train"])
                ### TODO: (Opt) comment out ###
                print(f"{key}: {val: .3f}")
                ### end of TODO block ###
            ### TODO: (Opt) comment out ###
            print("-" * 100)
            ### end of TODO block ###
            self.global_steps["train"] += 1

        return log_dicts

    @torch.no_grad()
    def eval_(self):
        self.encoder.eval()
        self.patch_pred_head.eval()
        self.clf_head.eval()

        log_dicts = []
        samples = self.sampler_train.sample()
        for sample in samples:
            log_dict = self.compute_loss_and_update_(sample, if_train=False)
            log_dicts.append(log_dict)
            for key, val in log_dict.items():
                self.writer.add_scalar(key, val, self.global_steps["eval"])
                ### TODO: (Opt) comment out ###
                print(f"{key}: {val: .3f}")
                ### end of TODO block ###
            ### TODO: (Opt) comment out ###
            print("-" * 100)
            ### end of TODO block ###
            self.global_steps["eval"] += 1

        return log_dicts

    def train(self):
        if self.params["if_notebook"]:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.params["num_updates"], desc="epochs")

        best_metric = 0
        for _ in pbar:
            train_log_dicts = self.train_()
            eval_log_dicts = self.eval_()

            try:
                eval_last_log_dict = eval_log_dicts[-1]
                eval_key = "eval_acc"
                if best_metric <= eval_last_log_dict[eval_key]:
                    best_metric = eval_last_log_dict[eval_key]
                    self.save_models_()
            except Exception as e:
                print(e)
                self.save_models_()

    def compute_loss_and_update_(self, sample, if_train=True):
        """
        sample: X_small, X_large, X_pos, X_next_small, X_next_large, X_next_pos, has_lesion
        (T, N, 1, P, P, P), (T, N, 1, 2P, 2P, 2P), (T, N, 3),
        (1, N, 1, P, P, P), (1, N, 1, 2P, 2P, 2P), (1, N, 3), (N,)
        """
        X_small, X_large, X_pos, X_next_small, X_next_large, X_next_pos, has_lesion = sample
        X_small = ptu.from_numpy((2 * X_small - 1))
        X_large = ptu.from_numpy((2 * X_large - 1))
        X_pos = ptu.from_numpy(X_pos).to(ptu.device)
        X_next_small = ptu.from_numpy((2 * X_next_small - 1)).to(ptu.device)
        X_next_large = ptu.from_numpy((2 * X_next_large - 1)).to(ptu.device)
        X_next_pos = ptu.from_numpy(X_next_pos).to(ptu.device)
        has_lesion = ptu.from_numpy(has_lesion).to(ptu.device).long()

        X_small_all = torch.cat([X_small, X_next_small], dim=0)  # (T + 1, N, 1, P, P, P)
        X_large_all = torch.cat([X_large, X_next_large], dim=0)  # (T + 1, N, 1, 2P, 2P, 2P)
        X_pos_all = torch.cat([X_pos, X_next_pos], dim=0)  # (T + 1, N, 3)

        # TODO: consider patch prediction loss
        # clf loss
        if not if_train:
            log_dict = {}
            X_emb_enc = self.encoder(X_small_all, X_large_all, X_pos_all, X_next_pos)  # (N, N_emb)
            X_clf_pred = self.clf_head(X_emb_enc)  # (N, 2)
            loss = self.ce_loss(X_clf_pred, has_lesion.long())

            with torch.no_grad():
                print(f"X_pred: {ptu.to_numpy(torch.softmax(X_clf_pred, dim=-1))}")
                print(f"X_gt: {ptu.to_numpy(has_lesion)}")
                print("-" * 100)

            acc, precision, recall, f1 = self.compute_statistics_(X_clf_pred, has_lesion)
            log_dict.update({
                "eval_loss": loss.item(),
                "eval_acc": acc,
                "eval_precision": precision,
                "eval_recall": recall,
                "eval_f1": f1
            })

            return log_dict

        else:
            log_dict = {}
            print("training...")
            for i in range(self.params["num_updates_clf"]):
                X_emb_enc = self.encoder(X_small_all, X_large_all, X_pos_all, X_pos)  # (N, N_emb)
                X_clf_pred = self.clf_head(X_emb_enc)  # (N, 2)
                loss = self.ce_loss(X_clf_pred, has_lesion.long())
                if i == 0:
                    acc, precision, recall, f1 = self.compute_statistics_(X_clf_pred, has_lesion)
                    log_dict.update({
                        "train_loss": loss.item(),
                        "train_acc": acc,
                        "train_precision": precision,
                        "train_recall": recall,
                        "train_f1": f1
                    })
                self.encoder_opt.zero_grad()
                self.clf_head_opt.zero_grad()
                self.patch_pred_head_opt.zero_grad()
                loss.backward()
                clip_val = self.params.get("if_clip_grad", None)
                if clip_val is not None:
                    nn.utils.clip_grad_value_(self.encoder.parameters(), clip_val)
                    nn.utils.clip_grad_value_(self.clf_head.parameters(), clip_val)
                    nn.utils.clip_grad_value_(self.patch_pred_head.parameters(), clip_val)
                self.encoder_opt.step()
                self.clf_head_opt.step()
                self.patch_pred_head_opt.step()

                with torch.no_grad():
                    print(f"X_pred: {ptu.to_numpy(torch.softmax(X_clf_pred, dim=-1))}")
                    print(f"X_gt: {ptu.to_numpy(has_lesion)}")
                    print("-" * 100)

            # acc, precision, recall, f1 = self.compute_statistics_(X_clf_pred, has_lesion)
            # log_dict.update({
            #     "train_loss": loss.item(),
            #     "train_acc": acc,
            #     "train_precision": precision,
            #     "train_recall": recall,
            #     "train_f1": f1
            # })

            return log_dict

    @torch.no_grad()
    def compute_statistics_(self, X_pred, X_gt):
        # X_pred: (N, 2), X_gt: (N,)
        X_pred = torch.softmax(X_pred, dim=-1)  # (N, 2)
        X_pred = ptu.to_numpy(X_pred)
        X_gt = ptu.to_numpy(X_gt).astype(int)
        X_pred_label = (X_pred[:, 1] > self.params["conf_score_threshold"]).astype(int)
        # print(f"from .compute_statistics(.): X_pred: {X_pred}")
        acc = accuracy_score(X_gt, X_pred_label)
        precision = precision_score(X_gt, X_pred_label)
        recall = recall_score(X_gt, X_pred_label)
        f1 = f1_score(X_gt, X_pred_label)

        return acc, precision, recall, f1

    def save_models_(self):
        torch.save(self.encoder.state_dict(), create_param_dir(self.params["params_save_dir"], "encoder.pt"))
        torch.save(self.clf_head.state_dict(), create_param_dir(self.params["params_save_dir"], "clf_head.pt"))
        torch.save(self.patch_pred_head.state_dict(), create_param_dir(self.params["params_save_dir"], "patch_pred_head.pt"))
