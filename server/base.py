from __future__ import annotations

import copy
import time
import torch

from argparse import Namespace
from easyfl.server import BaseServer
from easyfl.tracking import metric
from easyfl.utils.float import rounding
from math import ceil
from omegaconf import OmegaConf

from attacker import Lie
from client import CustomizedClient
from utils import flatten_models, unflatten_tensor
from .agg_funs import agg_funs

class RobustServer(BaseServer):
    def start(self, model, clients):
        """Start federated learning process, including training and testing.

        Args:
            model (nn.Module): The model to train.
            clients (list[:obj:`BaseClient`]|list[str]): Available clients.
                Clients are actually client grpc addresses when in remote training.
        """
        # Setup
        self._start_time = time.time()
        self._reset()
        self.set_model(model)
        self.set_clients(clients)
        # customize
        byz_clients = self.get_byz_clients(clients)
        self.attacker = Lie(self.conf, byz_clients)

        if self._should_track():
            self._tracker.create_task(self.conf.task_id, OmegaConf.to_container(self.conf))

        # Get initial testing accuracies
        if self.conf.server.test_all:
            if self._should_track():
                self._tracker.set_round(self._current_round)
            self.test()
            self.save_tracker()

        while not self.should_stop():
            self._round_time = time.time()

            self._current_round += 1
            self.print_("\n-------- round {} --------".format(self._current_round))

            # Train
            self.pre_train()
            self.train()
            self.post_train()

            # Test
            if self._do_every(self.conf.server.test_every, self._current_round, self.conf.server.rounds):
                self.pre_test()
                self.test()
                self.post_test()

            # Save Model
            self.save_model()

            self.track(metric.ROUND_TIME, time.time() - self._round_time)
            self.save_tracker()

        self.print_("Accuracies: {}".format(rounding(self._accuracies, 4)))
        self.print_("Cumulative training time: {}".format(rounding(self._cumulative_times, 2)))

    def train(self):
        """Training process of federated learning."""
        self.print_("--- start training ---")

        self.selection(self._clients, self.conf.server.clients_per_round)
        self.grouping_for_distributed()
        self.compression()

        begin_train_time = time.time()
        self.distribution_to_train()
        end_train_time = time.time()
        train_time = end_train_time - begin_train_time
        self.print_("Honest client train time: {}".format(train_time))

        start_attack_time = time.time()
        self.attacker.attack(self.selected_clients, self)
        end_attack_time = time.time()
        attack_time = end_attack_time - start_attack_time
        self.print_("Byzantine client attack time: {}".format(attack_time))

        start_aggretate_time = time.time()
        self.aggregation()
        end_aggretate_time = time.time()
        aggregate_time = end_aggretate_time - start_aggretate_time
        self.print_("Aggregate time: {}".format(aggregate_time))

        self.track(metric.TRAIN_TIME, train_time)
        ### track other times

    def get_num_byz_clients(self):
        num_clients = self.conf.data.num_of_clients
        byz_ratio: float = self.conf.attacker.byz_ratio
        num_byz = ceil(num_clients * byz_ratio)
        assert 0 <= num_byz <= num_clients, f"invalid byz_ratio {byz_ratio}"
        return num_byz

    def get_byz_clients(self, clients: list[CustomizedClient]):
        num_byz = self.get_num_byz_clients()
        
        byz_clients = clients[:num_byz]
        for client in byz_clients:
            client.set_byz()
        
        return byz_clients
        
    def set_attacker(self, attacker):
        self.attacker = attacker

    def aggregate(self, models, weights):
        if self.conf.server.use_gas:
            agg_model = self.gas_aggregate(models, weights)
        else:
            flat_models, struct = flatten_models(models)

            base_agg = agg_funs[self.conf.server.base_agg]   
            n_sel_byz = sum((1 if selected_client.is_byz else 0) for selected_client in self.selected_clients)
            knowledge = Namespace(n_byz=n_sel_byz)
            flat_agg_model = base_agg(flat_models, knowledge, self.conf)

            agg_state_dict = unflatten_tensor(flat_agg_model, struct)
            agg_model = copy.deepcopy(models[0])
            agg_model.load_state_dict(agg_state_dict)
        return agg_model
    
    @torch.no_grad()
    def gas_aggregate(self, models, weights):
        # flatten
        flat_models, struct = flatten_models(models)
        # splitting
        groups = self.split(flat_models)
        # identification
        base_agg = agg_funs[self.conf.server.base_agg]
        n_cl = len(flat_models)
        n_sel_byz = sum((1 if selected_client.is_byz else 0) for selected_client in self.selected_clients)
        knowledge = Namespace(n_byz=n_sel_byz)
        identification_scores = torch.zeros(n_cl)
        for group in groups:
            group_agg = base_agg(group, knowledge, self.conf)
            group_scores = (group - group_agg).square().sum(dim=-1).sqrt().cpu()
            identification_scores += group_scores
        _, cand_idxs = identification_scores.topk(k=n_cl - n_sel_byz, largest=False)
        n_agg_byz = sum([self.selected_clients[i].is_byz for i in cand_idxs.tolist()])
        self.print_(f"Aggregated byzantine / selected byzantine: {n_agg_byz} / {n_sel_byz}")
        # aggregation
        flat_agg_model = flat_models[cand_idxs].mean(dim=0)
        # unflatten
        agg_state_dict = unflatten_tensor(flat_agg_model, struct)
        agg_model = copy.deepcopy(models[0])
        agg_model.load_state_dict(agg_state_dict)
        
        return agg_model

    
    @torch.no_grad()
    def split(self, flat_models):
        d = flat_models.shape[1]
        shuffled_dims = torch.randperm(d).to(flat_models.device)
        p = self.conf.server.gas_p
        partition = torch.chunk(shuffled_dims, chunks=p)
        groups = [flat_models[:, partition_i] for partition_i in partition]
        return groups