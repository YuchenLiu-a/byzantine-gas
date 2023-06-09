from __future__ import annotations

from client import CustomizedClient
from easyfl.server.base import MODEL
from typing import (
    Tuple,
)

class Attacker:
    def __init__(self, conf, byz_clients: list[CustomizedClient]) -> None:
        self.conf = conf
        self.byz_clients = byz_clients
    
    def attack(self, sampled_clients: list, server):
        raise Exception('instantiate attack')

    def get_ref_models(self, sampled_clients: list[CustomizedClient]):
        ref_models = [sampled_client.model for sampled_client in sampled_clients]
        return ref_models
    
    def set_byz_uploaded_content(self, sampled_clients, byz_state_dict, server):
        sampled_byz_clients = self.get_sampled_byz_clients(sampled_clients)
        for sampled_byz_client in sampled_byz_clients:
            server._client_uploads[MODEL][sampled_byz_client.cid].load_state_dict(byz_state_dict)

    def get_sampled_byz_clients(self, sampled_clients):
        sampled_byz_clients = list(set(sampled_clients).intersection(self.byz_clients))
        return sampled_byz_clients