import easyfl
from server import RobustServer
from client import CustomizedClient

# Customized configuration.
config = {
    "attacker": {"byz_ratio": 0.2, "lie_z": 1.5},
    "data": {"dataset": "cifar10", "root": "./datasets", "split_type": "dir", "num_of_clients": 100},
    "server": {"rounds": 10, "clients_per_round": 10, "use_gas": True, "gas_p": 1000, "base_agg": "bulyan"},
    "client": {"local_epoch": 1},
    "model": "resnet18",
    "test_mode": "test_in_server",
    "gpu": 1,
}

easyfl.register_server(RobustServer)
easyfl.register_client(CustomizedClient)

# Initialize federated learning with default configurations.
easyfl.init(config)
# Execute federated learning training.
easyfl.run()