from easyfl.client import BaseClient

class CustomizedClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(CustomizedClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        self.is_byz = False
    
    def set_byz(self, is_byz: bool=True):
        self.is_byz = is_byz
        pass
    
    def post_train(self):
        self.model.cpu()