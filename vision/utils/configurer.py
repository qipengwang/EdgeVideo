from .logger import Logger
import os



class Configurer:
    def __init__(self, config_file: str = 'default.cfg'):
        self.config_file = config_file
        self.logger = Logger.get_logger()
        self.init_default_cfg()
        self.read_config()
        self.check()
        self.log_config()
    
    def init_default_cfg(self):
        self.intattr = ['t_max', 'batch_size', 'num_epochs', 'num_workers', 'validation_epochs', 'debug_steps']
        self.floatattr = ['gamma', 'mb2_width_mult', 'lr', 'learning_rate', 'momentum', 'weight_decay', 'base_net_lr', 'extra_layers_lr']
        self.boolattr = ['balance_data', 'freeze_base_net', 'freeze_net', 'ssd320', 'use_cuda']
        self.dataset_type = 'city_scapes'  # voc
        self.datasets = ['data', ]  # []
        self.validation_dataset = 'data'
        self.balance_data = False
        self.city = 'berlin'
        self.net = 'mb3-large-ssd-lite'
        self.freeze_base_net = False
        self.freeze_net = False
        self.mb2_width_mult = 1.0
        self.ssd320 = False
        self.lr = self.learning_rate = 1e-3
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.gamma = 0.1
        self.base_net_lr = None
        self.extra_layers_lr = None
        self.base_net = None
        self.pretrained_ssd = None
        self.resume = None
        self.scheduler = 'multi-step'
        self.milestones = '80,100'
        self.t_max = 120
        self.batch_size = 16
        self.num_epochs = 200
        self.num_workers = 4
        self.validation_epochs = 5
        self.debug_steps = 100
        self.use_cuda = True
        self.checkpoint_folder = 'models/'
    
    def check(self):
        assert self.net in ['mb1-ssd', 'mb1-ssd-lite', 'mb2-ssd-lite', 'mb3-large-ssd-lite', 'mb3-small-ssd-lite', 'vgg16-ssd']
        assert self.scheduler in  ['multi-step', 'cosine']
    
    def read_config(self):
        if not os.path.exists(self.config_file): return
        with open(self.config_file) as f:
            for line in f:
                arg, val = line.strip().split()
                if arg in self.intattr:
                    self.__setattr__(arg, int(val))
                elif arg in self.floatattr:
                    self.__setattr__(arg, float(val))
                    if arg in ['lr', 'learning_rate']:
                        self.__setattr__('lr', float(val))
                        self.__setattr__('learning_rate', float(val))
                elif arg in self.boolattr:
                    self.__setattr__(arg, bool(val))
                elif arg == 'datasets':
                    if val not in self.datasets:
                        self.datasets.append(val)
                else:
                    self.__setattr__(arg, val)
                    
    def log_config(self):
        configs = vars(self)
        self.logger.info('================= Config =================')
        for k, v in configs.items():
            self.logger.info(f'\t{k} = {type(v)}:{v}')
        self.logger.info('================= ====== =================')

