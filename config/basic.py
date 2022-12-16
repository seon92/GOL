import torchvision.transforms as transforms
from PIL import Image

class ConfigBasic:
    def __init__(self,):
        self.dataset = None
        self.setting = None
        self.logscale = False
        self.set_optimizer_parameters()
        self.set_training_opts()
        self.set_network()

    def set_dataset(self):
        if self.dataset == 'morph':
            if self.logscale:
                self.tau = 0.1
            else:
                self.tau = 2

            self.img_root = '/hdd/2020/Research/datasets/Agedataset/img/morph'
            if self.setting == 'A':
                self.is_filelist = True
                self.train_file = f'/hdd/2020/Research/datasets/Agedataset/morph_fold/Setting_A/setting_A_train_fold{self.fold}.txt'
                self.test_file = f'/hdd/2020/Research/datasets/Agedataset/morph_fold/Setting_A/setting_A_test_fold{self.fold}.txt'
                self.delimeter = ","
                self.img_idx = 4
                self.lb_idx = 3

            elif self.setting == 'B':
                self.delimeter = " "
                self.img_idx = 3
                self.lb_idx = 2
                if self.fold == 1:
                    self.is_filelist = True
                    self.train_file = f'/hdd/2020/Research/datasets/Agedataset/morph_fold/Setting_B/Setting_B_S1_train.txt'
                    self.test_file = f'/hdd/2020/Research/datasets/Agedataset/morph_fold/Setting_B/Setting_B_S2+S3_test.txt'
                else:
                    self.is_filelist = True
                    self.train_file = f'/hdd/2020/Research/datasets/Agedataset/morph_fold/Setting_B/Setting_B_S2_train.txt'
                    self.test_file = f'/hdd/2020/Research/datasets/Agedataset/morph_fold/Setting_B/Setting_B_S1+S3_test.txt'

            elif self.setting == 'C':
                self.delimeter = " "
                self.img_idx = 0
                self.lb_idx = 2
                self.is_filelist = True
                self.train_file = f'/hdd/2020/Research/datasets/Agedataset/morph_fold/Setting_C/setting_C_train_fold{self.fold}.txt'
                self.test_file = f'/hdd/2020/Research/datasets/Agedataset/morph_fold/Setting_C/setting_C_test_fold{self.fold}.txt'

            elif self.setting == 'D':
                self.delimeter = " "
                self.img_idx = 0
                self.lb_idx = 2
                self.is_filelist = True
                self.train_file = f'/hdd/2020/Research/datasets/Agedataset/morph_fold/Setting_D/setting_D_train_fold{self.fold}.txt'
                self.test_file = f'/hdd/2020/Research/datasets/Agedataset/morph_fold/Setting_D/setting_D_test_fold{self.fold}.txt'
            else:
                raise ValueError(f'setting {self.setting} is out of range.')

        elif self.dataset == 'adience':
            self.is_filelist = False
            self.train_file = f'/hdd/2021/Research/99_dataset/Adience/adience_F{self.fold}_train_algn_[0_7].pickle'
            self.test_file = f'/hdd/2021/Research/99_dataset/Adience/adience_F{self.fold}_test_algn_[0_7].pickle'
            self.tau = 1

        elif self.dataset =='clap':
            self.delimeter = " "
            self.img_idx = 0
            self.lb_idx = 1
            self.is_filelist = True
            self.img_root = '/hdd/2020/Research/datasets/Agedataset/img/CLAP/2015'
            if self.fold == 'eval_on_test':
                self.train_file = '/hdd/2020/Research/datasets/Agedataset/clap_split/CLAP_trainval.txt'
                self.test_file = '/hdd/2020/Research/datasets/Agedataset/clap_split/CLAP_test.txt'
            elif self.fold == 'eval_on_val':
                self.train_file = '/hdd/2020/Research/datasets/Agedataset/clap_split/CLAP_train.txt'
                self.test_file = '/hdd/2020/Research/datasets/Agedataset/clap_split/CLAP_val.txt'
            else:
                raise ValueError(f'check fold: it should be [eval_on_test] or [eval_on_val], but {self.fold} is given.')
        else:
            raise ValueError(f'{self.dataset} is out of range!')

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.transform_tr = transforms.Compose([
                                                lambda x: Image.fromarray(x),
                                                transforms.RandomCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                self.normalize
                                                ])

        self.transform_te = transforms.Compose([
                                                lambda x: Image.fromarray(x),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                self.normalize
                                                ])

    def set_optimizer_parameters(self):
        # *** Optimizer
        self.adam = True
        self.learning_rate = 0.0001
        self.lr_decay_epochs = [30, 50, 100]
        self.lr_decay_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0005

        # *** Scheduler
        self.scheduler = 'cosine'

    def set_network(self):
        self.model = 'T_v0'
        self.backbone = 'vgg16bn'
        self.ckpt = None

    def set_training_opts(self):
        # *** Print Option
        self.val_freq = 3
        self.print_freq = 50

        # *** Training
        self.batch_size = 16
        self.num_workers = 1
        self.epochs = 100

        # *** Save option
        self.save_freq = 100
        self.wandb = False

    def set_test_opts(self):
        self.ckpt = None
