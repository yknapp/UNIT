"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
from trainer import UNIT_Trainer
from torch.autograd import Variable
import torch
from torchvision import transforms


class Lyft2KittiConverter:
    def __init__(self, config, checkpoint, a2b=1, seed=10):
        self.a2b = a2b
        self.encode = None
        self.decode = None

        # set seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # load config
        config = get_config(config)  # Load experiment setting

        # setup data loader
        self.setup_data_loader(config, checkpoint)

    def setup_data_loader(self, config, checkpoint):
        trainer = UNIT_Trainer(config)
        try:
            state_dict = torch.load(checkpoint)
            trainer.gen_a.load_state_dict(state_dict['a'])
            trainer.gen_b.load_state_dict(state_dict['b'])
        except:
            state_dict = pytorch03_to_pytorch04(torch.load(checkpoint))
            trainer.gen_a.load_state_dict(state_dict['a'])
            trainer.gen_b.load_state_dict(state_dict['b'])

        trainer.cuda()
        trainer.eval()
        self.encode = trainer.gen_a.encode if self.a2b else trainer.gen_b.encode  # encode function
        self.decode = trainer.gen_b.decode if self.a2b else trainer.gen_a.decode  # decode function

    def transform(self, lidar_bev_input):
        with torch.no_grad():
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            pointcloud = transform(lidar_bev_input).unsqueeze(0).cuda()
            pointcloud = pointcloud.to(dtype=torch.float)
            pointcloud = Variable(pointcloud)

        # Start testing
        content, _ = self.encode(pointcloud)

        outputs = self.decode(content)
        outputs = (outputs + 1) / 2.

        # convert to numpy array
        lidar_bev_output = outputs.data.detach().cpu().numpy()  # to numpy array

        return lidar_bev_output[0, :, :, :]
