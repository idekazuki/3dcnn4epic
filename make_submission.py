import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import tensorboardX as tbx
import pickle
from functools import partial
from read_gulpio import EpicDataset_4_Vmodel
from train_model import train_fn
from validation_model import validation_fn
from spatial_transforms import CenterCrop, Compose, ComposeVideo, RandomCornerCrop, RandHorFlipVideo,ScaleVideo, Scale
from opts import parse_opts
from model import generate_model
from mean import get_mean
from collections import OrderedDict
import json

WEIGHTS_DIR = "./weights/"
verb_class_count, noun_class_count = 125, 352

if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

class CNN_3D(nn.Module):
    def __init__(self, opt):
        super(CNN_3D, self).__init__()

        #3d net
        self.net_3d = generate_model(opt)

        #FC layer
        #self.fc_final = nn.Linear(in_features=512, out_features=400, bias=True)

        #FC layer for verb
        self.fc_v = nn.Linear(in_features=32768, out_features=verb_class_count, bias=True)

        #FC layer for noun
        self.fc_n = nn.Linear(in_features=32768, out_features=noun_class_count, bias=True)

    def forward(self, x):
        """
        Args:
            x (torch.Size): torch.Size([batch_num, num_segments=16, 3, 224, 224]))

        """
        #bs, ns, c, h, w = x.shape
        #out = x.view(-1, c, h, w)
        #out = self.eco_2d(out)
        #out = out.view(-1, ns, 96, 28, 28)
        out = self.net_3d(x)
        #out = self.fc_final(out)
        out_v = self.fc_v(out)
        out_n = self.fc_n(out)
        return out_v, out_n

if __name__ == "__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400
    opt.epoch = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #net = generate_model(opt)
    net = CNN_3D(opt)
    print(net)
    print('loading model {}'.format(opt.model))

    #make model param key list
    para_name = []
    for i, name in enumerate(net.state_dict()):
        name_sep = name.split('.')
        if name_sep[-1] == 'num_batches_tracked':
            continue
        para_name.append(name)
        #print(i, name)

    for i, name in enumerate(para_name):
        print(i, name)

    model_data = torch.load(opt.model)
    model_data_dict = model_data['state_dict']
    #make pre train model key list
    pre_para_name = []
    for i, key in enumerate(model_data_dict.keys()):
        print(i, key)
        pre_para_name.append(key)


    #translate pretrain param to eco model
    for i, (name, val) in enumerate(model_data_dict.items()):
        #Since the structure is different after the last layer, the copying of weights is stopped.
        if name == 'module.fc.weight':
            print('stop load w')
            break
        print(name)
        net.state_dict()[para_name[i]].copy_(val)

    del model_data_dict
    print(net)

    if torch.cuda.device_count() > 1:
        print('use ', torch.cuda.device_count(), ' gpus')
        net = nn.DataParallel(net)

    net.to(device)
#    img_transpose = [Scale(256)]
#    temporal_transpose = [
#                        RandomCornerCrop([256, 224, 192, 168]),
#                        RandHorFlipVideo(),
#                        ScaleVideo(224)
#                        ]
#    transforms = ComposeVideo(img_transpose, temporal_transpose)
#    dataset = EpicDataset_4_Vmodel(transforms, class_type="verb+noun")
#    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
#    train_len = int(len(dataset) * 0.8)
#    val_len = int(len(dataset) - train_len)
#    torch.manual_seed(42)
#    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
#    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
#    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
#
#    #loss function, optimizer
#    criterion = torch.nn.CrossEntropyLoss()
#    optimizer = torch.optim.SGD(net.parameters(),
#                                0.001,
#                                momentum=0.9,
#                                weight_decay=5e-4)
#

## you have to define checkpoint data path
    save_mode = 'unseen'
    checkpoint_path = './logfile/resnext/save_20.pth'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['epoch_num'])
    del checkpoint

########################################################################
# data processing
    img_transpose = [Scale(256)]
    temporal_transpose = [
                         RandomCornerCrop([256]),
                         ScaleVideo(224)
                         ]
    transforms = ComposeVideo(img_transpose, temporal_transpose)
    datasets = EpicDataset_4_Vmodel(transforms, path='/home/yanai-lab/ide-k/ide-k/epic/data/processed/gulp/rgb_test_' + save_mode +'/', class_type='test')
    dataloader = torch.utils.data.DataLoader(datasets, batch_size = 16, shuffle=False)

    net.eval()

    #open save json file
    with open('./submit/' + save_mode + '.json') as f:
        df = json.load(f, object_pairs_hook=OrderedDict)
        
    for num, (data, label) in enumerate(dataloader):
        data = np.array(data) 
        #data = data.transpose(0, 2, 1, 3, 4)
        data = torch.as_tensor(data).float().to(device)
        print(data.size())
        out = net(data)          
        #x = torch.argmax(out[0], axis=1)
        #x = x.cpu().detach().numpy()
        #v_pre_list = np.append(v_pre_list, x)
        out_v = out[0].cpu().detach().numpy()
        out_n = out[1].cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        

        #y = torch.argmax(out[1], axis=1)
        #y = y.cpu().detach().numpy()
        #n_pre_list = np.append(n_pre_list, y)

        #v_ans_list = np.append(v_ans_list, label[0])
        #n_ans_list = np.append(n_ans_list, label[1])

#        print(v_ans_list.shape)
#        print(v_pre_list.shape)
#        print(n_ans_list.shape)
#        print(n_pre_list.shape)
#save as json file
        
        print(num)
        for i, (v, n) in enumerate(zip(out_v, out_n)):
            v_dict = {}
            n_dict = {}
            for v_num, v_val in enumerate(v):
                v_dict[str(v_num)] = float(v_val)

            for n_num, n_val in enumerate(n):
                n_dict[str(n_num)] = float(n_val)

            df['results'][str(label[i])] = {"verb":v_dict, "noun":n_dict}

        with open('./submit/' + save_mode + '.json', 'w') as f:
            json.dump(df, f, indent=2, ensure_ascii=False)

