import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb

from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from models.basic_model import AVClassifier
from utils.utils import setup_seed, weight_init


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE'])

    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film', 'metamodal'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--audio_path', default='/home/hexiang/data/datasets/CREMA-D/AudioWAV/', type=str)
    parser.add_argument('--visual_path', default='/home/hexiang/data/datasets/CREMA-D/', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', required=True, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')

    parser.add_argument('--inverse', action='store_true', help='inverse effectiveness')

    parser.add_argument('--inverseGE', action='store_true', help='inverse effectiveness GE')
    return parser.parse_args()

args = get_arguments()

import logging

# 创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建一个handler，用于写入日志文件
file_handler = logging.FileHandler('ckpt/log_{}_inverse_{}_GE_{}_bs_{}.txt'.format(args.modulation, args.inverse, args.inverseGE, args.batch_size))
file_handler.setLevel(logging.INFO)

# 创建一个handler，用于将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 给logger添加handler
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(args)

writer_path = os.path.join(args.tensorboard_path, args.dataset)
if not os.path.exists(writer_path):
    os.mkdir(writer_path)
log_name = '{}_{}_inverse_{}_GE_{}_bs_{}'.format(args.fusion_method, args.modulation, args.inverse, args.inverseGE, args.batch_size)
writer = SummaryWriter(os.path.join(writer_path, log_name))


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    logger.info("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    coeff_av_max = -1
    for step, (spec, image, label) in enumerate(dataloader):

        #pdb.set_trace()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        a, v, out = model(spec.unsqueeze(1).float(), image.float())

        if args.fusion_method == 'sum':
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                     model.module.fusion_module.fc_y.bias)
            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                     model.module.fusion_module.fc_x.bias)
        else:
            weight_size = model.module.fusion_module.fc_out.weight.size(1)
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                     + model.module.fusion_module.fc_out.bias / 2)

            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                     + model.module.fusion_module.fc_out.bias / 2)

        loss = criterion(out, label)
        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)
        loss.backward()

        # Modulation starts here !
        score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
        score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
        score_av = sum([softmax(out)[i][label[i]] for i in range(out.size(0))])

        ratio_v = score_v / score_a
        ratio_a = 1 / ratio_v
        ratio_av = (score_a + score_v) / score_av

        """
        Below is the Eq.(10) in our CVPR paper:
                1 - tanh(alpha * rho_t_u), if rho_t_u > 1
        k_t_u =
                1,                         else
        coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
        """

        if ratio_v > 1:
            coeff_v = 1 - tanh(args.alpha * relu(ratio_v))
            coeff_a = 1
        else:
            coeff_a = 1 - tanh(args.alpha * relu(ratio_a))
            coeff_v = 1
        coeff_av = 1 + tanh(torch.tensor(1.0)) - tanh(relu(ratio_av))  # a 和 v 越弱, av出来越强

        if args.use_tensorboard:
            iteration = epoch * len(dataloader) + step

            writer.add_scalars('score', {'a': score_a,
                                         'v': score_v,
                                         'av': score_av}, iteration)

            writer.add_scalars('Coefficient', {'a': coeff_a,
                                               'v': coeff_v,
                                               'av': coeff_av}, iteration)

        if args.modulation_starts <= epoch <= args.modulation_ends: # bug fixed
            for name, parms in model.named_parameters():
                layer = str(name).split('.')[1]

                if 'audio' in layer and len(parms.grad.size()) == 4:
                    if args.modulation == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeff_a + \
                                     torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif args.modulation == 'OGM':
                        parms.grad *= coeff_a

                if 'visual' in layer and len(parms.grad.size()) == 4:
                    if args.modulation == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeff_v + \
                                     torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif args.modulation == 'OGM':
                        parms.grad *= coeff_v

                if 'fusion' in layer:
                    if args.inverse is True:
                        if args.inverseGE is True:
                            parms.grad *= coeff_av + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        else:
                            parms.grad *= coeff_av
        else:
            pass


        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()

    scheduler.step()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            a, v, out = model(spec.unsqueeze(1).float(), image.float())

            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                         model.module.fusion_module.fc_y.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                         model.module.fusion_module.fc_x.bias / 2)
            else:
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                #pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=8, pin_memory=True)

    if args.train:

        best_acc = 0.0

        best_models = []

        for epoch in range(args.epochs):

            logger.info('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:

                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

                # writer.add_scalars('Loss', {'Total Loss': batch_loss,
                #                             'Audio Loss': batch_loss_a,
                #                             'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

            else:
                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = '{}_inverse_{}_GE_{}_alpha_{}_' \
                             'bs_{}_modulate_starts_{}_ends_{}_' \
                             'epoch_{}_acc_{}.pth'.format(args.modulation,
                                                          args.inverse,
                                                          args.inverseGE,
                                                          args.alpha,
                                                          args.batch_size,
                                                          args.modulation_starts,
                                                          args.modulation_ends,
                                                          epoch, acc)

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'alpha': args.alpha,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                logger.info('The best model has been saved at {}.'.format(save_dir))
                logger.info("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                logger.info("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))

                # 更新已保存的最佳模型列表
                best_models.append((acc, save_dir))
                best_models.sort(key=lambda x: x[0], reverse=True)  # 按准确率降序排序

                # 如果保存的模型超过1个，则删除准确率最低的模型
                while len(best_models) > 1:
                    _, oldest_model_path = best_models.pop()  # 获取准确率最低的模型
                    os.remove(oldest_model_path)  # 删除该模型文件

            else:
                logger.info("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                logger.info("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model.load_state_dict(state_dict)
        logger.info('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        logger.info('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
