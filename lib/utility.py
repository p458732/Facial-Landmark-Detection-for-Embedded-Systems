import json
import os.path as osp
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from timm.scheduler.cosine_lr import CosineLRScheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

# private package
from conf import *
from lib.backbone.swin_v2 import swin_v2
from lib.backbone.efficientFormerv2 import efficientformerv2_l
from lib.backbone.mobilevit_v2 import mobile_vit_v2
from lib.dataset import AlignmentDataset
from lib.backbone import StackedHGNetV1, efficientformerv2_s0
from lib.loss import *
from lib.metric import NME, FR_AUC, MSE
from lib.utils import convert_secs2time
from lib.utils import AverageMeter


def get_config(args):
    config = None
    config_name = args.config_name
    if config_name == "alignment":
        config = Alignment(args)
    else:
        assert NotImplementedError

    return config


def get_dataset(config, tsv_file, image_dir, loader_type, is_train):
    dataset = None
    if loader_type == "alignment":
        dataset = AlignmentDataset(
            tsv_file,
            image_dir,
            transforms.Compose([transforms.ToTensor()]),
            config.width,
            config.height,
            config.channels,
            config.means,
            config.scale,
            config.classes_num,
            config.crop_op,
            config.aug_prob,
            config.edge_info,
            config.flip_mapping,
            is_train,
            encoder_type=config.encoder_type
        )
    else:
        assert False
    return dataset


def get_dataloader(config, data_type, world_rank=0, world_size=1):
    loader = None
    if data_type == "train":
        dataset = get_dataset(
            config,
            config.train_tsv_file,
            config.train_pic_dir,
            config.loader_type,
            is_train=True)
        if world_size > 1:
            sampler = DistributedSampler(
                dataset, rank=world_rank, num_replicas=world_size, shuffle=True)
            loader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size // world_size,
                                num_workers=config.train_num_workers, pin_memory=True, drop_last=True)
        else:
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                num_workers=config.train_num_workers)
    elif data_type == "val":
        dataset = get_dataset(
            config,
            config.val_tsv_file,
            config.val_pic_dir,
            config.loader_type,
            is_train=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.val_batch_size,
                            num_workers=config.val_num_workers)
    elif data_type == "test":
        dataset = get_dataset(
            config,
            config.test_tsv_file,
            config.test_pic_dir,
            config.loader_type,
            is_train=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.test_batch_size,
                            num_workers=config.test_num_workers)
    else:
        assert False
    return loader


def get_optimizer(config, net):
    params = net.parameters()
    my_list = ['out_pointmaps.conv.weight', 'out_pointmaps.conv.bias', 'out_edgemaps.conv.weight', 'out_edgemaps.conv.bias', 'out_heatmaps.conv.weight', 'out_heatmaps.conv.bias']
    params = []
    base_params = []
    for name, param in net.named_parameters():
        if name in my_list:
            params.append(param)
        else:
            base_params.append(param)
    optimizer = None
    if config.optimizer == "sgd":
        optimizer = optim.SGD(
            params,
            lr=config.learn_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov)
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            params,
            lr=config.learn_rate)
    elif config.optimizer == "adamW":
        optimizer = optim.AdamW([
            {'params': base_params}], lr=config.learn_rate, weight_decay=config.weight_decay)
        # optimizer = optim.AdamW([
        #     {'params': base_params}, 
        #     {'params': params, 'lr': 1e-3, 'weight_decay': 2e-5}], lr=config.learn_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            params,
            lr=config.learn_rate,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.epsilon,
            weight_decay=config.weight_decay
        )
    else:
        assert False
    return optimizer


def get_scheduler(config, optimizer):
    if config.scheduler == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=config.milestones, gamma=config.gamma)
    elif config.scheduler == "Cosine":
        
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                        T_max = 500)
    else:
        assert False
    return scheduler

def get_student_net(config):
    net = None
    if config.student_net == "stackedHGnet_v1":
        net = StackedHGNetV1(config=config,
                             classes_num=config.classes_num,
                             edge_info=config.edge_info,
                             nstack=config.nstack,
                             add_coord=config.add_coord,
                             decoder_type=config.decoder_type)
        #stat(net, (3, 256, 256))
    elif config.student_net == "efficientFormer_v2_l":
        net = efficientformerv2_l(pretrained=True, edge_info=config.edge_info,)
    elif config.student_net == "efficientFormer_v2_s0":
        net = efficientformerv2_s0(pretrained=True, edge_info=config.edge_info,)
    elif config.student_net == "mobile_vit_v2":
        net = mobile_vit_v2()
    elif config.student_net == "swin_v2":
        net = swin_v2()
    else:
        assert False
    return net

def get_teacher_net(config):
    if config.teacher_net == "stackedHGnet_v1":
        net = StackedHGNetV1(config=config,
                             classes_num=config.classes_num,
                             edge_info=config.edge_info,
                             nstack=config.nstack,
                             add_coord=config.add_coord,
                             decoder_type=config.decoder_type)
        #stat(net, (3, 256, 256))
    elif config.teacher_net == "efficientFormer_v2_l":
        net = efficientformerv2_l(pretrained=True, edge_info=config.edge_info,)
    elif config.teacher_net == "efficientFormer_v2_s0":
        net = efficientformerv2_s0(pretrained=True, edge_info=config.edge_info,)
    elif config.teacher_net == "mobile_vit_v2":
        net = mobile_vit_v2()
    elif config.teacher_net == "swin_v2":
        net = swin_v2()
    else:
        assert False
    return net
def get_net(config, teacher = False):
    if teacher:
        net = StackedHGNetV1(config=config,
                             classes_num=config.classes_num,
                             edge_info=config.edge_info,
                             nstack=config.nstack,
                             add_coord=config.add_coord,
                             decoder_type=config.decoder_type)
        return  net
    net = None
    if config.net == "stackedHGnet_v1":
        net = StackedHGNetV1(config=config,
                             classes_num=config.classes_num,
                             edge_info=config.edge_info,
                             nstack=config.nstack,
                             add_coord=config.add_coord,
                             decoder_type=config.decoder_type)
        #stat(net, (3, 256, 256))
    elif config.net == "efficientFormer_v2_l":
        net = efficientformerv2_l(pretrained=True, edge_info=config.edge_info,)
    elif config.net == "efficientFormer_v2_s0":
        net = efficientformerv2_s0(pretrained=True, edge_info=config.edge_info,)
    elif config.net == "mobile_vit_v2":
        net = mobile_vit_v2()
    elif config.net == "swin_v2":
        net = swin_v2()
    else:
        assert False
    return net


def get_criterions(config):
    criterions = list()
    for k in range(config.label_num):
        if config.criterions[k] == "AWingLoss":
            criterion = AWingLoss()
        elif config.criterions[k] == "smoothl1":
            criterion = SmoothL1Loss()
        elif config.criterions[k] == "l1":
            criterion = F.l1_loss
        elif config.criterions[k] == 'l2':
            criterion = F.mse_loss
        elif config.criterions[k] == "STARLoss":
            criterion = STARLoss(dist=config.star_dist, w=config.star_w)
        elif config.criterions[k] == "STARLoss_v2":
            criterion = STARLoss_v2(dist=config.star_dist, w=config.star_w)
        else:
            assert False
        criterions.append(criterion)
    return criterions


def set_environment(config):
    if config.device_id >= 0:
        assert torch.cuda.is_available() and torch.cuda.device_count() > config.device_id
        torch.cuda.empty_cache()
        config.device = torch.device("cuda", config.device_id)
        config.use_gpu = True
    else:
        config.device = torch.device("cpu")
        config.use_gpu = False

    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_flush_denormal(True)  # ignore extremely small value
    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    model.cuda()

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
                
def forward(config, test_loader, net, student= False, val_dataloader=None):
    # ave_metrics = [[0, 0] for i in range(config.label_num)]
    list_nmes = [[] for i in range(config.label_num)]
    list_mses = [[] for i in range(config.label_num)]
    metric_nme = NME(nme_left_index=config.nme_left_index,
                     nme_right_index=config.nme_right_index)
    metric_mse = MSE()
    metric_fr_auc = FR_AUC(data_definition=config.data_definition)

    output_pd = None

    net = net.float().to(config.device)
    net.eval()
    #jit_model = torch.jit.script(net, torch.randn((1,3,256,256)).cuda(), strict=False)
    #torch.jit.save(jit_model, "mobilenetv2_base.jit.pt")
    # test quant
    # with torch.no_grad():
    #     data = iter(val_dataloader)
    #     data = data.next()['data'].float().to(confi)
    #     jit_model = torch.jit.trace(net )
    #     torch.jit.save(jit_model, "mobilenetv2_base.jit.pt")
    #baseline_model = torch.jit.load("mobilenetv2_base.jit.pt").eval()
    #quant_modules.initialize()
    #calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(test_loader,
    #                                          use_cache=False,
    #                                          algo_type=torch_tensorrt.ptq.CalibrationAlgo.MINMAX_CALIBRATION,
    #                                          device=torch.device('cuda:0'))

    #compile_spec = {
    #         "inputs": [torch_tensorrt.Input([64, 3, 256, 256])],
    #         "enabled_precisions": torch.int8,
    #         "calibrator": calibrator,
    #         "truncate_long_and_double": True
            
    #     }
    # net = torch_tensorrt.compile(baseline_model, **compile_spec)
    dataset_size = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    if config.logger is not None:
        config.logger.info("Forward process, Dataset size: %d, Batch size: %d" % (
            dataset_size, batch_size))
    for i, sample in enumerate(tqdm(test_loader)):
        input = sample["data"].float().to(config.device, non_blocking=True)
        labels = list()
        if isinstance(sample["label"], list):
            for label in sample["label"]:
                label = label.float().to(config.device, non_blocking=True)
                labels.append(label)
        else:
            label = sample["label"].float().to(
                config.device, non_blocking=True)
            for k in range(label.shape[1]):
                labels.append(label[:, k])
        labels = config.nstack * labels

        with torch.no_grad():
            if student:
                output, heatmap, landmarks, _ = net(input)
            else:
                output, heatmap, landmarks, _ = net(input)

            # metrics
        if student:
            for k in range(config.label_num):
                if config.metrics[k] is not None:
                    list_nmes[k] += metric_nme.test(landmarks, labels[k])
                    list_mses[k] += metric_mse.test(landmarks, labels[k])
        else:
            for k in range(config.label_num):
                if config.metrics[k] is not None:
                    list_nmes[k] += metric_nme.test(output[k], labels[k])
                    list_mses[k] += metric_mse.test(output[k], labels[k])

    metrics = [[np.mean(nmes), ] + metric_fr_auc.test(nmes) + [np.mean(list_mses[idx]), ]
               for idx, nmes in enumerate(list_nmes)]  

    return output_pd, metrics

def compute_student_loss(config, teacher_output ,teacher_heatmap, teacher_labels , student_output, student_heatmap,  student_labels, student_inter_feat, teacher_inter_feat, labels, criterions):
    batch_weight = 1.0
    sum_loss = 0
    losses = list()
    # star loss
    # for k in range(config.label_num):
    #     if config.criterions[k] in ['smoothl1', 'l1', 'l2', 'WingLoss', 'AWingLoss']:
    #         loss = criterions[k](student_output[k], labels[k])
    #     elif config.criterions[k] in ["STARLoss", "STARLoss_v2"]:
    #         _k = int(k / 3) if config.use_AAM else k
    #         loss = criterions[k](student_heatmap[_k], labels[k])
    #     else:
    #         assert NotImplementedError
    #     loss = batch_weight * loss
    #     sum_loss += config.loss_weights[k] * loss
    #     loss = float(loss.data.cpu().item())
    #     losses.append(loss)
        
    # # KD heatmap loss
    # kd_loss = nn.MSELoss()(teacher_heatmap[-1], student_heatmap[-1])
    # sum_loss += 1000 * kd_loss
    # kd_loss = float(kd_loss.data.cpu().item())
    # losses.append(1000 * kd_loss)
    
    # test
    loss =  nn.MSELoss()(teacher_inter_feat, student_inter_feat)
    sum_loss = loss
    loss = float(loss.data.cpu().item())
    losses.append( loss)
    # KD landmark loss
    # kd_loss_2 = nn.SmoothL1Loss()(student_output[0], teacher_output[0])
    # sum_loss += 10 * kd_loss_2
    # kd_loss_2 = float(kd_loss_2.data.cpu().item())
    # losses.append(10 * kd_loss_2)
    #teacher_heatmap_sm = teacher_heatmap[-1].reshape((-1, 51, 4096))
    #teacher_heatmap_sm = torch.nn.functional.softmax(teacher_heatmap_sm, dim=2).reshape((-1, 51, 64, 64))
    #teacher_heatmap = teacher_heatmap[-1] / teacher_heatmap[-1].sum([2,3]).unsqueeze(-1).unsqueeze(-1)
    
    #kd_loss = nn.SmoothL1Loss()(student_labels, teacher_labels)
    
    # python main.py --mode=train_student --device_ids=0 --image_dir=images/ --annot_dir=./annotations/ --data_definition=ivslab --learn_rate 0.0001 --batch_size 32
    # gt loss 0.05
    #gt_loss = nn.L1Loss()(student_labels, gt_labels[0])
    #sum_loss =  gt_loss + kd_loss
    #return [float(kd_loss.data.cpu().item()), float(gt_loss.data.cpu().item())], sum_loss
    return losses, sum_loss

def compute_loss(config, criterions, output, labels, heatmap=None, landmarks=None):
    batch_weight = 1.0
    sum_loss = 0
    losses = list()
    for k in range(config.label_num):
        if config.criterions[k] in ['smoothl1', 'l1', 'l2', 'WingLoss', 'AWingLoss']:
            loss = criterions[k](output[k], labels[k])
        elif config.criterions[k] in ["STARLoss", "STARLoss_v2"]:
            _k = int(k / 3) if config.use_AAM else k
            loss = criterions[k](heatmap[_k], labels[k])
        else:
            assert NotImplementedError
        loss = batch_weight * loss
        sum_loss += config.loss_weights[k] * loss
        loss = float(loss.data.cpu().item())
        losses.append(loss)
    assert torch.isnan(sum_loss).sum() == 0, print(sum_loss)
    return losses, sum_loss

def forward_backward_student(config, train_loader, teacher_net, student_net, criterions ,optimizer=None, epoch=None):
    train_model_time = AverageMeter()
    ave_losses = [0] * config.label_num
    
    student_net = student_net.float().to(config.device)
    student_net.train(True)
    teacher_net = teacher_net.float().to(config.device)
    teacher_net.eval()
    
    dataset_size = len(train_loader.dataset)
    batch_size = config.batch_size  # train_loader.batch_size
    
    if config.logger is not None:
        config.logger.info(config.note)
        config.logger.info("Forward Backward process, Dataset size: %d, Batch size: %d" % (
            dataset_size, batch_size))
        
    iter_num = len(train_loader)
    epoch_start_time = time.time()
    
    for iter, sample in enumerate(train_loader):
        iter_start_time = time.time()
        # input
        input = sample["data"].float().to(config.device, non_blocking=True)
        # labels
        labels = list()
        if isinstance(sample["label"], list):
            for label in sample["label"]:
                label = label.float().to(config.device, non_blocking=True)
                labels.append(label)
        else:
            label = sample["label"].float().to(
                config.device, non_blocking=True)
            for k in range(label.shape[1]):
                labels.append(label[:, k])
        labels = config.nstack * labels
        # forward
        input = input.to('cuda:0')
        student_output, student_heatmaps, student_landmarks, student_inter_feat  = student_net(input)
        teacher_output, teacher_heatmaps, teacher_landmarks, teacher_inter_feat = teacher_net(input)
        
        # loss
        losses, sum_loss = compute_student_loss(
            config, teacher_output, teacher_heatmaps, teacher_landmarks, student_output, student_heatmaps, student_landmarks, student_inter_feat, teacher_inter_feat, labels, criterions  )
        ave_losses = list(map(sum, zip(ave_losses, losses)))

        # backward
        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        sum_loss.backward()
        # torch.nn.utils.clip_grad_norm_(net_module.parameters(), 128.0)
        optimizer.step()
        
        # output
        train_model_time.update(time.time() - iter_start_time)
        last_time = convert_secs2time(
            train_model_time.avg * (iter_num - iter - 1), True)
        if iter % config.display_iteration == 0 or iter + 1 == len(train_loader):
            if config.logger is not None:
                losses_str = ' Average Loss: {:.6f}'.format(
                    sum(losses) / len(losses))
                for k, loss in enumerate(losses):
                    losses_str += ', L{}: {:.3f}'.format(k, loss)
                config.logger.info(
                    ' -->>[{:03d}/{:03d}][{:03d}/{:03d}]'.format(
                        epoch, config.max_epoch, iter, iter_num)
                    + last_time + losses_str)
                
    epoch_end_time = time.time()
    epoch_total_time = epoch_end_time - epoch_start_time
    epoch_load_data_time = epoch_total_time - train_model_time.sum
    if config.logger is not None:
        config.logger.info("Train/Epoch: %d/%d, Average total time cost per iteration in this epoch: %.6f" % (
            epoch, config.max_epoch, epoch_total_time / iter_num))
        config.logger.info("Train/Epoch: %d/%d, Average loading data time cost per iteration in this epoch: %.6f" % (
            epoch, config.max_epoch, epoch_load_data_time / iter_num))
        config.logger.info("Train/Epoch: %d/%d, Average training model time cost per iteration in this epoch: %.6f" % (
            epoch, config.max_epoch, train_model_time.avg))

    ave_losses = [loss / iter_num for loss in ave_losses]
    if config.logger is not None:
        config.logger.info("Train/Epoch: %d/%d, Average Loss in this epoch: %.6f" % (
            epoch, config.max_epoch, sum(ave_losses) / len(ave_losses)))
    for k, ave_loss in enumerate(ave_losses):
        if config.logger is not None:
            config.logger.info(
                "Train/Loss%03d in this epoch: %.6f" % (k, ave_loss))
    
def forward_backward(config, train_loader, net_module, net, net_ema, criterions, optimizer, epoch):
    train_model_time = AverageMeter()
    ave_losses = [0] * config.label_num

    net_module = net_module.float().to(config.device)
    net_module.train(True)
    # for m in net_module.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #           m.eval()
    dataset_size = len(train_loader.dataset)
    batch_size = config.batch_size  # train_loader.batch_size
    batch_num = max(dataset_size / max(batch_size, 1), 1)
    if config.logger is not None:
        config.logger.info(config.note)
        config.logger.info("Forward Backward process, Dataset size: %d, Batch size: %d" % (
            dataset_size, batch_size))

    iter_num = len(train_loader)
    epoch_start_time = time.time()
    if net_module != net:
        train_loader.sampler.set_epoch(epoch)
    for iter, sample in enumerate(train_loader):
        iter_start_time = time.time()
        # input
        input = sample["data"].float().to(config.device, non_blocking=True)
        # labels
        labels = list()
        if isinstance(sample["label"], list):
            for label in sample["label"]:
                label = label.float().to(config.device, non_blocking=True)
                labels.append(label)
        else:
            label = sample["label"].float().to(
                config.device, non_blocking=True)
            for k in range(label.shape[1]):
                labels.append(label[:, k])
        labels = config.nstack * labels
        # forward
        output, heatmaps, landmarks = net_module(input)

        # loss  config, teacher_heatmap, student_heatmap, gt_labels, student_labels=None
        #losses, sum_loss = compute_student_loss(config, None, None, None, labels, landmarks)
        losses, sum_loss = compute_loss(
            config, criterions, output, labels, heatmaps, landmarks)
        ave_losses = list(map(sum, zip(ave_losses, losses)))

        # backward
        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        sum_loss.backward()
        # torch.nn.utils.clip_grad_norm_(net_module.parameters(), 128.0)
        optimizer.step()

        if net_ema is not None:
            accumulate_net(net_ema, net, 0.5 ** (config.batch_size / 10000.0))
            # accumulate_net(net_ema, net, 0.5 ** (8 / 10000.0))

        # output
        train_model_time.update(time.time() - iter_start_time)
        last_time = convert_secs2time(
            train_model_time.avg * (iter_num - iter - 1), True)
        if iter % config.display_iteration == 0 or iter + 1 == len(train_loader):
            if config.logger is not None:
                losses_str = ' Average Loss: {:.6f}'.format(
                    sum(losses) / len(losses))
                for k, loss in enumerate(losses):
                    losses_str += ', L{}: {:.3f}'.format(k, loss)
                config.logger.info(
                    ' -->>[{:03d}/{:03d}][{:03d}/{:03d}]'.format(
                        epoch, config.max_epoch, iter, iter_num)
                    + last_time + losses_str)

    epoch_end_time = time.time()
    epoch_total_time = epoch_end_time - epoch_start_time
    epoch_load_data_time = epoch_total_time - train_model_time.sum
    if config.logger is not None:
        config.logger.info("Train/Epoch: %d/%d, Average total time cost per iteration in this epoch: %.6f" % (
            epoch, config.max_epoch, epoch_total_time / iter_num))
        config.logger.info("Train/Epoch: %d/%d, Average loading data time cost per iteration in this epoch: %.6f" % (
            epoch, config.max_epoch, epoch_load_data_time / iter_num))
        config.logger.info("Train/Epoch: %d/%d, Average training model time cost per iteration in this epoch: %.6f" % (
            epoch, config.max_epoch, train_model_time.avg))

    ave_losses = [loss / iter_num for loss in ave_losses]
    if config.logger is not None:
        config.logger.info("Train/Epoch: %d/%d, Average Loss in this epoch: %.6f" % (
            epoch, config.max_epoch, sum(ave_losses) / len(ave_losses)))
    for k, ave_loss in enumerate(ave_losses):
        if config.logger is not None:
            config.logger.info(
                "Train/Loss%03d in this epoch: %.6f" % (k, ave_loss))


def accumulate_net(model1, model2, decay):
    """
        operation: model1 = model1 * decay + model2 * (1 - decay)
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(
            other=par2[k].data.to(par1[k].data.device),
            alpha=1 - decay)

    par1 = dict(model1.named_buffers())
    par2 = dict(model2.named_buffers())
    for k in par1.keys():
        if par1[k].data.is_floating_point():
            par1[k].data.mul_(decay).add_(
                other=par2[k].data.to(par1[k].data.device),
                alpha=1 - decay)
        else:
            par1[k].data = par2[k].data.to(par1[k].data.device)


def save_model(config, epoch, net, net_ema, optimizer, scheduler, pytorch_model_path):
    # save pytorch model
    state = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch
    }
    if config.ema:
        state["net_ema"] = net_ema.state_dict()

    torch.save(state, pytorch_model_path)
    if config.logger is not None:
        config.logger.info(
            "Epoch: %d/%d, model saved in this epoch" % (epoch, config.max_epoch))
