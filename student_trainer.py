import os
import sys
import time
import argparse
import traceback
import torch
import torch.nn as nn
from lib import utility
from lib.utils import AverageMeter, convert_secs2time

os.environ["MKL_THREADING_LAYER"] = "GNU"

def student_train(args):
    device_ids = args.device_ids
    nprocs = len(device_ids)
    if nprocs > 1:
        torch.multiprocessing.spawn(
            train_worker, args=(nprocs, 1, args), nprocs=nprocs,
            join=True)
    elif nprocs == 1:
        train_worker(device_ids[0], nprocs, 1, args)
    else:
        assert False
        
def train_worker(world_rank, world_size, nodes_size, args):
    # initialize config.
    config = utility.get_config(args)
    config.device_id = world_rank if nodes_size == 1 else world_rank % torch.cuda.device_count()
    # set environment
    utility.set_environment(config)
    # initialize instances, such as writer, logger and wandb.
    if world_rank == 0:
        config.init_instance()

    if config.logger is not None:
        config.logger.info("\n" + "\n".join(["%s: %s" % item for item in config.__dict__.items()]))
        config.logger.info("Loaded configure file %s: %s" % (config.type, config.id))


    # student model 
    student_net = utility.get_student_net(config)
    student_net.train(True)
    student_net = student_net.float().to(config.device)
    
    # teacher model
    teacher_net = utility.get_teacher_net(config)
    teacher_net = teacher_net.float().to(config.device)
    teacher_net.eval()
    
    if config.ema and world_rank == 0:
        net_ema = utility.get_student_net(config)
        net_ema = net_ema.float().to(config.device)
        net_ema.eval()
        utility.accumulate_net(net_ema, student_net, 0)
    else:
        net_ema = None

    # multi-GPU training
    net_module = teacher_net
    # load pretrain teacher model
    try:
        pretrained_weight = "/disk2/icml/STAR/ivslab/stackedHGnet_v1_0.0310/model/best_model.pkl"
        checkpoint = torch.load(pretrained_weight)
        teacher_net.load_state_dict(checkpoint["net"], strict=False)
        if net_ema is not None:
            net_ema.load_state_dict(checkpoint["net_ema"], strict=False)
        if config.logger is not None:
            config.logger.warn("Successed to load pretrain model %s." % pretrained_weight)
        start_epoch = 0
    except:
        start_epoch = 0
        if config.logger is not None:
            config.logger.warn("Failed to load pretrain model %s." % pretrained_weight)

    if config.logger is not None:
        config.logger.info("Loaded network")

    # data - train, val
    train_loader = utility.get_dataloader(config, "train", world_rank, world_size)
    if world_rank == 0:
        val_loader = utility.get_dataloader(config, "val")
    if config.logger is not None:
        config.logger.info("Loaded data")

    # forward & backward
    if config.logger is not None:
        config.logger.info("Optimizer type %s. Start training..." % (config.optimizer))
    if not os.path.exists(config.model_dir) and world_rank == 0:
        os.makedirs(config.model_dir)

    # training
    best_metric, best_net = None, None
    epoch_time, eval_time = AverageMeter(), AverageMeter()
    optimizer = utility.get_optimizer(config, student_net)
    scheduler = utility.get_scheduler(config, optimizer)
    for i_epoch, epoch in enumerate(range(config.max_epoch + 1)):
        try:
            epoch_start_time = time.time()
            if epoch >= start_epoch:
                # forward and backward
                if epoch != start_epoch:
                    utility.forward_backward_student(config, train_loader, teacher_net, student_net, optimizer, epoch)

                # validating
                if epoch % config.val_epoch == 0 and epoch != 0 and world_rank == 0:
                    eval_start_time = time.time()
                    epoch_nets = {"net": student_net, "net_ema": student_net}
                    for net_name, epoch_net in epoch_nets.items():
                        if epoch_net is None:
                            continue
                        result, metrics = utility.forward(config, val_loader, student_net, student=True)
                        for k, metric in enumerate(metrics):
                            if config.logger is not None and len(metric) != 0:
                                config.logger.info(
                                    "Val_{}/Metric{:3d} in this epoch: [NME {:.6f}, FR {:.6f}, AUC {:.6f}, MSE {:.6f}]".format(
                                        net_name, k, metric[0], metric[1], metric[2], metric[3]))

                        # update best model.
                        cur_metric = metrics[config.key_metric_index][0]
                        if best_metric is None or best_metric > cur_metric:
                            best_metric = cur_metric
                            best_net = epoch_net
                            current_pytorch_model_path = os.path.join(config.model_dir, "best_model.pkl")
                            # current_onnx_model_path = os.path.join(config.model_dir, "train.onnx")
                            utility.save_model(
                                config,
                                epoch,
                                best_net,
                                net_ema,
                                optimizer,
                                scheduler,
                                current_pytorch_model_path)
                    if best_metric is not None:
                        config.logger.info(
                            "Val/Best_Metric%03d in this epoch: %.6f" % (config.key_metric_index, best_metric))
                    eval_time.update(time.time() - eval_start_time)

                # saving model
                if epoch == config.max_epoch and world_rank == 0:
                    current_pytorch_model_path = os.path.join(config.model_dir, "last_model.pkl")
                    # current_onnx_model_path = os.path.join(config.model_dir, "model_epoch_%s.onnx" % epoch)
                    utility.save_model(
                        config,
                        epoch,
                        student_net,
                        net_ema,
                        optimizer,
                        scheduler,
                        current_pytorch_model_path)

                
            # adjusting learning rate
            if epoch > 0:
                scheduler.step()
            epoch_time.update(time.time() - epoch_start_time)
            last_time = convert_secs2time(epoch_time.avg * (config.max_epoch - i_epoch), True)
            if config.logger is not None:
                config.logger.info(
                    "Train/Epoch: %d/%d, Learning rate decays to %s, " % (
                        epoch, config.max_epoch, str(scheduler.get_last_lr())) \
                    + last_time + 'eval_time: {:4.2f}, '.format(eval_time.avg) + '\n\n')

        except:
            traceback.print_exc()
            config.logger.error("Exception happened in training steps")

    if config.logger is not None:
        config.logger.info("Training finished")

    try:
        if config.logger is not None and best_metric is not None:
            new_folder_name = config.folder + '-fin-{:.4f}'.format(best_metric)
            new_work_dir = os.path.join(config.ckpt_dir, config.data_definition, new_folder_name)
            os.system('mv {} {}'.format(config.work_dir, new_work_dir))
    except:
        traceback.print_exc()

