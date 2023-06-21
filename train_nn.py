import os
import time
from typing import Any, Text, Mapping
import torch
import torch_geometric as pyg
from data.gden import GraphDynamicEarthNet
from helper import parser
from network import models, pad
from network import wrapper as wrapper_lib
from utils.setup_helper import *

from torch.utils.tensorboard import SummaryWriter

def main(opts: Any, config: Mapping[Text, Any]) -> None:
    """Runs the training.
    Args:
        opts (Any): Options specifying the training configuration.
    """
    log = open(log_file, 'a')
    log_print = lambda ms: parse.log(ms, log)

    # Define network
    network = models.define_model(config, model=config['NETWORK']['NAME'], gpu_ids=opts.gpu_ids)
    log_print(str(network))
    log_print(f'NUMBER OF PARAMS: {sum(p.numel() for p in network.parameters() if p.requires_grad)}')
    log_print(f"Init {config['NETWORK']['NAME']} as network")

    # Configure data
    data_config = config['DATA']
    LABEL_COLORS = np.array([(96, 96, 96), (204, 204, 0), (0, 204, 0), (0, 0, 153), (153, 76, 0), (0, 128, 255), (138, 178, 198)])
    LABEL_NAMES = ["impervious surface", "agriculture", "forest & other vegetation", "wetlands", "soil", "water", "snow & ice"]

    train_data = GraphDynamicEarthNet(den_root=data_config['DEN_ROOT'], gden_root=data_config['GDEN_ROOT'], mode=data_config["TRAIN_SUBSET"],
                                    k_slic=data_config["SLIC_K"], smoothness=data_config["SLIC_SMOOTHNESS"])
    val_data = GraphDynamicEarthNet(den_root=data_config['DEN_ROOT'], gden_root=data_config['GDEN_ROOT'], mode=data_config["VAL_SUBSET"],
                                    k_slic=data_config["SLIC_K"], smoothness=data_config["SLIC_SMOOTHNESS"])
    test_data = GraphDynamicEarthNet(den_root=data_config['DEN_ROOT'], gden_root=data_config['GDEN_ROOT'], mode=data_config["TEST_SUBSET"],
                                    k_slic=data_config["SLIC_K"], smoothness=data_config["SLIC_SMOOTHNESS"])

    collate_fn = lambda x: pad.pad_collate(x, pad_value=config['NETWORK']['PAD_VALUE'])
    train_loader = pyg.loader.DataLoader(train_data, batch_size=config['TRAINING']['BATCH_SIZE'],
                                               shuffle=True, num_workers=data_config['NUM_WORKERS'],
                                               pin_memory=True, drop_last=False, collate_fn=collate_fn)
    val_loader = pyg.loader.DataLoader(val_data, batch_size=config['EVALUATION']['BATCH_SIZE'],
                                             shuffle=False, num_workers=data_config['NUM_WORKERS'],
                                             pin_memory=True, drop_last=False, collate_fn=collate_fn)
    test_loader = pyg.loader.DataLoader(test_data, batch_size=config['EVALUATION']['BATCH_SIZE'],
                                             shuffle=False, num_workers=data_config['NUM_WORKERS'],
                                             pin_memory=True, drop_last=False, collate_fn=collate_fn)

    iter_per_epoch = len(train_loader)
    wrapper = wrapper_lib.NetworkWrapper(network, iter_per_epoch, opts, config)

    log_print(f"Load datasets from {data_config['GDEN_ROOT']}: train_set={len(train_data)} val_set={len(val_data)} test_set={len(test_data)}")
    
    # Initialize a Tensorboard writer to track loss and metric values
    writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(log_file),"tensorboard/"), comment=f"_{config['NETWORK']['NAME']}_LR_{config['TRAINING']['LR']}_BS_{config['TRAINING']['BATCH_SIZE']}")

    # Initialize validation and test metrics
    best_epoch = 0
    best_mIoU = wrapper.best_acc
    best_Acc = 0.0
    best_cIoU = [0.0] * (len(data_config["CLASSES"])-config["LOSS"]["IGNORE_INDEX"])
    test_mIoU = 0.0
    test_Acc = 0.0
    test_cIoU = [0.0] * (len(data_config["CLASSES"])-config["LOSS"]["IGNORE_INDEX"])

    n_epochs = config['TRAINING']['EPOCHS']
    wrapper.save_ckpt(0, os.path.dirname(log_file), best_acc=0., is_best=True) # In case of n_epochs=0
    log_print(f'Start training from epoch {opts.start_epoch} to {n_epochs}, best acc: {best_mIoU}')
    for epoch in range(opts.start_epoch, n_epochs):
        start_time = time.time()
        log_print(f'\n\n>>> EPOCH {epoch+1}')
        wrapper.train_epoch(epoch, train_loader, log_print)
        wrapper.save_ckpt(epoch, os.path.dirname(log_file), best_acc=best_mIoU, last_ckpt=True)

        # Eval on training set
        metrics_train = wrapper.eval_model(epoch, train_loader, log_print)
        log_print(f'\nTraining {epoch + 1} \nnodeAcc:{metrics_train.nodeAcc:.4f} meanIoU:{metrics_train.mIoU:.4f}')
        writer.add_scalar('Loss/train', metrics_train.loss, epoch+1)
        writer.add_scalar('Acc/train', metrics_train.nodeAcc, epoch+1)
        writer.add_scalar('mIoU/train', metrics_train.mIoU, epoch+1)
        writer.add_scalars('IoU/train',dict(zip(LABEL_NAMES, metrics_train.c.values())), epoch+1)

        # Eval on validation set
        metrics_val = wrapper.eval_model(epoch, val_loader, log_print)
        log_print(f'\nEvaluate {epoch + 1} \nnodeAcc:{metrics_val.nodeAcc:.4f} meanIoU:{metrics_val.mIoU:.4f}')
        writer.add_scalar('Loss/val', metrics_val.loss, epoch+1)
        writer.add_scalar('Acc/val', metrics_val.nodeAcc, epoch+1)
        writer.add_scalar('mIoU/val', metrics_val.mIoU, epoch+1)
        writer.add_scalars('IoU/val',dict(zip(LABEL_NAMES, metrics_val.c.values())), epoch+1)

        # Eval on test set
        metrics_test = wrapper.eval_model(epoch, test_loader, log_print)
        log_print(f'\nTest {epoch + 1} \nnodeAcc:{metrics_test.nodeAcc:.4f} meanIoU:{metrics_test.mIoU:.4f}')
        writer.add_scalar('Loss/test', metrics_test.loss, epoch+1)
        writer.add_scalar('Acc/test', metrics_test.nodeAcc, epoch+1)
        writer.add_scalar('mIoU/test', metrics_test.mIoU, epoch+1)
        writer.add_scalars('IoU/test',dict(zip(LABEL_NAMES, metrics_test.c.values())), epoch+1)

        # Save the best model according the validation mIoU
        mean_iou = metrics_val.mIoU
        if mean_iou > best_mIoU:
            best_epoch = epoch+1
            best_mIoU = metrics_val.mIoU
            best_Acc = metrics_val.nodeAcc
            best_cIoU = metrics_val.c
            test_mIoU = metrics_test.mIoU
            test_Acc = metrics_test.nodeAcc
            test_cIoU = metrics_test.c
            wrapper.save_ckpt(epoch, os.path.dirname(log_file), best_acc=best_mIoU, is_best=True)
            log_print(f'>>Save best model: epoch={epoch + 1} best_iou:{best_mIoU:.4f}')

        #Program statistics
        rss, vms = get_sys_mem()
        max_gpu_mem = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        log_print(f'Memory usage: rss={rss:.2f}GB vms={vms:.2f}GB MaxGPUMem:{max_gpu_mem:.2f}GB Time:{(time.time() - start_time):.2f}s')

        # Early stopping
        if epoch > best_epoch + config["TRAINING"]["EARLY_STOPPING"]:
            log_print(f"\n >> Early stopping at {epoch+1} epoch <<")
            break
    
    # Load best model
    ckpt = load_weights(os.path.join(os.path.dirname(log_file), 'best_ckpt.pth'), wrapper.device)
    wrapper.net.load_state_dict(ckpt['model_dict'])

    # Final evaluation
    ## Eval on training set
    metrics_train = wrapper.eval_model(best_epoch, train_loader, log_print)

    ## Eval on validation set
    metrics_val = wrapper.eval_model(best_epoch, val_loader, log_print)

    ## Eval on test set
    metrics_test = wrapper.eval_model(best_epoch, test_loader, log_print)

    best_mIoU = metrics_val.mIoU
    best_Acc = metrics_val.nodeAcc
    best_cIoU = metrics_val.c
    test_mIoU = metrics_test.mIoU
    test_Acc = metrics_test.nodeAcc
    test_cIoU = metrics_test.c

    log_print("\n\n[Evaluation of the model at the best epoch]")
    log_print(f"\n\n[VAL] at {best_epoch} epoch, best_mIoU={best_mIoU} best_Acc={best_Acc} best_cIoU={best_cIoU}")
    log_print(f"\n\n[TEST] at {best_epoch} epoch, test_mIoU={test_mIoU} test_Acc={test_Acc} test_cIoU={test_cIoU}")

if __name__ == '__main__':
    parse = parser.Parser()
    opt, log_file = parse.parse()
    opt.is_Train = True
    make_deterministic(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpu_ids)

    config = parser.read_yaml_config(opt.config)
    main(opts=opt, config=config)