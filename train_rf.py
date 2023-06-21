import os
import time
from typing import Any, Text, Mapping
import torch
from data.oden import ObjectDynamicEarthNet
from helper import parser
from utils.metrics import eval_metrics
from utils.setup_helper import *
from sklearn.ensemble import RandomForestClassifier
import pickle

def main(opts: Any, config: Mapping[Text, Any]) -> None:
    """Runs the training.
    Args:
        opts (Any): Options specifying the training configuration.
    """
    log = open(log_file, 'a')
    log_print = lambda ms: parse.log(ms, log)

    # Define classifier
    clf = RandomForestClassifier(n_estimators=config['NETWORK']['N_TREES'], max_depth=config['NETWORK']['MAX_DEPTH'], random_state=opt.seed, bootstrap=True, oob_score=True)
    log_print(str(clf))

    # Configure data
    data_config = config['DATA']

    train_data = ObjectDynamicEarthNet(den_root=data_config['DEN_ROOT'], gden_root=data_config['GDEN_ROOT'], mode=data_config['TRAIN_SUBSET'],
                                        k_slic=data_config["SLIC_K"], smoothness=data_config["SLIC_SMOOTHNESS"])
    val_data = ObjectDynamicEarthNet(den_root=data_config['DEN_ROOT'], gden_root=data_config['GDEN_ROOT'], mode=data_config['VAL_SUBSET'],
                                        k_slic=data_config["SLIC_K"], smoothness=data_config["SLIC_SMOOTHNESS"])
    test_data = ObjectDynamicEarthNet(den_root=data_config['DEN_ROOT'], gden_root=data_config['GDEN_ROOT'], mode=data_config['TEST_SUBSET'],
                                        k_slic=data_config["SLIC_K"], smoothness=data_config["SLIC_SMOOTHNESS"])
    
    log_print(f"Load datasets from {data_config['GDEN_ROOT']}: train_set={len(train_data)} val_set={len(val_data)}")

    start_time = time.time()
    # Train on training set
    X = train_data[:]["region"].x.squeeze()
    y = train_data[:]["region"].y.argmax(-1).squeeze() # Take the majority class of the region as label
    ignore_class_mask = y != config['LOSS']['IGNORE_INDEX']
    X = X[ignore_class_mask]
    y = y[ignore_class_mask]
    clf.fit(X, y)
    nodeAcc = clf.score(X, y)
    log_print(f'\nTraining RandomForest \nnodeAcc:{nodeAcc:.2f} \nOOBscore:{clf.oob_score_:.2f}')
    pickle.dump(clf,open(os.path.join(os.path.dirname(log_file), "rf_model.pickle"),"wb"))

    pred = torch.as_tensor(clf.predict_proba(X.numpy()))
    correct, labeled, inter, union = eval_metrics(pred, y, len(config['DATA']['CLASSES'])-1, config['LOSS']['IGNORE_INDEX'])

    train_nodeAcc = 1.0 * correct / (np.spacing(1) + labeled)
    train_IoU = 1.0 * inter / (np.spacing(1) + union)
    train_mIoU = train_IoU.mean()

    log_print(f"\nEvaluate RandomForest on training set \nnodeAcc:{train_nodeAcc:.4f} meanIoU:{train_mIoU:.4f} c:{dict(zip(config['DATA']['CLASSES'], train_IoU))}")

    # Eval on validation set
    X = val_data[:]["region"].x.squeeze()
    y = val_data[:]["region"].y.argmax(-1).squeeze() # Take the majority class of the region as label
    ignore_class_mask = y != config['LOSS']['IGNORE_INDEX']
    X = X[ignore_class_mask]
    y = y[ignore_class_mask]

    pred = torch.as_tensor(clf.predict_proba(X.numpy()))
    correct, labeled, inter, union = eval_metrics(pred, y, len(config['DATA']['CLASSES'])-1, config['LOSS']['IGNORE_INDEX'])

    # PRINT INFO
    val_nodeAcc = 1.0 * correct / (np.spacing(1) + labeled)
    val_IoU = 1.0 * inter / (np.spacing(1) + union)
    val_mIoU = val_IoU.mean()

    log_print(f"\nEvaluate RandomForest on validation set \nnodeAcc:{val_nodeAcc:.4f} meanIoU:{val_mIoU:.4f} c:{dict(zip(config['DATA']['CLASSES'], val_IoU))}")

    # Eval on test set
    X = test_data[:]["region"].x.squeeze()
    y = test_data[:]["region"].y.argmax(-1).squeeze() # Take the majority class of the region as label
    ignore_class_mask = y != config['LOSS']['IGNORE_INDEX']
    X = X[ignore_class_mask]
    y = y[ignore_class_mask]

    pred = torch.as_tensor(clf.predict_proba(X.numpy()))
    correct, labeled, inter, union = eval_metrics(pred, y, len(config['DATA']['CLASSES'])-1, config['LOSS']['IGNORE_INDEX'])

    # PRINT INFO
    test_nodeAcc = 1.0 * correct / (np.spacing(1) + labeled)
    test_IoU = 1.0 * inter / (np.spacing(1) + union)
    test_mIoU = test_IoU.mean()

    log_print(f"\nEvaluate RandomForest on test set \nnodeAcc:{test_nodeAcc:.4f} meanIoU:{test_mIoU:.4f} c:{dict(zip(config['DATA']['CLASSES'], test_IoU))}")

    #Program statistics
    rss, vms = get_sys_mem()
    max_gpu_mem = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    log_print(f'Memory usage: rss={rss:.2f}GB vms={vms:.2f}GB MaxGPUMem:{max_gpu_mem:.2f}GB Time:{(time.time() - start_time):.2f}s')


if __name__ == '__main__':
    parse = parser.Parser()
    opt, log_file = parse.parse()
    opt.is_Train = True
    make_deterministic(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpu_ids)

    config = parser.read_yaml_config(opt.config)
    main(opts=opt, config=config)


    