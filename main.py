import os.path
import sys
import copy
import torch
from config import get_config
import time
from utils import _grid
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
# from tqdm import tqdm
import numpy as np
from torch import optim
from Dataset import get_dataset_dl
from network import Network
from torchsummary import summary
from utils import fig_hist
from network import VGG16
from utils import CPv
import warnings
import platform

warnings.filterwarnings('ignore')

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('Agg')


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def loss_epoch(model, loss_func, dataset_dl, opt=None):
    run_loss = 0.0
    t_metric = 0.0
    len_data = len(dataset_dl.dataset)
    # internal loop over dataset
    for xb, yb in dataset_dl:
        # move batch to device
        xb = torch.as_tensor(xb)
        yb = torch.as_tensor(yb)
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)  # get model output
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)  # get loss per batch
        run_loss += loss_b  # update running loss

        if metric_b is not None:  # update running metric
            t_metric += metric_b

    loss = run_loss / float(len_data)  # average loss value
    metric = t_metric / float(len_data)  # average metric value

    return loss, metric


def loss_batch(loss_func, output, target, opt=None):
    # 计算损失
    loss = loss_func(output, target)

    # 获取预测的类别
    pred = output.argmax(dim=1, keepdim=True)

    # 计算性能指标
    metric_b = pred.eq(target.view_as(pred)).sum().item()

    # 如果提供了优化器，则进行梯度下降
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


def train_val(model, params, verbose=True, textBrowser=None):
    # Get the parameters
    epochs = params["epochs"]
    loss_func = params["f_loss"]
    opt = params["optimiser"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_scheduler = params["lr_change"]
    weight_path = params["weight_path"]

    # loss_history和metric_history用于绘图
    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    ''' Train Model n_epochs '''
    for epoch in range(epochs):

        ''' Get the Learning Rate '''
        current_lr = get_lr(opt)
        if verbose:
            ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
            print(ttt + 'Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))
            textBrowser.append(ttt + 'Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))
            # 设置滚动条到最低部
            textBrowser.ensureCursorVisible()  # 游标可用
            cursor = textBrowser.textCursor()  # 设置游标
            pos = len(textBrowser.toPlainText())  # 获取文本尾部的位置
            cursor.setPosition(pos)  # 游标位置设置为尾部
            textBrowser.setTextCursor(cursor)  # 滚动到游标位置

        '''

        Train Model Process

        '''

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)

        # collect losses
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        '''

        Evaluate Model Process

        '''

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)

        # store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            # store weights into a local file
            torch.save(model.state_dict(), weight_path.format(config.DATA.name))
            if verbose:
                ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
                print(ttt + "Copied best model weights!")
                textBrowser.append(ttt + "Copied best model weights!")
                # 设置滚动条到最低部
                textBrowser.ensureCursorVisible()  # 游标可用
                cursor = textBrowser.textCursor()  # 设置游标
                pos = len(textBrowser.toPlainText())  # 获取文本尾部的位置
                cursor.setPosition(pos)  # 游标位置设置为尾部
                textBrowser.setTextCursor(cursor)  # 滚动到游标位置

        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        # learning rate schedule
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            if verbose:
                ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
                print(ttt + "Loading best model weights!")
                textBrowser.append(ttt + "Loading best model weights!")
                # 设置滚动条到最低部
                textBrowser.ensureCursorVisible()  # 游标可用
                cursor = textBrowser.textCursor()  # 设置游标
                pos = len(textBrowser.toPlainText())  # 获取文本尾部的位置
                cursor.setPosition(pos)  # 游标位置设置为尾部
                textBrowser.setTextCursor(cursor)  # 滚动到游标位置
            model.load_state_dict(best_model_wts)

        if verbose:
            ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
            print(ttt + f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100 * val_metric:.2f}")
            textBrowser.append(
                ttt + f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100 * val_metric:.2f}")
            # 设置滚动条到最低部
            textBrowser.ensureCursorVisible()  # 游标可用
            cursor = textBrowser.textCursor()  # 设置游标
            pos = len(textBrowser.toPlainText())  # 获取文本尾部的位置
            cursor.setPosition(pos)  # 游标位置设置为尾部
            textBrowser.setTextCursor(cursor)  # 滚动到游标位置
            print("-" * 10)
            textBrowser.append("-" * 10)
            # 设置滚动条到最低部
            textBrowser.ensureCursorVisible()  # 游标可用
            cursor = textBrowser.textCursor()  # 设置游标
            pos = len(textBrowser.toPlainText())  # 获取文本尾部的位置
            cursor.setPosition(pos)  # 游标位置设置为尾部
            textBrowser.setTextCursor(cursor)  # 滚动到游标位置

            # load best model weights
    model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history


def inference(model, dataset_dl, device, config):
    len_data = float(len(dataset_dl) * config.TEST.batch_size)
    model = model.to(device)  # move model to device
    metric_c = 0
    model.eval()
    with torch.no_grad():
        # with tqdm(total=100) as pbar:
        for xb, yb in dataset_dl:
            xb = torch.as_tensor(xb)
            yb = torch.as_tensor(yb)
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)
            pred = output.argmax(dim=1, keepdim=True)
            metric_b = pred.eq(yb.view_as(pred)).sum().item()
            metric_c += metric_b
            # pbar.update(100 / len(dataset_dl))

    return float(metric_c / len_data) * 100


def main_training(dataset_name, output_folder_path, Device, yaml_filename, textBrowser):
    # print(type(textBrowser))
    global device
    if Device == 'GPU':
        device = torch.device("cuda")
    elif Device == 'CPU':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global config
    if os.path.exists(yaml_filename):
        config = get_config(yaml_filename)
    else:
        config = get_config('configs/configMNIST.yaml')

    # 定义模型参数
    params_model = {
        "shape_in": (config.DATA.shape_in_c, config.DATA.shape_in_h, config.DATA.shape_in_w),
        "initial_filters": config.MODEL.initial_filters,
        "num_fc1": config.MODEL.num_fc,
        "dropout_rate": config.MODEL.dropout_rate,
        "num_classes": config.TRAIN.epoch_num,
        "weight_path": config.MODEL.weight_path
    }

    params_train = {
        "train": None,
        "val": None,
        "epochs": config.TRAIN.epoch_num,
        "optimiser": None,
        "lr_change": None,
        "f_loss": nn.NLLLoss(reduction="sum"),
        "weight_path": output_folder_path + '/' + 'ui.pt',
    }
    config.DATA.name = dataset_name
    train_dl, val_dl = get_dataset_dl(config=config)

    cnn_model = Network(config).to(device=device)
    if config.TRAIN.OPTIMIZER.NAME == 'Adam':
        opt = optim.Adam(cnn_model.parameters(), lr=config.TRAIN.lr)
    else:
        opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
    params_train['train'] = train_dl
    params_train['val'] = val_dl
    params_train['optimiser'] = opt
    params_train['lr_change'] = ReduceLROnPlateau(opt,
                                                  mode='min',
                                                  factor=0.5,
                                                  patience=20,
                                                  verbose=False)
    summary(cnn_model, input_size=params_model['shape_in'], device=device.type)
    cnn_model, loss_hist, metric_hist = train_val(cnn_model, params_train, textBrowser=textBrowser)
    fig_hist(params=params_train, loss_hist=loss_hist, metric_hist=metric_hist, config=config)
    # main_testing()


def main_testing(dataset_name, pth_path, Device, yaml_filename, textBrowser):
    global device
    if Device == 'GPU':
        device = torch.device("cuda")
    elif Device == 'CPU':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global config
    if os.path.exists(yaml_filename):
        config = get_config(yaml_filename)
    else:
        config = get_config('configs/configMNIST.yaml')

    config.DATA.name = dataset_name
    # 定义模型参数
    params_model = {
        "shape_in": (config.DATA.shape_in_c, config.DATA.shape_in_h, config.DATA.shape_in_w),
        "initial_filters": config.MODEL.initial_filters,
        "num_fc1": config.MODEL.num_fc,
        "dropout_rate": config.MODEL.dropout_rate,
        "num_classes": config.TRAIN.epoch_num,
        "weight_path": config.MODEL.weight_path if pth_path == None else pth_path
    }
    train_set, val_set = get_dataset_dl(config=config)
    ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
    print(ttt, len(val_set), 'samples found')
    textBrowser.append(str(len(val_set)) + ' samples found')
    # 设置滚动条到最低部
    textBrowser.ensureCursorVisible()  # 游标可用
    cursor = textBrowser.textCursor()  # 设置游标
    pos = len(textBrowser.toPlainText())  # 获取文本尾部的位置
    cursor.setPosition(pos)  # 游标位置设置为尾部
    textBrowser.setTextCursor(cursor)  # 滚动到游标位置
    cnn_model = Network(config).to(device=device)
    cnn_model.load_state_dict(torch.load(params_model["weight_path"]))
    acc = inference(model=cnn_model, dataset_dl=val_set, device=device, config=config)
    ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
    print(ttt + 'Test accuracy on test set: %0.2f%%\n' % acc)
    textBrowser.append(ttt + 'Test accuracy on test set: %0.2f%%\n' % acc)
    # 设置滚动条到最低部
    textBrowser.ensureCursorVisible()  # 游标可用
    cursor = textBrowser.textCursor()  # 设置游标
    pos = len(textBrowser.toPlainText())  # 获取文本尾部的位置
    cursor.setPosition(pos)  # 游标位置设置为尾部
    textBrowser.setTextCursor(cursor)  # 滚动到游标位置


from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


def predict_img(img_path, weight_path, textBrowser=None):
    is_fashion = True if "Fashion" in weight_path else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config('configs/configMNIST.yaml')
    model = Network(config).to(device=device)
    model.load_state_dict(torch.load(weight_path))

    # 加载输入图像
    # image_path = 'path/to/your/image.jpg'
    image = Image.open(img_path)
    # 预处理输入图像
    image_tensor = transform(image)
    # image_tensor = image_tensor.unsqueeze(0)

    # 将模型设置为评估模式
    model.eval()

    # 使用模型进行预测
    with torch.no_grad():
        output = model(image_tensor.to(device=device))

    # 获取预测结果
    _, predicted = torch.max(output, 1)

    res = predicted.item()

    ls = ["T恤", "裤子", "套衫", "裙子", "外套", "凉鞋", "汗衫", "运动鞋", "包", "踝靴"]
    if is_fashion:
        res = ls[predicted.item()]
    ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
    textBrowser.append(ttt + "预测结果为:{}".format(res))
    # 设置滚动条到最低部
    textBrowser.ensureCursorVisible()  # 游标可用
    cursor = textBrowser.textCursor()  # 设置游标
    pos = len(textBrowser.toPlainText())  # 获取文本尾部的位置
    cursor.setPosition(pos)  # 游标位置设置为尾部
    textBrowser.setTextCursor(cursor)  # 滚动到游标位置
    # 打印预测结果
    ttt = time.strftime("【%Y-%m-%d %H:%M】 ", time.localtime())
    print(ttt + predicted.item())
