import torch
import numpy as np
import wandb
import time
from tqdm import tqdm
import os
import torchvision

from get_device import get_device
from metrics import pixel_accuracy, mIoU

CLASSES: list[str] = ['border_close', 'border_further', 'water', 'bottom_surface', 'unlabelled']


def image_log(current_epoch, x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor, val=False) -> None:
    """
    Функция отвечает за логирование медиа данных при помощи библитеки wandb

    :param x: изображение, которае подается на вход нейросетевого алгоритма
    :param y: истинная сегментация входного изображения
    :param y_pred: результат работы нейросетевого алгоритма, предсказанная сегментация
    :param val: флаг показывающий из какого этапа работы неройсетевого алгоритма была вызванна данная функция
    :return:
    """
    x_cur = x[0].cpu().detach().numpy()
    y_cur = y[0].cpu().detach().numpy()
    y_pred_cur = y_pred[0].cpu().detach().numpy()

    # np.save(f'./valid_image_log/{"train" if not val else "valid"}_{current_epoch}', y_pred_cur)
    # np.save(f'./valid_image_log/orig_{"train" if not val else "valid"}_{current_epoch}', y_cur)
    # np.save(f'./valid_image_log/image_{"train" if not val else "valid"}_{current_epoch}', x_cur)

    x_img = wandb.Image(x_cur)
    y_img = wandb.Image(y_cur)
    y_pred_img = wandb.Image(y_pred_cur)

    wandb.log({'epoch': current_epoch, f'{"train" if not val else "valid"}_image': x_img,
               f'{"train" if not val else "valid"}_real': y_img,
               f'{"train" if not val else "valid"}_pred': y_pred_img})

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_step(model, device, train_loader, criterion, optimizer):
    running_loss, iou_score, accuracy = 0, 0, 0
    for i, data in enumerate(tqdm(train_loader)):
        # training phase
        image_tiles, mask_tiles = data

        image = image_tiles.to(device);
        mask = mask_tiles.to(device);
        # forward
        output = model(image)
        loss = criterion(output, mask)
        # evaluation metrics
        iou_score += mIoU(output, mask, n_classes=len(CLASSES))
        accuracy += pixel_accuracy(output, mask)
        # backward
        loss.backward()
        optimizer.step()  # update weight
        optimizer.zero_grad()  # reset gradient

        running_loss += loss.item()

    return running_loss/len(train_loader), iou_score/len(train_loader), accuracy/len(train_loader)


def val_step(model, device, val_loader, criterion, save=False, save_path=None):
    val_loss, val_iou_score, val_accuracy = 0, 0, 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader)):

            image_tiles, mask_tiles = data

            image = image_tiles.to(device);
            mask = mask_tiles.to(device);

            output = model(image)

            # evaluation metrics
            val_iou_score += mIoU(output, mask, n_classes=len(CLASSES))
            val_accuracy += pixel_accuracy(output, mask)
            # loss
            loss = criterion(output, mask)
            val_loss += loss.item()
    return val_loss/len(val_loader), val_iou_score/len(val_loader), val_accuracy/len(val_loader)


def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, save_path, encoder_name: str):

    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []
    val_iou = []; val_acc = []
    train_iou = []; train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1; not_improve=0

    device = get_device()


    save_dir = f'/{type(criterion).__name__}_{type(model).__name__}_{encoder_name}/'
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="segmentation-project",
        name=save_dir[1:-1],
        # Track hyperparameters and run metadata
        config={
            "learning_rate": get_lr(optimizer),
            "epochs": epochs,
        })

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()

        # training loop
        model.train()
        running_loss, iou_score, train_accuracy = train_step(model, device, train_loader, criterion, optimizer)

        # validation loop
        model.eval()
        val_loss, val_iou_score, val_accuracy = val_step(model, device, val_loader, criterion)

        #calculatio mean for each batch
        train_losses.append(running_loss)
        val_losses.append(val_loss)

        wandb.log({"train_loss": running_loss, "val_loss": val_loss, "lr": get_lr(optimizer),
                   "train_iou": iou_score, "val_iou": val_iou_score,
                   "train_accuracy": train_accuracy, "val_accuracy": val_accuracy})

        if min_loss > val_loss:
            print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, val_loss))
            min_loss = val_loss
            decrease += 1
            if decrease % 5 == 0:
                if not os.path.exists(save_path+save_dir):
                    os.makedirs(save_path+save_dir)
                print('saving model...')
                torch.save(model, save_path+save_dir+f'{save_dir[1:-1]}_mIoU'+'-{:.3f}.pt'.format(val_iou_score)) #Train

        else:
            not_improve += 1
            min_loss = val_loss
            print(f'Loss Not Decrease for {not_improve} time')

        #iou
        val_iou.append(val_iou_score)
        train_iou.append(iou_score)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)
        print("Epoch:{}/{}..".format(e+1, epochs),
              "Train Loss: {:.3f}..".format(running_loss),
              "Val Loss: {:.3f}..".format(val_loss),
              "Train mIoU:{:.3f}..".format(iou_score),
              "Val mIoU: {:.3f}..".format(val_iou_score),
              "Train Acc:{:.3f}..".format(train_accuracy),
              "Val Acc:{:.3f}..".format(val_accuracy),
              "Time: {:.2f}m".format((time.time()-since)/60))

        # step the learning rate
        lrs.append(get_lr(optimizer))
        scheduler.step(val_losses[-1])

    history = {'train_loss' : train_losses, 'val_loss': val_losses,
               'train_miou' :train_iou, 'val_miou':val_iou,
               'train_acc' :train_acc, 'val_acc':val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m' .format((time.time()-fit_time)/60))
    return history