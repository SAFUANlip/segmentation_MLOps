from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchinfo import summary
import torch
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import *
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from fit import fit, CLASSES, val_step
from segmentation_models_pytorch.losses import LovaszLoss


SAVE_DIR = "../results/"
DATA_DIR: str = '../data/'


x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def train():
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=transforms.ToTensor(),
        classes=CLASSES,
        img_shape=1024
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=None,
        preprocessing=transforms.ToTensor(),
        classes=CLASSES,
        img_shape=1024
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4)

    encoder_name = "timm-mobilenetv3_small_100"

    criterion = nn.CrossEntropyLoss() #LovaszLoss(mode='multiclass')

    model = torch.load('../runs/Linknet_timm-mobilenetv3_small_100/Linknet_timm-mobilenetv3_small_100_mIoU-0.661.pt')
    # smp.Linknet(
    #     encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights='imagenet',
    #     in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=len(CLASSES),  # model output channels (number of classes in your dataset)
    # )

    max_lr = 1e-2
    epoch = 60
    weight_decay = 1e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = fit(epoch, model, train_loader, valid_loader, criterion, optimizer, sched, save_path='../runs/', encoder_name=encoder_name)


def test(show=False, save=False):
    best_model = torch.load(
        '../runs/CrossEntropyLoss_Linknet_timm-mobilenetv3_small_100/CrossEntropyLoss_Linknet_timm-mobilenetv3_small_100_mIoU-0.649.pt')

    device_test = 'cpu'
    sample = torch.rand(1, 1, 1024,1024).to(device_test)
    scripted_model = torch.jit.trace(best_model.to(device_test), sample.to(device_test))
    scripted_model.save('script_cpu_seg_5cls_update.pth')

    best_model.eval()
    test_dataset = Dataset(
        x_test_dir, y_test_dir,
        augmentation=None,
        preprocessing=transforms.ToTensor(),
        classes=CLASSES,
    )
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=4)


    device = 'cuda'
    best_model.to(device)
    criterion = LovaszLoss(mode='multiclass')
    loss, iou, acc = val_step(best_model, device, test_loader, criterion=criterion)

    print(
          "Test Loss: {:.3f}..".format(loss),
          "Test mIoU:{:.3f}..".format(iou),
          "Test Acc:{:.3f}..".format(acc),
    )

    for i in range(len(test_dataset)):
        #n = np.random.choice(len(test_dataset))

        image_vis = test_dataset[i][0].squeeze() #.astype('uint8')
        img_name = test_dataset.ids[i]
        image, gt_mask = test_dataset[i]

        gt_mask = gt_mask.squeeze()

        x_tensor = image.unsqueeze(0).to('cuda')
        pr_mask = best_model(x_tensor)
        pr_mask = torch.argmax(F.softmax(pr_mask, dim=1), dim=1)
        pr_mask = (pr_mask.squeeze().cpu().detach().numpy()) * 255//(len(CLASSES)-1)

        if save:
            cv2.imwrite(SAVE_DIR+"/lovaz/"+'seg_'+img_name, pr_mask)

        if show:
            visualize(
                image=image_vis,
                ground_truth_mask=gt_mask,
                predicted_mask=pr_mask
            )

if __name__ == '__main__':

    #train()
    test(show=False, save=False)


