import torch
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
from utils.loss import DiceLoss, TverskyLoss, BoundaryLoss, BoundaryDoULoss


def train(args, model, dataloader, device):
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    tversky_loss = TverskyLoss(args.num_classes)
    boundary_loss = BoundaryLoss(args.num_classes)
    boundary_dou_loss = BoundaryDoULoss(args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    loss = 0.
    global_step = 0
    max_iterations = args.epochs * len(dataloader)
    alpha = 1

    for epoch in range(1, args.epochs + 1):
        train_iterator = tqdm(dataloader, desc=f"Epoch: %3d/{args.epochs} | Step: %3d/{len(dataloader)} | Loss: %f" % (
            epoch, 0, loss), dynamic_ncols=True)

        for step, batch in enumerate(train_iterator):
            image, label = (batch["image"].to(device), batch["label"].to(device))

            if image.size()[1] == 1:
                image = image.repeat(1, 3, 1, 1)

            predict = model(image)

            c_loss = ce_loss(predict, label[:].long())
            d_loss = dice_loss(predict, label, softmax=True)
            t_loss = tversky_loss(predict, label[:])  # for tversky loss
            b_loss = boundary_loss(predict, label[:])  # for boundary loss
            bd_loss = boundary_dou_loss(predict, label[:])  # for boundary DoU loss
            # 89.01 3.33 || 88.81 3.48
            # loss = 0.4 * c_loss + 0.6 * d_loss
            # loss = bd_loss
            # loss = (c_loss + d_loss) * alpha + b_loss * (1-alpha)  # for boundary loss
            loss = (c_loss + d_loss) * 0.2 + bd_loss * 0.6
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            new_lr = args.base_lr * (1.0 - global_step / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            step += 1
            global_step += 1

            train_iterator.set_description(
                f"Epoch: %3d/{args.epochs} | Step: %3d/{len(dataloader)} | Loss: %f" % (epoch, step, loss))

        # ===== for boundary loss ===== #
        # alpha -= 0.01
        # alpha = max(alpha, 0.01)
        # print(alpha)
        # ===== for boundary loss ===== #

        if epoch == args.epochs:
            save_model_path = f"{args.save_dir}/model.pth"
            torch.save(model.state_dict(), save_model_path)
            print(f"\nTraining Finish. Save model to {save_model_path}")
