import dataset
from darknet53 import *
import torch.utils.data as data
import torch
import os

save_path = r'F:\yolo\new\param'
save_param =r'para_net.pt'

def loss_fn(output, target, alpha):
    output = output.permute(0, 2, 3, 1)
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

    mask_obj = target[..., 0] > 0
    mask_noobj = target[..., 0] == 0
    loss_obj = loss_f1(output[mask_obj].double(),target[mask_obj])
    loss_noobj = loss_f1(output[mask_noobj].double(),target[mask_noobj])
    # loss_obj = torch.mean(torch.sum((output[mask_obj].double() - target[mask_obj]) ** 2))
    # loss_noobj = torch.mean(torch.sum((output[mask_noobj].double() - target[mask_noobj]) ** 2))

    loss = alpha * loss_obj + (1 - alpha) * loss_noobj

    return loss


if __name__ == '__main__':

    myDataset = dataset.MyDataset()
    train_loader = data.DataLoader(myDataset, batch_size=2, shuffle=True,num_workers=8)

    net = MainNet()
    net.train()
    # net.load_state_dict(torch.load(os.path.join(save_path,save_param)))

    opt = torch.optim.Adam(net.parameters())
    for epoch in range(5000):
        for target_13, target_26, target_52, img_data in train_loader:
            output_13, output_26, output_52 = net(img_data)

            loss_f1 = nn.MSELoss(reduction ='sum')
            # print(target_26[:,:,0,0])
            # print(output_26[:,:,0,0])
            loss_13 = loss_fn(output_13, target_13, 0.9)
            loss_26 = loss_fn(output_26, target_26, 0.9)
            loss_52 = loss_fn(output_52, target_52, 0.9)

            loss = loss_13 + loss_26 + loss_52
            # print(loss.type())

            opt.zero_grad()
            loss.backward()
            opt.step()

            print('epoch:',epoch,'总损失：',loss.item(),'loss_13',loss_13.item(),'loss_26',loss_26.item(),'loss_52',loss_52.item())

        if epoch %100 ==0:
            if not os.path.exists(os.path.join(save_path, str(epoch))):
                os.makedirs(os.path.join(save_path, str(epoch)))
            torch.save(net.state_dict(), os.path.join(save_path, str(epoch), save_param))
