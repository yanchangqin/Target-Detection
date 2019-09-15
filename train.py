import torch
import os

import torch.utils.data as data
import dataset
from darknet53 import *

save_path = r'F:\yolo\new\param_combine'
save_param =r'para_net.pt'

def loss_function(out,label,a):
    out = out.permute(0,2,3,1)
    out = out.view(out.size(0),out.size(1),out.size(2),3,-1)
    mask_nonzero = label[...,0]>0
    label_nonzero = label[mask_nonzero].float()
    out_nonzero = out[mask_nonzero]
    # print(label_nonzero.size())
    # print(label_nonzero[0])
    # print(label_nonzero[:,1:3])
    loss_nonzero_conf = loss_fn1(out_nonzero[:,0],label_nonzero[:,0])
    loss_nonzero_coord = loss_fn1(out_nonzero[:,1:3], label_nonzero[:,1:3])
    loss_nonzero_coord2 = loss_fn2(out_nonzero[:,3:5], label_nonzero[:,3:5])

    label_nonzero =label_nonzero.long()
    # print(label_nonzero[:,5:])
    # print(label_nonzero[:,5:].argmax(dim = 1))

    loss_nonzero_classify = loss_fn3(out_nonzero[:,5:], label_nonzero[:,5:].argmax(dim =1))

    loss_nonzero = loss_nonzero_conf+loss_nonzero_coord+loss_nonzero_classify+loss_nonzero_coord2

    mask_zero = label[..., 0] == 0
    label_zero = label[mask_zero].float()
    out_zero = out[mask_zero].cpu()

    loss_zero_conf = loss_fn1(out_zero[:,0], label_zero[:,0])

    loss = a*loss_nonzero+(1-a)*loss_zero_conf

    return loss

if __name__ == '__main__':
    net = MainNet()
    if torch.cuda.is_available():
        net = net.cuda()
    mydata = dataset.MyDataset()
    label_data = data.DataLoader(mydata,batch_size=2,shuffle=True,num_workers=6)

    # net.load_state_dict(torch.load(os.path.join(save_path,save_param)))
    optimizer = torch.optim.Adam((net.parameters()))
    for epoch in range(20000):
        for label_13,label_26,label_52,img_data in label_data:
            # print('label_13',label_13[:,:,0,0])
            if torch.cuda.is_available():
                img_data = img_data.cuda()
            out_13,out_26,out_52 = net(img_data)
            # print('out',out_13[:,:,0,0])
            loss_fn1 = nn.BCEWithLogitsLoss()
            loss_fn2 = nn.MSELoss()
            loss_fn3 = nn.CrossEntropyLoss()
            out_13 = out_13.cpu()
            out_26 = out_26.cpu()
            out_52 = out_52.cpu()
            loss_13 = loss_function(out_13, label_13, 0.9)
            loss_26 = loss_function(out_26, label_26, 0.9)
            loss_52 = loss_function(out_52, label_52, 0.9)
            loss = loss_13+loss_26+loss_52

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('批次：', epoch, '总损失：', loss.item(),'loss_13:',loss_13.item(),'loss_26:',loss_26,'loss_52:',loss_52)
        if epoch % 100 == 0:
            if not os.path.exists(os.path.join(save_path, str(epoch))):
                os.makedirs(os.path.join(save_path, str(epoch)))
            torch.save(net.state_dict(), os.path.join(save_path, str(epoch), save_param))


