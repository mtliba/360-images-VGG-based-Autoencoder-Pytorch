import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from data_loader import Autoencoder_dataset
from model import Autoencoder
import os

root ='path to your image dataset'
def img_denorm(img, mean, std):
    #for ImageNet the mean and std are:
    #mean = np.asarray([ 0.485, 0.456, 0.406 ])
    #std = np.asarray([ 0.229, 0.224, 0.225 ])

    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))
    res = denormalize(res)

    #Image needs to be clipped since the denormalize function will map some
    #values below 0 and above 1
    res = torch.clamp(res, 0, 1)
    res = res.view(res.size(0), 3, 576, 288)
    
    return(res)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# setting hyperparameters
batch_size = 128,
num_epochs = 150,
learning_rate = 1e-4
if not os.path.exists('./decoded_images'):
     os.mkdir('./decoded_images')
def main():
    

    trainset = Autoencoder_dataset(True ,root,transforms=transforms.Compose([
        transforms.Rescale(576,288),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225]))
    ]))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    valset = Autoencoder_dataset(False ,root,transforms=transforms.Compose([
        transforms.Rescale(576,288),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225]))
    ]))
    val_loader = DataLoader(valset, batch_size=batch_size)


    model = Autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)

    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch)
        for data in train_loader:
            img, _ = data
            img = (img).cuda()
            output = model(img)
            loss = criterion(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, loss.data[0]))
        with torch.no_grad():
            output_val = model(input)
            loss_val = criterion(output_val, target)
            print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, loss_val.data[0]))
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, filename=os.path.join('./', 'checkpoint_{}.tar'.format(epoch)))


        if epoch % 25 == 0:
            pic = img_denorm(output.cpu().data)
            save_image(pic, './decoded_images/image_{}.png'.format(epoch))


if __name__ == '__main__':

    main()
