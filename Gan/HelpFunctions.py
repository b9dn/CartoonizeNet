import torch
import torch.utils.data
from torchvision import transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# wyswietlanie obrazkow
def print_photos(phot_1, phot_2=None, nrow=8):
    if phot_2 is None:
        plt.imshow(np.transpose(vutils.make_grid(phot_1, normalize=True, nrow=nrow).cpu(), (1, 2, 0)))
    else:
        plt.imshow(np.transpose(vutils.make_grid([*phot_1,*phot_2], normalize=False, nrow=nrow).cpu(), (1, 2, 0)))
    plt.show()


# zapis sieci
def save_net(net, path):
    #torch.save(net, path)
    torch.save(net.state_dict(), path)


# wczytanie calej sieci lub samego state dict
def load_net(path_to_net_state, path_to_net=None, netN=None):
    assert(netN or path_to_net)
    if path_to_net is not None:
        net = torch.load(path_to_net, map_location='cpu')
        net.load_state_dict(torch.load(path_to_net_state, map_location='cpu'))
        return net
    else:
        netN.load_state_dict(torch.load(path_to_net_state, map_location='cpu'))
        return netN


# test sieci na batchu obrazkow
def test_net(path_to_net_state, path_to_real_images, path_to_net=None, netN=None, batch_size=16):
    netG = load_net(path_to_net_state, path_to_net=path_to_net, netN=netN)

    transformer = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])

    dataset_real = dset.ImageFolder(root=path_to_real_images, transform=transformer)
    dataloader_real = torch.utils.data.DataLoader(dataset_real, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)

    it = iter(dataloader_real)
    o = next(it)[0]
    out = netG(o)
    print_photos(o, out)


# pobranie batcha obrazkow
def get_batch_images(path_to_images, batch_size=16):
    transformer = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    dataset = dset.ImageFolder(root=path_to_images, transform=transformer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)

    return next(iter(dataloader))[0]


# przerobienie jednego obrazka i zapis
def test_image(path_to_net_state, net, image_path, path_to_save, print_image=False):
    net = load_net(path_to_net_state, netN=net)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image = transform(Image.open(str(image_path))).unsqueeze(0)
    out = net(image)
    if print_image:
        print_photos(out)

    vutils.save_image(out, path_to_save)
