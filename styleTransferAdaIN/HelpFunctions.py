import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import net
from function import adaptive_instance_normalization


# wyswietlanie obrazkow
def print_photos(phot_1, phot_2=None, nrow=8):
    if phot_2 is None:
        plt.imshow(np.transpose(vutils.make_grid(phot_1, normalize=True, nrow=nrow).cpu(), (1, 2, 0)))
    else:
        plt.imshow(np.transpose(vutils.make_grid([*phot_1, *phot_2], normalize=False, nrow=nrow).cpu(), (1, 2, 0)))
    plt.show()


# zapis sieci
def save_net(net, path):
    # torch.save(net, path)
    torch.save(net.state_dict(), path)


# wczytanie calej sieci lub samego state dict
def load_net(path_to_net_state, path_to_net=None, netN=None):
    assert (netN or path_to_net)
    if path_to_net is not None:
        net = torch.load(path_to_net, map_location='cpu')
        net.load_state_dict(torch.load(path_to_net_state, map_location='cpu'))
        return net
    else:
        netN.load_state_dict(torch.load(path_to_net_state, map_location='cpu'))
        return netN


# transform do tensora
def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


# test sieci na batchu obrazkow
def test_net(vgg_state_path, decoder_state_path, images_path, style_path,
             vgg=net.vgg, decoder=net.decoder, batch_size=16):
    decoder = load_net(decoder_state_path, path_to_net=None, netN=decoder)
    vgg = load_net(vgg_state_path, path_to_net=None, netN=vgg)
    vgg = nn.Sequential(*list(vgg.children())[:31])

    transformer = test_transform(256, True)

    dataset_real = dset.ImageFolder(root=images_path, transform=transformer)
    dataloader_real = torch.utils.data.DataLoader(dataset_real, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)

    it = iter(dataloader_real)
    o = next(it)[0]

    style = transformer(Image.open(str(style_path)))
    style = style.unsqueeze(0).repeat(o.size(0), 1, 1, 1)

    out = style_transfer(vgg, decoder, o, style)
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
def test_image(vgg_state_path, decoder_state_path, content_path, style_path, path_to_save,
               vgg=net.vgg, decoder=net.decoder, print_image=False):

    decoder = load_net(decoder_state_path, path_to_net=None, netN=decoder)
    vgg = load_net(vgg_state_path, path_to_net=None, netN=vgg)
    vgg = nn.Sequential(*list(vgg.children())[:31])

    transformer_content = test_transform(0, False)
    transformer_style = test_transform(256, True)

    style = transformer_style(Image.open(str(style_path)))
    style = style.unsqueeze(0)

    content = transformer_content(Image.open(str(content_path)))
    content = content.unsqueeze(0)

    out = style_transfer(vgg, decoder, content, style)

    if print_image:
        print_photos(out)

    vutils.save_image(out, path_to_save)
