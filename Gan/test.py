from HelpFunctions import *
from Generator import *

path_to_net_state = "models/netG83_state"
path_to_real_images = "M:/coco2017/"
path_to_image = "M:/imgs/building.jpg"
path_to_save_image = "out/building_cartoon.jpg"

netG = Generator()

test_net(path_to_net_state, path_to_real_images, netN=netG, batch_size=16)

test_image(path_to_net_state, netG, path_to_image, path_to_save_image, print_image=True)
