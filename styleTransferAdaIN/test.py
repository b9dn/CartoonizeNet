from HelpFunctions import *
import net

vgg_state_path = "models/vgg_normalised.pth"
decoder_state_path = "models/net43_state"
images_path = "M:/coco2017/"
style_path = "input/style/comic.jpg"

path_to_image = "M:/imgs/building.jpg"
path_to_save_image = "out/building_cartoon.jpg"

vgg = net.vgg
decoder = net.decoder

test_net(vgg_state_path, decoder_state_path, images_path, style_path, batch_size=8)

test_image(vgg_state_path, decoder_state_path, path_to_image, style_path, path_to_save_image)
