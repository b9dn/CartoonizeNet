import argparse
from HelpFunctions import *

parser = argparse.ArgumentParser()

parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_batch', type=str,
                    help='File path to the directory with images')
parser.add_argument('--style', type=str, default="input/style/comic.jpg",
                    help='File path to the style image')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/net43_state')
parser.add_argument('--output', type=str, default='out/image.jpg',
                    help='File path to save image')

args = parser.parse_args()

assert(args.content or args.content_batch)
if args.content_batch:
    test_net(args.vgg, args.decoder, args.content_batch, args.style, batch_size=8)
if args.content:
    test_image(args.vgg, args.decoder, args.content, args.style, args.output, print_image=False)
