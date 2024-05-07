import argparse
from HelpFunctions import *
from Generator import *

parser = argparse.ArgumentParser()

parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_batch', type=str,
                    help='File path to the directory with images')
parser.add_argument('--generator', type=str, default='models/netG83_state')
parser.add_argument('--output', type=str, default='out/image.jpg',
                    help='File path to save image')

args = parser.parse_args()

netG = Generator()

assert(args.content or args.content_batch)
if args.content_batch:
    test_net(args.generator, args.content_batch, netN=netG, batch_size=8)
if args.content:
    test_image(args.generator, netG, args.content, args.output, print_image=True)
