import model.networks as networks
import argparse
from PIL import Image
import core.logger as Logger
import logging
import torchvision.transforms.functional as TF
import torch
import model as Model
import core.metrics as Metrics
import os
from multiprocessing import Pool
import traceback
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/diff2x.json',
                        help='JSON file for configuration')

    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')  
    parser.add_argument('-i', '--image', type=str,
                        default="misc/diff2x_demo_image.png",
                        help='Image file to upscale')
                        
    logger = logging.getLogger('base')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    logger.info(Logger.dict2str(opt))   
    diff2x(opt, args.image, logger)

def process_tile(x, y, tile_size, img, final_image, diffusion):
    print(f"Doing {x}x{y}")
    crop_rect = (x * tile_size, y * tile_size, min(img.size[0], (x + 1) * tile_size), min(img.size[1], (y + 1) * tile_size))
    tile = img.crop(crop_rect).resize((tile_size * 2, tile_size * 2))
    tile = TF.to_tensor(tile).to(diffusion.device)
    result = diffusion.netG.super_resolution(tile.unsqueeze(0), True)
    upscaled_tile = TF.to_pil_image(result[0].squeeze(0))
    final_image.paste(upscaled_tile, (crop_rect[0] * 2, crop_rect[1] * 2))

def diff2x(opt, input_image, logger):
    with torch.no_grad():
        try:
            result_path = '{}'.format(opt['path']['results'])
            os.makedirs(result_path, exist_ok=True)
            logger.info('Loading Model')
            diffusion = Model.create_model(opt)
            diffusion.set_new_noise_schedule(
                opt['model']['beta_schedule']['val'], schedule_phase='val')

            logger.info(f'Opening image "{input_image}"...')
            img = Image.open(input_image)
            tile_size = 32
            tcx = img.size[0] // tile_size
            tcy = img.size[1] // tile_size
            logger.info('Begin Model Inference.')
            final_image = Image.new('RGB', (img.size[0] * 2, img.size[1] * 2))
            # pool = Pool(1)
            for y in range(tcy):
                row = []
                for x in range(tcx):
                    process_tile(x, y, tile_size, img, final_image, diffusion)
                    # pool.apply_async(process_tile, (x, y, tile_size, img, final_image, diffusion)) 
                    # break
                # break
            # pool.close()
            # pool.join()
        except KeyboardInterrupt:
            print('KeyboardInterrupt exception is caught')
            final_image.save("partial_image.png")
            sys.exit(0)
        final_image.save("partial_image.png")
        

if __name__ == "__main__":
    main()