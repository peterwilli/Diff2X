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

def tile_to_tensor(tile: Image):
    min_max = (-1, 1) 
    tensor = TF.to_tensor(tile) * (min_max[1] - min_max[0]) + min_max[0]
    return tensor

def tensor_to_tile(tensor) -> Image:
    min_max = (-1, 1) 
    tensor = tensor.squeeze(0).float().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    return TF.to_pil_image(tensor)

def process_tile(x, y, tile_size, img, final_image, diffusion):
    crop_rect = (x * tile_size, y * tile_size, min(img.size[0], (x + 1) * tile_size), min(img.size[1], (y + 1) * tile_size))
    tile = img.crop(crop_rect).resize((tile_size * 2, tile_size * 2), Image.BICUBIC)
    tensor = tile_to_tensor(tile).to(diffusion.device)
    tensor_up = diffusion.netG.super_resolution(tensor.unsqueeze(0), False)
    upscaled_tile = tensor_to_tile(tensor_up)
    final_image.paste(upscaled_tile, (crop_rect[0] * 2, crop_rect[1] * 2))
    
def process_tiles(positions, tile_size, img, final_image, diffusion):
    result = None
    for pos in positions:
        x = pos[0]
        y = pos[1]
        crop_rect = (x * tile_size, y * tile_size, min(img.size[0], (x + 1) * tile_size), min(img.size[1], (y + 1) * tile_size))
        tile = img.crop(crop_rect).resize((tile_size * 2, tile_size * 2), Image.BICUBIC)
        tensor = tile_to_tensor(tile)
        if result is None:
            result = tensor.unsqueeze(0)
        else:
            result = torch.cat((result, tensor.unsqueeze(0)), dim=0)
    result_up = diffusion.netG.super_resolution(result.to(diffusion.device), False)
    for i, pos in enumerate(positions):
        x = pos[0]
        y = pos[1]
        upscaled_tile = tensor_to_tile(result_up[i, ...])
        final_image.paste(upscaled_tile, (x * tile_size * 2, y * tile_size * 2))

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
            img = Image.open(input_image).convert('RGB')
            tile_size = 32
            tcx = img.size[0] // tile_size
            tcy = img.size[1] // tile_size
            logger.info('Begin Model Inference.')
            final_image = Image.new('RGB', (img.size[0] * 2, img.size[1] * 2))
            tile_input = []
            for y in range(tcy):
                row = []
                for x in range(tcx):
                    tile_input.append((x, y))
            tile_batch = 20
            logging.info(f"Making {len(tile_input)} tiles in {len(tile_input) // tile_batch} batches...")
            for tc in range(0, len(tile_input), tile_batch):
                logging.info(f"Batch {tc // tile_batch}")
                tiles_to_process = tile_input[tc:tc + min(tile_batch, len(tile_input))]
                process_tiles(tiles_to_process, tile_size, img, final_image, diffusion)
        except KeyboardInterrupt:
            print('KeyboardInterrupt exception is caught')
            final_image.save("partial_image.png")
            sys.exit(0)
        final_image.save("final_image.png")
        

if __name__ == "__main__":
    main()