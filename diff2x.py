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
import traceback
import sys
import threading
from queue import Empty, Queue

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
                        default="misc/test_image_by_peterwilli.png",
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
    tensor_up = diffusion.netG.super_resolution(tensor.unsqueeze(0), name = f"Tile {x + 1}x{y + 1}", continous = False)
    upscaled_tile = tensor_to_tile(tensor_up)
    final_image.paste(upscaled_tile, (crop_rect[0] * 2, crop_rect[1] * 2))

def tile_worker(queue):
    while True:
        try:
            item = queue.get()
        except Empty:
            logger.info('Loading Model')
            break
        else:
            process_tile(*item)
            queue.task_done()

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
            tile_batch = 10
            logging.info(f"Setting up {tile_batch} workers...")
            queue = Queue()
            workers = []
            for i in range(tile_batch):
                worker = threading.Thread(target=tile_worker, args=(queue,))
                worker.daemon = True
                worker.start()
                workers.append(worker)
            logging.info(f"Making {tcy * tcx} tiles in {(tcy * tcx) // tile_batch} batches...")
            for y in range(tcy):
                row = []
                for x in range(tcx):
                    item = (x, y, tile_size, img, final_image, diffusion)
                    queue.put(item)
            for worker in workers:
                worker.join()
        except KeyboardInterrupt as e:
            print('KeyboardInterrupt exception is caught')
            final_image.save("partial_image.png")
            raise e
        final_image.save("final_image.png")
        

if __name__ == "__main__":
    main()