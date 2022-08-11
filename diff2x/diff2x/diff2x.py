from .model import networks
from . import model as Model
from .core import metrics as Metrics
from .core import logger as Logger
import argparse
from PIL import Image
import logging
import torchvision.transforms.functional as TF
import torch
import os
import traceback
import sys
import threading
from queue import Empty, Queue

def upscale_image(image, params = {}):
    opt = {
        "name": "diffusion2x",
        "phase": "train",
        "gpu_ids": [0],
        "path": {
            "resume_state": "./I130000_E13",
        },
        "datasets": {
            "train": {
                "name": "PeterDataset",
                "mode": "HR",
                "dataroot": "/content/dataset_train_32_64",
                "datatype": "img",
                "l_resolution": 32,
                "r_resolution": 64,
                "batch_size": 32,
                "num_workers": 16,
                "use_shuffle": True,
                "data_len": -1,
            },
            "val": {
                "name": "PeterDatasetVal",
                "mode": "LRHR",
                "dataroot": "dataset/dataset_val_32_64",
                "datatype": "img",
                "l_resolution": 32,
                "r_resolution": 64,
                "data_len": 3,
            },
        },
        "model": {
            "which_model_G": "sr3",
            "finetune_norm": False,
            "unet": {
                "in_channel": 6,
                "out_channel": 3,
                "inner_channel": 128,
                "norm_groups": 16,
                "channel_multiplier": [1, 2, 2],
                "attn_res": [16],
                "res_blocks": 4,
                "dropout": 0.2,
            },
            "beta_schedule": {
                "train": {
                    "schedule": "linear",
                    "n_timestep": 1000,
                    "linear_start": 1e-06,
                    "linear_end": 0.01,
                },
                "val": {
                    "schedule": "linear",
                    "n_timestep": 10,
                    "linear_start": 1e-06,
                    "linear_end": 0.01,
                },
            },
            "diffusion": {"image_size": 64, "channels": 3, "conditional": True},
        },
        "train": {
            "n_iter": 100000,
            "val_freq": 1000,
            "save_checkpoint_freq": 1000,
            "print_freq": 50,
            "optimizer": {"type": "adam", "lr": 1e-05},
            "ema_scheduler": {
                "step_start_ema": 5000,
                "update_ema_every": 1,
                "ema_decay": 0.9999,
            },
        },
        "wandb": {"project": "diff2x"},
        "distributed": False,
        "log_wandb_ckpt": False,
        "log_eval": False,
        "enable_wandb": False,
    }
    opt.update(params)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))   
    diff2x(opt, image, logger)

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
    return upscaled_tile

class TileWorker:
    def __init__(self, tiles, queue: Queue, amount_to_make: int):
        self._running = False
        self._thread = None
        self._queue = queue
        self._tiles = tiles
        self._amount_to_make = amount_to_make

    def terminate(self):
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def join(self):
        self._thread.join()
        
    def _run(self):
        while self._running and len(self._tiles) < self._amount_to_make:
            try:
                item = self._queue.get()
            except Empty:
                continue
            else:
                tile = process_tile(*item)
                self._tiles[f"{item[0]}x{item[1]}"] = tile
                self._queue.task_done()

def diff2x(opt, input_image, logger):
    with torch.no_grad():
        final_image = None
        try:
            logger.info('Loading Model')
            diffusion = Model.create_model(opt)
            diffusion.set_new_noise_schedule(
                opt['model']['beta_schedule']['val'], schedule_phase='val')
            logger.info(f'Opening image "{input_image}"...')
            img = Image.open(input_image).convert('RGB')
            final_image = Image.new('RGB', (img.size[0] * 2, img.size[1] * 2))
            tile_size = 32
            tcx = img.size[0] // tile_size
            tcy = img.size[1] // tile_size
            logger.info('Begin Model Inference.')
            tile_batch = 10
            logging.info(f"Setting up {tile_batch} workers...")
            queue = Queue()
            workers = []
            tiles_made = {}
            for i in range(tile_batch):
                worker = TileWorker(tiles_made, queue, tcy * tcx)
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
            if final_image is not None:
                final_image.save("partial_image.png")
            for worker in workers:
                worker.terminate()
            raise e
        final_image.save("final_image.png")

if __name__ == "__main__":
    main()