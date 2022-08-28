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
import numpy as np
import threading
from threading import Lock
from queue import Empty, Queue

class ThreadSafeCounter():
    # constructor
    def __init__(self):
        # initialize counter
        self._counter = 0
        # initialize lock
        self._lock = Lock()
 
    # increment the counter
    def increment(self, n):
        with self._lock:
            self._counter += n
 
    # get the counter value
    def value(self):
        with self._lock:
            return self._counter

def upscale_image(image, params = {}, on_image_update = None):
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
                    "n_timestep": 100,
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
    return diff2x(opt, image, logger, on_image_update)

def tile_to_tensor(tile: Image):
    min_max = (-1, 1) 
    tensor = TF.to_tensor(tile) * (min_max[1] - min_max[0]) + min_max[0]
    return tensor

def tensor_to_tile(tensor) -> Image:
    min_max = (-1, 1) 
    tensor = tensor.squeeze(0).float().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    return TF.to_pil_image(tensor)

def make_transparency_mask(size, overlap_pixels, remove_borders = []):
    size_x = size[0] - overlap_pixels * 2
    size_y = size[1] - overlap_pixels * 2
    for letter in ['l', 'r']:
        if letter in remove_borders:
            size_x += overlap_pixels
    for letter in ['t', 'b']:
        if letter in remove_borders:
            size_y += overlap_pixels
    mask = np.ones((size_y, size_x), dtype=np.uint8) * 255
    mask = np.pad(mask, mode = 'linear_ramp', pad_width=overlap_pixels, end_values = 0)

    if 'l' in remove_borders:
        mask = mask[:, overlap_pixels:mask.shape[1]]
    if 'r' in remove_borders:
        mask = mask[:, 0:mask.shape[1] - overlap_pixels]
    for letter in ['t', 'b']:
        if letter in remove_borders:
            mask = mask[overlap_pixels:mask.shape[0], :]
    return mask

def add_overlap_rect(rect: [int], overlap: int, image_size: [int]):
    rect = list(rect)
    if rect[0] > 0:
        rect[0] -= overlap
    if rect[1] > 0:
        rect[1] -= overlap
    if rect[2] < image_size[0]:
        rect[2] += overlap
    if rect[3] < image_size[1]:
        rect[3] += overlap
    return rect

def process_tile(x, y, tile_size, tile_border, img, final_image, diffusion):
    crop_rect = (x * tile_size, y * tile_size, min(img.size[0], (x + 1) * tile_size), min(img.size[1], (y + 1) * tile_size))
    crop_rect_with_overlap = add_overlap_rect(crop_rect, tile_border, img.size)
    tile = img.crop(crop_rect_with_overlap)
    tile = tile.resize((tile.size[0] * 2, tile.size[1] * 2), Image.BICUBIC)
    tensor = tile_to_tensor(tile).to(diffusion.device)
    tensor_up = diffusion.netG.super_resolution(tensor.unsqueeze(0), name = f"Tile {x + 1}x{y + 1}", continous = False)
    upscaled_tile = tensor_to_tile(tensor_up)
    remove_borders = []
    if x == 0:
        remove_borders.append("l")
    elif crop_rect[2] == img.size[0]:
        remove_borders.append("r")
    if y == 0:
        remove_borders.append("t")
    elif crop_rect[3] == img.size[1]:
        remove_borders.append("b")
    transparency_mask = Image.fromarray(make_transparency_mask((tile.size[0], tile.size[1]), tile_border, remove_borders=remove_borders), mode="L")    
    final_image.paste(upscaled_tile, (crop_rect_with_overlap[0] * 2, crop_rect_with_overlap[1] * 2), transparency_mask)
    return upscaled_tile

class TileWorker:
    def __init__(self, queue: Queue, on_image_update):
        self._running = False
        self._thread = None
        self._queue = queue
        self._on_image_update = on_image_update

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
        while self._running:
            try:
                item = self._queue.get()
            except Empty:
                self.terminate()
            else:
                process_tile(*item)
                self._on_image_update(item[5])
                self._queue.task_done()

class Diff2XResult:
    def __init__(self, queue, final_image, tiles_done_counter: ThreadSafeCounter, total_tiles_to_make: int):
        self._queue = queue
        self._total_tiles_to_make = total_tiles_to_make
        self._tiles_done_counter = tiles_done_counter
        self.final_image = final_image

    def progress_pct(self):
        return self._tiles_done_counter.value() / self._total_tiles_to_make

    def wait(self):
        self._queue.join()

def diff2x(opt, input_image, logger, on_image_update):
    with torch.no_grad():
        logger.info('Loading Model')
        diffusion = Model.create_model(opt)
        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['val'], schedule_phase='val')
        logger.info(f'Opening image "{input_image}"...')
        img = Image.open(input_image).convert('RGB')
        final_image = Image.new('RGB', (img.size[0] * 2, img.size[1] * 2))
        tile_border = 8
        tile_size = 32
        tcx = img.size[0] // tile_size
        tcy = img.size[1] // tile_size
        logger.info('Begin Model Inference.')
        tile_batch = 5
        logging.info(f"Setting up {tile_batch} workers...")
        queue = Queue()
        workers = []
        tiles_done_counter = ThreadSafeCounter()
        for i in range(tile_batch):
            def _on_image_update(image):
                tiles_done_counter.increment(1)
                if on_image_update is not None:
                    on_image_update(image)
            worker = TileWorker(queue, _on_image_update)
            workers.append(worker)
        logging.info(f"Making {tcy * tcx} tiles in {(tcy * tcx) // tile_batch} batches...")
        for y in range(tcy):
            row = []
            for x in range(tcx):
                item = (x, y, tile_size, tile_border, img, final_image, diffusion)
                queue.put(item)
        for worker in workers:
            worker.start()
        return Diff2XResult(queue, final_image, tiles_done_counter, tcx * tcy)
if __name__ == "__main__":
    main()