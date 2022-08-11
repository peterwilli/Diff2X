import asyncio
import os
from typing import Dict
from .diff2x import upscale_image
from jina import Executor, requests, DocumentArray

def Diff2Executor(Executor):
    @requests(on='/upscale')
    async def upscale_image(self, docs: DocumentArray, parameters: Dict, **kwargs):
        pass # TODO