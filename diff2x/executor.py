import asyncio
import os
from typing import Dict
from .diff2x import upscale_image
from jina import Executor, requests, DocumentArray, Document
import json
import fasteners
import threading
import throttle

class Diff2Executor(Executor):
    @requests(on='/upscale')
    async def exec_upscale_image(self, docs: DocumentArray, parameters: Dict, **kwargs):
        doc = docs[0]
        path_tmp_image = os.path.join("/tmp", f"upscale_{doc.id}.png")
        path_tmp_json = os.path.join("/tmp", f"upscale_{doc.id}.json")
        path_tmp_lock = os.path.join("/tmp", f"{doc.id}.lock")

        @throttle.wrap(0.5, 1)
        def on_image_update(image):
            with fasteners.InterProcessLock(path_tmp_lock):
                img_out = open(path_tmp_image, 'wb')
                image.save(img_out)
                img_out.flush()
                img_out.close()
                with open(path_tmp_json, "w") as f:
                    f.write(json.dumps({
                        'progress': result.progress_pct()
                    }))
                    
        result = upscale_image(doc.uri, on_image_update = on_image_update)
        result.wait()
        result_doc = Document()
        result_doc.load_pil_image_to_datauri(result.final_image)
        return DocumentArray([result_doc])

class ProgressExecutor(Executor):
    @requests(on='/upscale_progress')
    async def exec_upscale_image(self, docs: DocumentArray, parameters: Dict, **kwargs):
        doc = docs[0]
        path_tmp_image = os.path.join("/tmp", f"upscale_{doc.id}.png")
        path_tmp_json = os.path.join("/tmp", f"upscale_{doc.id}.json")
        path_tmp_lock = os.path.join("/tmp", f"{doc.id}.lock")
        if os.path.exists(path_tmp_image) and os.path.exists(path_tmp_json):
            lock = fasteners.InterProcessLock(path_tmp_lock)
            lock.acquire()
            with open(path_tmp_json, 'r') as f:
                image_json = json.loads(f.read())
            doc.uri = path_tmp_image
            doc.tags = dict(
                progress = image_json['progress']
            )
            doc = doc.convert_uri_to_datauri(base64 = True, charset = 'utf-8')
            lock.release()
            return DocumentArray([doc])
        else:
            return DocumentArray([])