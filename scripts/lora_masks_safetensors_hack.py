# Most of the codes are from 
# https://github.com/kohya-ss/sd-webui-additional-networks
#
# I just added support to allow more masks for the LoRA models

import os
import mmap
import torch
import json
import safetensors.torch


# PyTorch 1.13 and later have _UntypedStorage renamed to UntypedStorage
UntypedStorage = torch.storage.UntypedStorage if hasattr(
    torch.storage, 'UntypedStorage') else torch.storage._UntypedStorage


def load_file(filename, device):
    """"Loads a .safetensors file without memory mapping that locks the model file.
    Works around safetensors issue: https://github.com/huggingface/safetensors/issues/164"""
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")
            metadata_bytes = m.read(n)
            metadata = json.loads(metadata_bytes)

    size = os.stat(filename).st_size
    storage = UntypedStorage.from_file(filename, False, size)
    offset = n + 8
    md = metadata.get("__metadata__", {})
    return {name: create_tensor(storage, info, offset) for name, info in metadata.items() if name != "__metadata__"}, md


DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    # "U64": torch.uint64,
    "I32": torch.int32,
    # "U32": torch.uint32,
    "I16": torch.int16,
    # "U16": torch.uint16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool
}


def create_tensor(storage, info, offset):
    """Creates a tensor without holding on to an open handle to the parent model
    file."""
    dtype = DTYPES[info["dtype"]]
    shape = info["shape"]
    start, stop = info["data_offsets"]
    return torch.asarray(storage[start + offset: stop + offset], dtype=torch.uint8).view(dtype=dtype).reshape(shape).clone().detach()
