# Most of the codes are from
# https://github.com/kohya-ss/sd-webui-additional-networks
#
# I just added support to allow more masks for the LoRA models

import os
import os.path

from modules import shared, sd_models, errors


MAX_TAB_COUNT = shared.opts.data.get("lora_masks_number_of_tabs", 4)
LORA_MODEL_EXTS = [".pt", ".ckpt", ".safetensors"]
available_loras = {}
available_lora_aliases = {}
forbidden_lora_aliases = {}


def walk_files(path, allowed_extensions=None):
    if not os.path.exists(path):
        return

    if allowed_extensions is not None:
        allowed_extensions = set(allowed_extensions)

    for root, dirs, files in os.walk(path):
        for filename in files:
            if allowed_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext not in allowed_extensions:
                    continue

            yield os.path.join(root, filename)


def is_safetensors(filename):
    return os.path.splitext(filename)[1] == ".safetensors"


def list_available_loras():

    metadata_tags_order = {"ss_sd_model_name": 1, "ss_resolution": 2,
                           "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}

    class LoraOnDisk:
        def __init__(self, name, filename):
            self.name = name
            self.filename = filename
            self.metadata = {}

            _, ext = os.path.splitext(filename)
            if ext.lower() == ".safetensors":
                try:
                    self.metadata = sd_models.read_metadata_from_safetensors(
                        filename)
                except Exception as e:
                    errors.display(e, f"reading lora {filename}")

            if self.metadata:
                m = {}
                for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                    m[k] = v

                self.metadata = m

            # those are cover images and they are too big to display in UI as text
            self.ssmd_cover_images = self.metadata.pop(
                'ssmd_cover_images', None)
            self.alias = self.metadata.get('ss_output_name', self.name)

    available_loras.clear()
    available_lora_aliases.clear()
    forbidden_lora_aliases.clear()
    forbidden_lora_aliases.update({"none": 1})

    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    candidates = list(walk_files(shared.cmd_opts.lora_dir,
                      allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    for filename in sorted(candidates, key=str.lower):
        if os.path.isdir(filename):
            continue

        name = os.path.splitext(os.path.basename(filename))[0]
        entry = LoraOnDisk(name, filename)

        available_loras[name] = entry

        if entry.alias in available_lora_aliases:
            forbidden_lora_aliases[entry.alias.lower()] = 1

        available_lora_aliases[name] = entry
        available_lora_aliases[entry.alias] = entry
