# Most of the codes are from 
# https://github.com/kohya-ss/sd-webui-additional-networks
#
# I just added support to allow more masks for the LoRA models

import os
from PIL import Image

import torch
import numpy as np

import modules.scripts as scripts
from modules import shared, script_callbacks
import gradio as gr

from modules.ui_components import FormRow

from scripts import lora_masks_lora_compvis
from scripts.lora_masks_model_util import list_available_loras, available_lora_aliases, MAX_TAB_COUNT


inflora_paste_params = {"txt2img": [], "img2img": []}
num_tabs = int(MAX_TAB_COUNT)


class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.latest_params = [(None, None)] * 3 * 3 * num_tabs
        self.latest_networks = []
        self.latest_model_hash = ""

    def title(self):
        return "LoRA models Masks for generating"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        global inflora_paste_params
        # NOTE: Changing the contents of `ctrls` means the XY Grid support may need
        # to be updated, see xyz_grid_support.py
        ctrls = []
        weight_sliders = []

        tabname = "txt2img"
        if is_img2img:
            tabname = "img2img"

        paste_params = inflora_paste_params[tabname]
        paste_params.clear()

        self.infotext_fields = []
        self.paste_field_names = []

        with gr.Group():
            with gr.Accordion("Lora Masks", open=False):
                gr.HTML(value="<p style='font-size: 1.4em; margin-bottom: 0.7em'>Watch 📺 <b style='color: red'><a href=\"https://youtu.be/q-KGRRFARk4\">video</a></b> for detailed explanation 🔍 ☕️ Please consider supporting me in Patreon <b style='color: red'><a href=\"https://www.patreon.com/lifeisboringsoprogramming\">here</a></b> 🍻</p>")
                
                with gr.Row():
                    enabled = gr.Checkbox(label="Enable", value=False)
                    ctrls.append(enabled)
                    self.infotext_fields.append((enabled, "LoraMasks Enabled"))

                for t in range(num_tabs):
                    with gr.Tab(f"Model {t*9+1} to {t*9+9}"):
                        for i in range(3):
                            for j in range(3):
                                with FormRow(variant="compact"):
                                    model = gr.Textbox(
                                        label=f"Model {t*9+i*3+j+1} (Use channel {['R', 'G', 'B'][j]} in mask {t*3+i+1})", value="")

                                    weight = gr.Slider(
                                        label=f"Weight {t*9+i*3+j+1}", value=0.0, minimum=-1.0, maximum=2.0, step=0.05, visible=True)

                                    paste_params.append({"model": model})

                                ctrls.extend((model, weight))

                                self.infotext_fields.extend(
                                    [
                                        (model, f"LoraMasks Model {t*9+i*3+j+1}"),
                                        (weight, f"LoraMasks Weight {t*9+i*3+j+1}"),
                                        (weight, f"LoraMasks Weight {t*9+i*3+j+1}"),
                                    ]
                                )

                            # mask for regions
                            with gr.Accordion(f"Mask {t*3+i+1} (For model {t*9+i*3+1} to {t*9+i*3+3})", open=False):
                                with gr.Row():
                                    mask_image = gr.Image(label="mask image:")
                                    ctrls.append(mask_image)

                for _, field_name in self.infotext_fields:
                    self.paste_field_names.append(field_name)

        return ctrls

    def set_infotext_fields(self, p, params):
        for i, t in enumerate(params):
            model, weight = t
            if model is None or model == "None" or len(model) == 0 or (weight == 0):
                continue
            p.extra_generation_params.update(
                {
                    "LoraMasks Enabled": True,
                    f"LoraMasks Model {i+1}": model,
                    f"LoraMasks Weight {i+1}": weight,
                }
            )

    def restore_networks(self, sd_model):
        unet = sd_model.model.diffusion_model
        text_encoder = sd_model.cond_stage_model

        if len(self.latest_networks) > 0:
            print("restoring last networks")
            for network, _ in self.latest_networks[::-1]:
                network.restore(text_encoder, unet)
            self.latest_networks.clear()

    def process_batch(self, p, *args, **kwargs):
        unet = p.sd_model.model.diffusion_model
        text_encoder = p.sd_model.cond_stage_model

        if not args[0]:
            self.restore_networks(p.sd_model)
            return

        if len(args[1:]) != 21 * num_tabs:
            raise RuntimeError(f"params size not matched")
        
        masks = []
        params = []        
        for t in range(num_tabs):
            params.append([args[t*21+1], args[t*21+2]])
            params.append([args[t*21+3], args[t*21+4]])
            params.append([args[t*21+5], args[t*21+6]])
            masks.append(args[t*21+7])

            params.append([args[t*21+8], args[t*21+9]])
            params.append([args[t*21+10], args[t*21+11]])
            params.append([args[t*21+12], args[t*21+13]])
            masks.append(args[t*21+14])

            params.append([args[t*21+15], args[t*21+16]])
            params.append([args[t*21+17], args[t*21+18]])
            params.append([args[t*21+19], args[t*21+20]])
            masks.append(args[t*21+21])

        # no latest network (cleared by check-off)
        models_changed = len(self.latest_networks) == 0
        models_changed = models_changed or self.latest_model_hash != p.sd_model.sd_model_hash
        if not models_changed:
            for (l_model, l_weight), (model, weight) in zip(
                self.latest_params, params
            ):
                if l_model != model or l_weight != weight:
                    models_changed = True
                    break

        list_available_loras()

        if models_changed:
            self.restore_networks(p.sd_model)
            self.latest_params = params
            self.latest_model_hash = p.sd_model.sd_model_hash

            for model, weight in self.latest_params:
                if model is None or model == "None" or len(model) == 0:
                    continue
                if weight == 0:
                    print(f"ignore because weight is 0: {model}")
                    continue

                available_lora = available_lora_aliases.get(model, None)
                model_path = available_lora.filename if available_lora is not None else None
                if model_path is None:
                    raise RuntimeError(f"model not found: {model}")

                # trim '"' at start/end
                if model_path.startswith('"') and model_path.endswith('"'):
                    model_path = model_path[1:-1]
                if not os.path.exists(model_path):
                    print(f"file not found: {model_path}")
                    continue

                print(
                    f"LoRA weight: {weight}, model: {model}")
                if True:
                    if os.path.splitext(model_path)[1] == ".safetensors":
                        from safetensors.torch import load_file

                        du_state_dict = load_file(model_path)
                    else:
                        du_state_dict = torch.load(
                            model_path, map_location="cpu")

                    network, info = lora_masks_lora_compvis.create_network_and_apply_compvis(
                        du_state_dict, weight, weight, text_encoder, unet
                    )
                    # in medvram, device is different for u-net and sd_model, so use sd_model's
                    network.to(p.sd_model.device, dtype=p.sd_model.dtype)

                    print(f"LoRA model {model} loaded: {info}")
                    self.latest_networks.append((network, model))
            if len(self.latest_networks) > 0:
                print("setting (or sd model) changed. new networks created.")

        latest_networks_map = {}
        for network, model in self.latest_networks:
            latest_networks_map[model] = (network, model)

        # apply mask: currently only top 3 networks are supported
        if len(self.latest_networks) > 0:

            for m, mask_image in enumerate(masks):
                if mask_image is not None:
                    mask_image = mask_image.astype(np.float32) / 255.0
                    print(f"use mask {m+1} image to control LoRA regions.")
                    for i, (model, _) in enumerate(params[m*3:m*3+3]):
                        if model is not None and model != 'None' and model != '' and model in latest_networks_map:

                            network, model = latest_networks_map[model]
                            if not hasattr(network, "set_mask"):
                                continue
                            mask = mask_image[:, :, i]  # R,G,B

                            if mask.max() <= 0:
                                continue
                            mask = torch.tensor(
                                mask, dtype=p.sd_model.dtype, device=p.sd_model.device)

                            network.set_mask(mask, height=p.height, width=p.width,
                                            hr_height=p.hr_upscale_to_y, hr_width=p.hr_upscale_to_x)
                            print(f"apply mask {m+1}. channel: {['R', 'G', 'B'][i]}, model: {model}")
                else:
                    for i, (model, _) in enumerate(params[m*3:m*3+3]):
                        if model is not None and model != 'None' and model != '' and model in latest_networks_map:
                            network, model = latest_networks_map[model]
                            if hasattr(network, "set_mask"):
                                network.set_mask(None)

        self.set_infotext_fields(p, self.latest_params)


def on_script_unloaded():
    if shared.sd_model:
        for s in scripts.scripts_txt2img.alwayson_scripts:
            if isinstance(s, Script):
                s.restore_networks(shared.sd_model)
                break


def on_ui_settings():
    section = ("lora_masks", "LoRA masks")

    shared.opts.add_option(
        "lora_masks_number_of_tabs", shared.OptionInfo(
            4, "Max number of tabs to show", section=section)
    )


def on_infotext_pasted(infotext, params):
    if "LoraMasks Enabled" not in params:
        params["LoraMasks Enabled"] = "False"

    for i in range(3 * 3 * num_tabs):
        if f"LoraMasks Model {i+1}" not in params:
            params[f"LoraMasks Model {i+1}"] = "None"
        if f"LoraMasks Weight {i+1}" not in params:
            params[f"LoraMasks Weight {i+1}"] = "0"


script_callbacks.on_script_unloaded(on_script_unloaded)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_infotext_pasted(on_infotext_pasted)
