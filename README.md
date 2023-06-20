<img src="images/02.png" />

# Stable Diffusion LoRA masks extension
A custom extension for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that allow applying mask to LoRA models. The core of the feature and most UI are from [kohya-ss/sd-webui-additional-networks](https://github.com/kohya-ss/sd-webui-additional-networks)

# Overview
* This project allows users to add mask area for the LoRA models
* It support any number of LoRA models and masks
* For each group of three LoRA models 1, 2, 3, user can specify a RGB image to mask the area of effect for those LoRA models
* Channel R corrsponding to the LoRA model 1 in the group, channel G for model 2, channel B for model 3
* the weight of each LoRA model is the product of the LoRA weight and the value in the corrsponding RGB channel
* For example, a value 128 in the channel R and weight value 1.0 in the weight slider equals (128 / 255) * 1.0 ~= 0.5

# Tutorial
There is a video to show how to use the extension

[![Stable diffusion tutorial - How to use Two or Three LoRA models in one image without in-paint](https://img.youtube.com/vi/jh-TrplWVA0/sddefault.jpg)](https://www.youtube.com/watch?v=jh-TrplWVA0)

# Stable Diffusion extension
This project can be run as a stable Diffusion extension inside the Stable Diffusion WebUI.

## Installation for stable Diffusion extension
* Copy and paste `https://github.com/lifeisboringsoprogramming/sd-webui-lora-masks.git` to URL for extension's git repository
* Press Install button
* Apply and restart UI when finished installing

<img src="images/webui-install.png" />

# Settings
To set the number of tabs
* Go to settings tab
<img src="images/webui-settings.png" />


# Screenshotsmasks
* with controlnet extension
* 8 different LoRA models
* 4 masks (each mask for 2 LoRA models)
* with Latent Couple extension

<img src="images/01.png" />
<img src="images/02.png" />
<img src="images/03.png" />
<img src="images/04.png" />
<img src="images/05.png" />
<img src="images/06.png" />

# YouTube Channel
Please subscribe to my YouTube channel, thank you very much. 

[https://bit.ly/3odzTKX](https://bit.ly/3odzTKX)

# Patreon
☕️ Please consider to support me in Patreon 🍻

[https://www.patreon.com/lifeisboringsoprogramming](https://www.patreon.com/lifeisboringsoprogramming)