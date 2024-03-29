import os

try:
    import detectron2
    import densepose
except ImportError:
    os.system('pip install ./preprocess/detectron2')
    os.system('pip install ./preprocess/detectron2/projects/DensePose')

import sys
import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from utils_stableviton import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.detectron2.projects.DensePose.apply_net_gradio import DensePose4Gradio
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

os.environ['GRADIO_TEMP_DIR'] = './tmp'  # TODO: turn off when final upload


openpose_model_hd = OpenPose(0)
parsing_model_hd = Parsing(0)
densepose_model_hd = DensePose4Gradio(
    cfg='preprocess/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml',
    model='https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl',
)
stable_viton_model_hd = ...  # TODO: write down stable viton model

category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

# import spaces  # TODO: turn on when final upload

# @spaces.GPU  # TODO: turn on when final upload


def process_hd(vton_img, garm_img, n_samples, n_steps, guidance_scale, seed):
    model_type = 'hd'
    category = 0  # 0:upperbody; 1:lowerbody; 2:dress

    with torch.no_grad():
        openpose_model_hd.preprocessor.body_estimation.model.to('cuda')

        stt = time.time()
        print('load images... ', end='')
        garm_img = Image.open(garm_img).resize((768, 1024))
        vton_img = Image.open(vton_img).resize((768, 1024))
        print('%.2fs' % (time.time() - stt))

        stt = time.time()
        print('get agnostic map... ', end='')
        keypoints = openpose_model_hd(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_hd(vton_img.resize((384, 512)))
        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)  # agnostic map
        print('%.2fs' % (time.time() - stt))

        stt = time.time()
        print('get densepose... ', end='')
        vton_img = vton_img.resize((768, 1024))  # size for densepose
        densepose = densepose_model_hd.execute(vton_img)  # densepose
        print('%.2fs' % (time.time() - stt))

        # # stable viton here
        # images = stable_viton_model_hd(
        #     vton_img,
        #     garm_img,
        #     masked_vton_img,
        #     densepose,
        #     n_samples,
        #     n_steps,
        #     guidance_scale,
        #     seed
        # )

    # return images


example_path = os.path.join(os.path.dirname(__file__), 'examples')
model_hd = os.path.join(example_path, 'model/model_1.png')
garment_hd = os.path.join(example_path, 'garment/00055_00.jpg')

with gr.Blocks(css='style.css') as demo:
    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div>
                <h1>StableVITON Demo 👕👔👗</h1>
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                    <a href='https://arxiv.org/abs/2312.01725'>
                        <img src="https://img.shields.io/badge/arXiv-2312.01725-red">
                    </a>
                    &nbsp;
                    <a href='https://rlawjdghek.github.io/StableVITON/'>
                        <img src='https://img.shields.io/badge/page-github.io-blue.svg'>
                    </a>
                    &nbsp;
                    <a href='https://github.com/rlawjdghek/StableVITON'>
                        <img src='https://img.shields.io/github/stars/rlawjdghek/StableVITON'>
                    </a>
                    &nbsp;
                    <a href='https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode'>
                        <img src='https://img.shields.io/badge/license-CC_BY--NC--SA_4.0-lightgrey'>
                    </a>
                </div>
            </div>
        </div>
        """
    )
    with gr.Row():
        gr.Markdown("## Experience virtual try-on with your own images!")
    with gr.Row():
        with gr.Column():
            vton_img = gr.Image(label="Model", type="filepath", height=384, value=model_hd)
            example = gr.Examples(
                inputs=vton_img,
                examples_per_page=14,
                examples=[
                    os.path.join(example_path, 'model/model_1.png'),  # TODO more our models
                    os.path.join(example_path, 'model/model_2.png'),
                    os.path.join(example_path, 'model/model_3.png'),
                ])
        with gr.Column():
            garm_img = gr.Image(label="Garment", type="filepath", height=384, value=garment_hd)
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=14,
                examples=[
                    os.path.join(example_path, 'garment/00055_00.jpg'),
                    os.path.join(example_path, 'garment/00126_00.jpg'),
                    os.path.join(example_path, 'garment/00151_00.jpg'),
                ])
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True, scale=1)
    with gr.Column():
        run_button = gr.Button(value="Run")
        # TODO: change default values (important!)
        n_samples = gr.Slider(label="Images", minimum=1, maximum=4, value=1, step=1)
        n_steps = gr.Slider(label="Steps", minimum=20, maximum=40, value=20, step=1)
        guidance_scale = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)

    ips = [vton_img, garm_img, n_samples, n_steps, guidance_scale, seed]
    run_button.click(fn=process_hd, inputs=ips, outputs=[result_gallery])

demo.launch()
