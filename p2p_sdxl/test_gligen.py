import torch
from diffusers import StableDiffusionGLIGENPipeline
from diffusers.utils import load_image

# # Insert objects described by text at the region defined by bounding boxes
# pipe = StableDiffusionGLIGENPipeline.from_pretrained(
#     "masterful/gligen-1-4-inpainting-text-box", variant="fp16", torch_dtype=torch.float16
# )
# pipe = pipe.to("cuda")

# input_image = load_image(
#     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/livingroom_modern.png"
# )
# prompt = "a birthday cake"
# boxes = [[0.2676, 0.6088, 0.4773, 0.7183]]
# phrases = ["a birthday cake"]

# images = pipe(
#     prompt=prompt,
#     gligen_phrases=phrases,
#     gligen_inpaint_image=input_image,
#     gligen_boxes=boxes,
#     gligen_scheduled_sampling_beta=1,
#     output_type="pil",
#     num_inference_steps=50,
# ).images

# images[0].save("./gligen-1-4-inpainting-text-box.jpg")

# Generate an image described by the prompt and
# insert objects described by text at the region defined by bounding boxes
pipe = StableDiffusionGLIGENPipeline.from_pretrained(
    "masterful/gligen-1-4-generation-text-box", variant="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "a sun is running across a tree"
boxes = [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]
phrases = ["a tree", "a sun"]

images = pipe(
    prompt=prompt,
    gligen_phrases=phrases,
    gligen_boxes=boxes,
    gligen_scheduled_sampling_beta=1,
    output_type="pil",
    num_inference_steps=50,
).images

images[0].save("./gligen-1-4-generation-text-box.jpg")
from PIL import Image, ImageDraw 
# 打开图片 

# 画图 
draw = ImageDraw.Draw(images[0]) 
H,W=images[0].height,images[0].width
draw.rectangle((boxes[0][0]*W, boxes[0][1]*W, boxes[0][2]*H, boxes[0][3]*H), outline='red', width=3)
draw.rectangle((boxes[1][0]*W, boxes[1][1]*W, boxes[1][2]*H, boxes[1][3]*H), outline='blue', width=3)

images[0].save("./gligen-1-4-generation-text-box_with_box.jpg")