import os
import tempfile
from pathlib import Path

import requests
import torch
import matplotlib as mpl
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.image.utils import TRANSFORM_RESIZE, TRANSFORM_CENTER_CROP_SIZE
from health_multimodal.text import get_cxr_bert_inference, TextInferenceEngine
from health_multimodal.image import get_biovil_resnet_inference, get_biovil_resnet, ImageInferenceEngine
from health_multimodal.text.utils import get_cxr_bert
from health_multimodal.vlp import ImageTextInferenceEngine
from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map

tokenizer, text_model = get_cxr_bert()
text_inference = TextInferenceEngine(tokenizer=tokenizer, text_model=text_model)

image_model = get_biovil_resnet()
transform = create_chest_xray_transform_for_inference(
    resize=TRANSFORM_RESIZE,
    center_crop_size=TRANSFORM_CENTER_CROP_SIZE,
)
image_inference = ImageInferenceEngine(image_model=image_model, transform=transform)

# todo evaluate with the ImageTextInferenceEngine
image_text_inference = ImageTextInferenceEngine(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # todo swap with right device
image_text_inference.to(device)


def plot_phrase_grounding(image_path: Path, text_prompt: str) -> None:
    similarity_map = image_text_inference.get_similarity_map_from_raw_data(
        image_path=image_path,
        query_text=text_prompt,
        interpolation="bilinear",
    )
    plot_phrase_grounding_similarity_map(
        image_path=image_path,
        similarity_map=similarity_map,
    )


def plot_phrase_grounding_from_url(image_url: str, text_prompt: str) -> None:
    image_path = Path("~/mcs_development/hi-ml/my_temp", "downloaded_chest_xray.jpg")
    response = requests.get(image_url)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(response.content)
    plot_phrase_grounding(image_path, text_prompt)


image_url = "https://openi.nlm.nih.gov/imgs/512/177/177/CXR177_IM-0503-1001.png"

text_prompt = "there is cardiomegaly"
plot_phrase_grounding_from_url(image_url, text_prompt)
