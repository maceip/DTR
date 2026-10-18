"""One pretrained vision factory, exported unchanged through either backend."""


def build():
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    torch.manual_seed(42)
    model = mobilenet_v3_small(weights=weights).eval()
    return {
        "task": "image.classification",
        "signatures": {"default": {"model": model, "inputs": {"image": torch.randn(1, 3, 224, 224)}, "outputs": ["logits"]}},
        "image": {"input_name": "image", "height": 224, "width": 224, "resize_shorter": 256,
                  "layout": "nchw", "scale": 1 / 255, "mean": [.485, .456, .406], "std": [.229, .224, .225]},
        "labels": weights.meta["categories"], "output_name": "logits",
        "metadata": {"model_source": "torchvision/MobileNet_V3_Small_Weights.IMAGENET1K_V1",
                     "purpose": "ImageNet whole-image classification; not object detection"},
    }
