import torch
from utils.measure_latency import measure_latency
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from typing import List

def compile_executorch_model_for_path():
    model_path = "model.pte"
    model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    
    sample_inputs = (torch.randn(1, 3, 224, 224), )

    et_program = to_edge_transform_and_lower(
        torch.export.export(model, sample_inputs),
        partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    with open(model_path, "wb") as f:
        f.write(et_program.buffer)
    
    return model_path