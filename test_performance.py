import torch
import pytest
import torch.nn as nn
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

from executorch.runtime import Runtime
from typing import List

@pytest.fixture(scope="module")
def executorch_model_path():
    model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    
    sample_inputs = (torch.randn(1, 3, 224, 224), )

    et_program = to_edge_transform_and_lower(
        torch.export.export(model, sample_inputs),
        partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    with open("model.pte", "wb") as f:
        f.write(et_program.buffer)
    
    return "model.pte"


def test_pytorch_and_executorch():
    # test executorch
    runtime = Runtime.get()

    input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
    program = runtime.load_program("model.pte")
    method = program.load_method("forward")
    output: List[torch.Tensor] = method.execute([input_tensor]) # output 1 (executorch)

    # test pytorch
    eager_reference_model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    eager_reference_output = eager_reference_model(input_tensor) # output 2 (pytorch)

    mae = nn.L1Loss()
    output = mae(output[0], eager_reference_output)
    mae_value = output.item()
    print("model_name: MobileNetV2")
    print("mean_absolute_difference:", mae_value)

    # assertions
    assert mae_value < 1e-4





    