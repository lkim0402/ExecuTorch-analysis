import torch
import pytest
import os
from utils.measure_latency import measure_latency
import torch.nn as nn
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.runtime import Runtime
from typing import List
from utils.model_compiler import compile_executorch_model_for_path


@pytest.fixture(scope="module")
def executorch_model_path():
    model_path = compile_executorch_model_for_path()
    assert model_path == "model.pte"
    yield model_path

    # teardown
    os.remove(model_path)

def test_pytorch_and_executorch(executorch_model_path):
    # test executorch
    runtime = Runtime.get()

    input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
    program = runtime.load_program(executorch_model_path)
    method = program.load_method("forward")
    output: List[torch.Tensor] = method.execute([input_tensor]) # output 1 (executorch)

    # test pytorch
    eager_reference_model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    eager_reference_output = eager_reference_model(input_tensor) # output 2 (pytorch)

    mae = nn.L1Loss()
    output = mae(output[0], eager_reference_output)
    mae_value = output.item()

    def inference_pytorch():
        eager_reference_model(input_tensor)

    def inference_executorch():
        method.execute([input_tensor])
    
    eager_avg, eager_max = measure_latency(inference_pytorch)
    avg, max = measure_latency(inference_executorch)

    print("model_name: mobilenet_v2")
    print("mean_absolute_difference:", mae_value)
    print(f"pytorch_latency: avg {eager_avg:.2f} ms, max {eager_max:.2f} ms")
    print(f"executorch_latency: avg {avg:.2f} ms, max {max:.2f} ms")

    # assertions
    assert mae_value < 1e-4






    