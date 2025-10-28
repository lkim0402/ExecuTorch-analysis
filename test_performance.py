import torch
import pytest
import os
import torch.nn as nn
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

from executorch.runtime import Runtime
from typing import List
import time

@pytest.fixture(scope="module")
def executorch_model_path():
    model_path = "model.pte"
    model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    
    sample_inputs = (torch.randn(1, 3, 224, 224), )

    et_program = to_edge_transform_and_lower(
        torch.export.export(model, sample_inputs),
        partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    with open(model_path, "wb") as f:
        f.write(et_program.buffer)
    
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

def measure_latency(fn_to_execute):

    warmup = 5
    rounds = 20
    latencies = []

    # warmup rounds
    for _ in range(warmup):
        fn_to_execute()

    # total rounds
    for _ in range(rounds):
        start = time.perf_counter()
        fn_to_execute()
        end = time.perf_counter()
        latencies.append((end - start) * 1000) # ms
    
    avg_lat = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    return avg_lat, max_lat
        






    