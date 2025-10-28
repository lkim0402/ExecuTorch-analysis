import torch
import os
import json
import argparse
from utils.measure_latency import measure_latency
import torch.nn as nn
from utils.model_compiler import compile_executorch_model_for_path
from executorch.runtime import Runtime

model_dict = {
    'resnet18.pte': 'resnet18',
    'mobilenetv2.pte': 'mobilenetv2'
}

def main():
    parser = argparse.ArgumentParser(description='This is a new command-line tool')
    # run_bench --model resnet18.pte --repeat 5
    parser.add_argument(
        '--model_path', 
        choices= ["resnet18.pte", "mobilenetv2.pte"],
        help='path of the model to use',
        type=str
    )
    parser.add_argument(
        '--repeat', 
        help='number of running inference',
        type=int
    )
    args = parser.parse_args()
    model_path = args.model_path
    repeat = args.repeat

    compile_executorch_model_for_path(model_path)
    executorch_get_benchmark(model_path, repeat)

def executorch_get_benchmark(model_path, repeat):
    model_name = model_dict[model_path]

    runtime = Runtime.get()
    input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
    program = runtime.load_program(model_path)
    method = program.load_method("forward")

    def inference_executorch():
        method.execute([input_tensor])
    
    avg, _ = measure_latency(inference_executorch, repeat)

    results_dict = {
        "model_name" : model_name,
        "latency_ms_avg" : round(avg, 2),
        "repeat" : repeat
    }

    print(json.dumps(results_dict, indent=4))

    # teardown
    os.remove(model_path)

if __name__ == "__main__":
    main()







    