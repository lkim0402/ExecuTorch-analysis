import pytest
import json
import sys
from unittest.mock import MagicMock, ANY
import run_bench 

@pytest.fixture
def mock_dependencies(mocker):
    mock_compile = mocker.patch('run_bench.compile_executorch_model_for_path')
    mock_measure = mocker.patch('run_bench.measure_latency', return_value=(12.34, None))
    
    # mock ExecuTorch Runtime
    mock_runtime = mocker.patch('run_bench.Runtime')
    mock_method = MagicMock()
    mock_program = MagicMock()
    mock_program.load_method.return_value = mock_method
    mock_runtime.get.return_value.load_program.return_value = mock_program
    
    # mock torch.randn to avoid torch dependency
    mocker.patch('run_bench.torch.randn', return_value="dummy_tensor")
    
    # mock os.remove to check if cleanup happens
    mock_remove = mocker.patch('run_bench.os.remove')
    mocker.patch('run_bench.os.path.exists', return_value=True) 

    # return dict of mocks
    return {
        "compile": mock_compile,
        "measure": mock_measure,
        "runtime": mock_runtime,
        "method": mock_method,
        "remove": mock_remove
    }

# capsys - captures print() statements
def test_executorch_get_benchmark(mock_dependencies, capsys):
    model_path = "resnet18.pte"
    repeat_count = 5
    
    run_bench.executorch_get_benchmark(model_path, repeat_count)
    
    # runtime check
    mock_dependencies["runtime"].get.assert_called_once()
    mock_dependencies["runtime"].get().load_program.assert_called_with(model_path)
    mock_dependencies["runtime"].get().load_program().load_method.assert_called_with("forward")
    
    # measure_latency check
    mock_dependencies["measure"].assert_called_with(ANY, repeat_count)
    
    # printed JSON output check
    captured = capsys.readouterr()
    output_json = json.loads(captured.out)
    
    expected_json = {
        "model_name": "resnet18",
        "latency_ms_avg": 12.34, # mock value
        "repeat": 5
    }
    assert output_json == expected_json
    
    # file removal check
    mock_dependencies["remove"].assert_called_with(model_path)

def test_main_function(mocker, monkeypatch):
    # mock functions called by main
    mock_compile = mocker.patch('run_bench.compile_executorch_model_for_path')
    mock_benchmark = mocker.patch('run_bench.executorch_get_benchmark')
    
    # simulate CLI args (using monkeypatch)
    test_args = ['run_bench.py', '--model_path', 'mobilenetv2.pte', '--repeat', '10']
    monkeypatch.setattr(sys, 'argv', test_args)
    
    # run main function
    run_bench.main()
    
    # assert that mocked functions were called with the correct args
    mock_compile.assert_called_once_with('mobilenetv2.pte')
    mock_benchmark.assert_called_once_with('mobilenetv2.pte', 10)
