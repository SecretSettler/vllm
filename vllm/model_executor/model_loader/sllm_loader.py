import torch
from torch import nn
import collections
import os
import gc
from typing import Dict, Optional
from vllm.config import LoadConfig, VllmConfig, ModelConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model, set_default_torch_dtype)

class ServerlessLLMLoader(BaseModelLoader):
    # DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = ({} if load_config.model_loader_extra_config is None
                        else load_config.model_loader_extra_config.copy())
        # self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        if extra_config:
            raise ValueError(f"Unexpected extra config keys for load format "
                             f"{load_config.load_format}: "
                             f"{load_config.model_loader_extra_config.keys()}")

    @staticmethod
    def _filter_subtensors(
            tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        same_storage_groups = collections.defaultdict(list)
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result = {}
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 > k:
                        # Same tensors, keep the one with the longer key.
                        break
                else:
                    result[k] = t
        return result
        
    def load_model(self, *, vllm_config: VllmConfig) -> nn.Module:
        from sllm_store.torch import load_dict
        from vllm.distributed import get_tensor_model_parallel_rank
        
        assert os.path.isdir(vllm_config.model_config.model)
        
        rank = get_tensor_model_parallel_rank()

        local_model_path = vllm_config.model_config.model
        local_model_path = os.path.join(local_model_path, f"rank_{rank}")

        def remove_prefix(path, prefix):
            # Normalize the paths to ensure consistency across different platforms
            path = os.path.normpath(path)
            prefix = os.path.normpath(prefix)
            
            # Check if the path starts with the prefix
            if path.startswith(prefix):
                # Return the path without the prefix
                return path[len(prefix):].lstrip(os.sep)
            
            # Return the original path if the prefix doesn't exist
            return path
        
        # vLLM needs a local model path to read model config but
        # ServerlessLLM Store requires a global model path as the model ID
        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = remove_prefix(local_model_path, storage_path)
        
        with set_default_torch_dtype(vllm_config.model_config.dtype):
            # with torch.device(device_config.device):
            with torch.device("cpu"):
                model = initialize_model(vllm_config=vllm_config)
                model = model.eval()
            # set all parameters to meta device
            state_dict = self._filter_subtensors(model.state_dict())
            key_list = list(state_dict.keys())
            
            for key, param in model.named_parameters(recurse=True):
                if key in key_list:
                    param.data = torch.empty(1, device="cuda")
            gc.collect()
            
            device_id = torch.cuda.current_device()
            device_map = {"": device_id}
            # Note: storage path is already included in the local model path
            sllm_state_dict = load_dict(model_path, device_map)
            
            for key, param in model.named_parameters(recurse=True):
                if key in key_list:
                    tensor = sllm_state_dict[key]
                    param.data = tensor
                    state_dict.pop(key)
            if state_dict:
                raise ValueError(
                    f"Missing keys {tuple(state_dict)} in loaded state!")
            
        return model
    
    def download_model(self, model_config: ModelConfig) -> None:
        pass

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank
        from sllm_store.torch import save_dict
        
        rank = get_tensor_model_parallel_rank()
        state_dict = ServerlessLLMLoader._filter_subtensors(model.state_dict())
        
        # move all tensors to CPU
        for key, tensor in state_dict.items():
            state_dict[key] = tensor.cpu().contiguous()

        save_path = os.path.join(path, f"rank_{rank}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        save_dict(state_dict, save_path)