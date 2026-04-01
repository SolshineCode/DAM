import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

# Qwen 3.5 and Nemotron-H norm classes (optional, requires recent transformers)
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
except ImportError:
    Qwen3_5RMSNorm = None

try:
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHRMSNorm
except ImportError:
    NemotronHRMSNorm = None

# Collect all supported RMSNorm classes
_NORM_CLASSES = [LlamaRMSNorm, MistralRMSNorm]
if Qwen3_5RMSNorm is not None:
    _NORM_CLASSES.append(Qwen3_5RMSNorm)
if NemotronHRMSNorm is not None:
    _NORM_CLASSES.append(NemotronHRMSNorm)
_NORM_CLASSES = tuple(_NORM_CLASSES)


def find_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]

def find_embedding_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, torch.nn.Embedding)]

def find_norm_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, _NORM_CLASSES)]