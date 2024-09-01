import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Union
from torch.nn.parameter import Parameter
import itertools

# Base class for a layer that will merge weights from multiple models
class DAMBaseLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_models=3,
        init_merger_values=[],
        dtype=None,
        non_linearity: str = 'tanh',  # Option to apply non-linearity
    ):
        super().__init__()

        # Store the number of models being merged
        self.num_models = num_models

        # Store the non-linearity to be applied on the merging coefficients
        self.non_linearity = non_linearity

        # If no initial values are provided, set equal initial merger values for each model
        if init_merger_values == []:
            init_merger_values = [1/num_models] * num_models

        # Initialize the list of weights for each model's layer
        self.weights = nn.ParameterList([Parameter(
            torch.zeros(out_features, in_features, dtype=dtype) * init_merger_values[i]
        ) for i in range(num_models)])

        # Initialize the list of merging coefficients for each model's layer
        self.mergers = nn.ParameterList([Parameter(
            torch.ones(in_features, dtype=dtype) * init_merger_values[i]
        ) for i in range(num_models)])

    # Method to compute the similarity between merging coefficients
    def compute_mergers_similarity(self, lambda_coef=None):
        if lambda_coef is None:
            return 0.0
    
        # Initialize similarity loss
        similarity_loss = 0.0
        
        # Create all possible pairs of merging coefficients
        combinations = list(itertools.combinations([p for p in self.mergers], 2))
        num_combinations = len(combinations)
    
        if num_combinations > 0:
            similarities = []
            for merger_a, merger_b in combinations:
                # Calculate cosine similarity between each pair of merging coefficients
                similarity = F.cosine_similarity(merger_a, merger_b, dim=0)
                similarities.append(similarity)
                
            # Average the similarities and multiply by the provided coefficient
            similarity_loss = torch.mean(torch.stack(similarities))
            similarity_loss *= lambda_coef
    
        return similarity_loss

    # Method to compute L1 and L2 regularization on the merging coefficients
    def compute_mergers_L1_L2_reg(self, lambda_coef_l1=None, lambda_coef_l2=None):
        device = self.mergers[0].device
        l1_reg = torch.tensor(0.0, device=device)
        l2_reg = torch.tensor(0.0, device=device)

        # Calculate L1 norm for each merging coefficient in the ParameterList and sum them
        if lambda_coef_l1 is not None:
            l1_reg += sum(merger.norm(1).to(device) for merger in self.mergers) * lambda_coef_l1

        # Calculate L2 norm for each merging coefficient in the ParameterList and sum them
        if lambda_coef_l2 is not None:
            l2_reg += sum(merger.norm(2).to(device) for merger in self.mergers) * lambda_coef_l2

        # Return the combined L1 and L2 regularization loss
        return l1_reg + l2_reg


    # Method to unfreeze the merging coefficients so they can be trained
    def unfreeze(self):
        for merger in self.mergers:
            merger.requires_grad = True

# Specialized class for a linear layer that uses the DAMBaseLayer
class DAMLinearLayer(DAMBaseLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_models=3,
        bias: bool=False,
        init_merger_values=[],
        dtype=None,
        non_linearity: str = 'tanh',  # Option to apply non-linearity
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            num_models=num_models,
            init_merger_values=init_merger_values,
            dtype=dtype,
            non_linearity=non_linearity,
        )

        # If no initial values are provided, set equal initial merger values for each model
        if init_merger_values == []:
            init_merger_values = [1/num_models] * num_models

        # If the layer has a bias, initialize the list of biases and bias mergers for each model
        if bias:
            self.biases = nn.ParameterList([nn.Parameter(torch.zeros(out_features, dtype=dtype) * init_merger_values[i]) for i in range(num_models)])
            self.bias_mergers = nn.ParameterList([nn.Parameter(torch.ones(1, dtype=dtype) * init_merger_values[i]) for i in range(num_models)])

    # Method to apply the specified non-linearity to the merging coefficients
    def apply_non_linearity(self, tensor):
        if self.non_linearity == 'tanh':
            return torch.tanh(tensor)
        elif self.non_linearity == 'sigmoid':
            return torch.sigmoid(tensor)
        elif self.non_linearity == 'relu':
            return torch.relu(tensor)
        else:
            return tensor  # If non_linearity is None or unsupported, return the tensor as is

    # Method to compute the combined weight for the merged layer
    def get_dam_weight(self):
        device = self.mergers[0].device
        constrained_mergers = [self.apply_non_linearity(merger) for merger in self.mergers] if self.non_linearity else self.mergers
        # Sum the weighted contributions of each model's weight using the (possibly constrained) merging coefficients
        return sum(merger.to(device) * weight.to(device) for merger, weight in zip(constrained_mergers, self.weights))
    
    # Method to compute the combined bias for the merged layer (if bias is used)
    def get_dam_bias(self):
        if hasattr(self, 'biases'):
            device = self.bias_mergers[0].device
            constrained_bias_mergers = [self.apply_non_linearity(merger) for merger in self.bias_mergers] if self.non_linearity else self.bias_mergers
            # Sum the weighted contributions of each model's bias using the (possibly constrained) bias merging coefficients
            return sum(merger.to(device) * bias.to(device) for merger, bias in zip(constrained_bias_mergers, self.biases))
        return None

    # Forward pass through the DAMLinearLayer
    def forward(self, hidden_states: torch.Tensor, model_index: Optional[int] = None) -> Union[torch.Tensor, list]:
        if model_index is not None:
            # Return the output from the specified model without merging
            weight = self.weights[model_index].to(hidden_states.device)
            bias = self.biases[model_index].to(hidden_states.device) if hasattr(self, 'biases') else None
            return F.linear(hidden_states, weight=weight, bias=bias)
        else:
            # Ensure the weights are on the same device as the input tensor
            weight = self.get_dam_weight().to(hidden_states.device)
            # Ensure the bias (if any) is on the same device as the input tensor
            bias = self.get_dam_bias().to(hidden_states.device) if self.get_dam_bias() is not None else None

            # Perform the linear transformation using the merged weight and bias
            return F.linear(hidden_states, weight=weight, bias=bias)