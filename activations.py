import torch
from jaxtyping import Int, Float
import sys
from transformers import BertTokenizer, BertModel

sys.path.append('../')
from models import get_model_from_name
from data import get_wikitext_103_sample


def get_layers_to_enumerate(model):
    model_name = model.config._name_or_path
    if 'gpt' in model_name:
        return model.transformer.h
    if 'pythia' in model_name:
        return model.gpt_neox.layers # à vérifier
    if 'bert' in model_name:
        return model.encoder.layer
    else:
        raise ValueError(f"Unsupported model: {model_name}.")


def get_residual_stream_activations(model,
                                    tokenizer):
    """ Returns a torch.Tensor of activations. 

    activations.shape = torch.Size([batch, seq, num_hidden_layers, hidden_size])
    """

    #model, tokenizer = models.get_model_from_name(model_name)
    wikitext_sample = get_wikitext_103_sample()
    input_ids = tokenizer.encode(wikitext_sample, return_tensors='pt', truncation=True)

    activations = torch.zeros((input_ids.shape[0], input_ids.shape[1], model.config.num_hidden_layers, model.config.hidden_size)) # batch, seq, n_layers, d_model

    def get_activation(layer_id):
        def hook(model, input, output):
            activations[:, :, layer_id, :] = output[0].detach()
        return hook

    layers_to_enum = get_layers_to_enumerate(model)
    for i, layer in enumerate(layers_to_enum):
        layer.register_forward_hook(get_activation(i))

    _ = model(input_ids)

    return activations, wikitext_sample


def get_residual_stream_activations_for_layer_wise_dmd(model_name: str = 'gpt2-small',
                                                       models = None,
                                                       model_types = None):
    """ Returns
    DMD takes as input a Tensor of shape [trials, timepoints, dimension]
    or [timepoints, dimension].
    """

    model, tokenizer = get_model_from_name(model_name)
    x, wikitext_sample = get_residual_stream_activations(model, tokenizer)

    if models == None:
        models = []
    if model_types == None:
        model_types = []
    
    for layer in range(0, model.config.num_hidden_layers):
        x_layer = x[:, :, layer, :]
        models.append(x_layer)
        model_types.append(f'{model_name} - layer {layer}')

    return models, model_types

