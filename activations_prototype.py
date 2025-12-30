import torch
from nnsight import LanguageModel
from tqdm import tqdm_notebook as tqdm

from counting_data import CountingSequence, generate_uniform_count_sequences, format_chat_prompt
from models import AVAILABLE_MODELS

torch.set_grad_enabled(False)

model = LanguageModel("google/gemma-3-12b-it", device_map="auto")
tokenizer = model.tokenizer

def extract_activations_for_sequence(
    model: LanguageModel,
    tokenizer,
    sequence: CountingSequence,
    layers: list[int] = None,
) -> dict:
    """
    Extract residual stream activations for a single sequence.
    
    Returns dict with:
        - 'activations': tensor of shape (num_layers, seq_len, hidden_dim)
        - 'token_positions': list mapping sequence idx -> token position
        - 'counts': list of counts at each sequence position
    """

    n_layers = len(model.model.language_model.layers) 
    target_layers = list(range(n_layers)) if layers is None else layers

    
    prompt = format_chat_prompt(sequence, tokenizer)
    tokens = tokenizer(prompt, return_tensors="pt").input_ids

    layer_activations = []

    with model.trace(tokens) as tracer:                
        for layer_idx in target_layers:
            hidden_states = model.model.language_model.layers[layer_idx].output[0]
            layer_activations.append(hidden_states.save())
    
    # Stack activations: (num_layers, seq_len, hidden_dim)
    activations = torch.stack([act.squeeze(0) for act in layer_activations])
    
    return activations

examples = generate_uniform_count_sequences(
    min_count=1,
    max_count=100,
    num_sequences=5,
    density_range=(0.05, 0.8)
)

for example in tqdm(examples):
    activations = extract_activations_for_sequence(model, tokenizer, example)
