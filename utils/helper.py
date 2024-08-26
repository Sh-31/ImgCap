import torch

def load_checkpoint(checkpoint_path, model, remove_prefix=True ,device='cpu', optimizer=None):
    """
    Loads a model and optimizer state from a checkpoint file, removing '_orig_mod.' prefix if present.

    Parameters:
    - checkpoint_path (str): Path to the checkpoint file.
    - model (torch.nn.Module): The model instance to load the state dict into.
    - optimizer (torch.optim.Optimizer, optional): The optimizer instance to load the state dict into.

    Returns:
    - model (torch.nn.Module): The model with loaded state dict.
    - optimizer (torch.optim.Optimizer, optional): The optimizer with loaded state dict (if provided).
    - epoch (int): The epoch number saved in the checkpoint.
    - train_loss (float): The training loss saved in the checkpoint.
    - val_loss (float): The validation loss saved in the checkpoint (if available).
    - bleu_score (float): The BLEU score saved in the checkpoint (if available).
    - cider_score (float): The CIDEr score saved in the checkpoint (if available).
    """

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))


    # Modify the state_dict to remove the `_orig_mod.` prefix, if it exists
    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        new_key = key.replace('_orig_mod.', '')  # Remove the prefix if present
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', None)
    train_loss = checkpoint.get('train_loss', None)
    val_loss = checkpoint.get('val_loss', None)
    bleu_score = checkpoint.get('bleu_score', None)
    cider_score = checkpoint.get('cider_score', None)

    return model, optimizer, epoch, train_loss, val_loss, bleu_score, cider_score
