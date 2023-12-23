"""Entry point for fine-tuning"""
import argparse


from accelerate import Accelerator
from accelerate.utils import set_seed
import torch


from finetune_diffusion.datasets import load_dataset
from finetune_diffusion.models import load_pipeline_components
from finetune_diffusion.training import train


def main(model_id, batch_size, seed, data_dir, initializer_token, placeholder_token, checkpoint_dir, checkpoint_freq, 
         validation_freq, validation_prompts, num_validation_images, keep_n_checkpoints, max_train_steps):

    accelerator = Accelerator()
    set_seed(seed)

    # load pipeline components
    tokenizer, noise_scheduler, text_encoder, vae, unet = load_pipeline_components(model_id)

    # maybe update text_encoder
    assert len(tokenizer.encode(initializer_token, add_special_tokens=False)) == 1, "Initializer token must be a single token."

    new_token_count = tokenizer.add_tokens([placeholder_token])
    assert new_token_count == 1, "Placeholder token must be a new token."

    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    initializer_token_id = tokenizer.convert_tokens_to_ids(initializer_token)
    with torch.no_grad():
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id].clone()

    # prepare dataset
    dataset = load_dataset(data_dir, tokenizer)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # train
    train(vae, unet, text_encoder, accelerator, checkpoint_dir, noise_scheduler, tokenizer, placeholder_token_id, checkpoint_freq, placeholder_token, keep_n_checkpoints, 
          validation_freq, model_id, seed, validation_prompts, num_validation_images, dataset_loader, max_train_steps)
    
    accelerator.end_training()


if __name__ == "__main__":
    model_id = 'runwayml/stable-diffusion-v1-5'
    batch_size = 1
    seed = 42
    data_dir = ''
    initializer_token = 'painting'
    placeholder_token = '<stained-glass>'
    checkpoint_dir = ''
    checkpoint_freq = 100
    validation_freq = 100
    validation_prompts = ['A <stained-glass> portrait of a puppy']
    num_validation_images = 4
    keep_n_checkpoints = 5
    max_train_steps = 1000

    main(model_id, batch_size, seed, data_dir, initializer_token, placeholder_token, checkpoint_dir, checkpoint_freq, 
         validation_freq, validation_prompts, num_validation_images, keep_n_checkpoints, max_train_steps)
