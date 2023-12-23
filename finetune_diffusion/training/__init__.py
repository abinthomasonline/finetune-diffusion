import os

from diffusers.optimization import get_scheduler
import torch
import torch.nn.functional as F

from .utils import maybe_checkpoint, maybe_evaluate


def train(vae, unet, text_encoder, accelerator, checkpoint_dir, noise_scheduler, tokenizer, placeholder_token_id, checkpoint_freq, placeholder_token, keep_n_checkpoints, 
          validation_freq, model_id, seed, validation_prompts, num_validation_images, dataset_loader, max_train_steps):
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    trainable_params = text_encoder.get_input_embeddings().parameters()
    optimizer = torch.optim.AdamW(trainable_params)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer)

    text_encoder.train()
    text_encoder, optimizer, dataset_loader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, dataset_loader, lr_scheduler)
    
    unet.to(accelerator.device)
    vae.to(accelerator.device)

    # Get the most recent checkpoint
    dirs = os.listdir(checkpoint_dir)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None

    global_step = 0
    if path is not None:
        accelerator.load_state(checkpoint_dir)
        global_step = int(path.split("-")[1])

    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    text_encoder.train()
    while True:
        for _, batch in enumerate(dataset_loader):
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
            latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            index_no_updates[placeholder_token_id : (placeholder_token_id + 1)] = False

            with torch.no_grad():
                accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]

            global_step += 1

            print(f"Step: {global_step} - Loss: {loss.item()}")

            maybe_checkpoint(global_step, checkpoint_freq, checkpoint_dir, accelerator, text_encoder, placeholder_token_id, placeholder_token, keep_n_checkpoints)

            maybe_evaluate(global_step, validation_freq, model_id, accelerator, text_encoder, tokenizer, unet, vae, seed, validation_prompts, num_validation_images)
            
            if global_step >= max_train_steps:
                break
        
        if global_step >= max_train_steps:
            break
