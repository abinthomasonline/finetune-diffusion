import os
import shutil
import safetensors
import torch

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

def maybe_checkpoint(global_step, checkpoint_freq, checkpoint_dir, accelerator, text_encoder, placeholder_token_id, placeholder_token, keep_n_checkpoints):
    if global_step % checkpoint_freq == 0:
        print(f"Saving checkpoint at step {global_step}...")
        weight_name = f"learned_embeds-steps-{global_step}.safetensors"
        save_path = os.path.join(checkpoint_dir, weight_name)
        learned_embeds = accelerator.unwrap_model(
                text_encoder).get_input_embeddings().weight[placeholder_token_id : (placeholder_token_id + 1)]
        learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
        
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})

        checkpoints = os.listdir(checkpoint_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        if len(checkpoints) >= keep_n_checkpoints:
            num_to_remove = len(checkpoints) - keep_n_checkpoints + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(checkpoint_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)


def maybe_evaluate(global_step, validation_freq, model_id, accelerator, text_encoder, tokenizer, unet, vae, seed, validation_prompts, num_validation_images):
    if global_step % validation_freq == 0:
        print(f"Running validation at step {global_step}...")
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            safety_checker=None,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference
        generator = torch.Generator(device=accelerator.device).manual_seed(seed)
        images = []
        for validation_prompt in validation_prompts:
            for _ in range(num_validation_images):
                with torch.autocast("cuda"):
                    image = pipeline(validation_prompt, num_inference_steps=25, generator=generator).images[0]
                images.append(image)

        del pipeline
        torch.cuda.empty_cache()
        return images
    