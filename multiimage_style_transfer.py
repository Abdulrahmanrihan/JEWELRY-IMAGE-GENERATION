import os
import random
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPProcessor, CLIPModel
# tqdm and matplotlib imports are present but not strictly necessary for the core generation logic used by streamlit app
# from tqdm.auto import tqdm
# import matplotlib.pyplot as plt
from glob import glob

class MultiImageStyleTransfer:
    def __init__(self, model_dir):
        """
        Initialize multi-image style transfer pipeline using local models.

        Args:
            model_dir (str): Directory containing the downloaded models.
                             Expected subdirectories: 'clip-vit-base-patch32' and 'stable-diffusion-2-1'.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        clip_model_path = os.path.join(model_dir, "clip-vit-base-patch32")
        sd_model_path = os.path.join(model_dir, "stable-diffusion-2-1")

        # Check if model directories exist
        if not os.path.isdir(clip_model_path):
            raise FileNotFoundError(f"CLIP model directory not found at: {clip_model_path}. Please ensure it exists and contains the model files.")
        if not os.path.isdir(sd_model_path):
            raise FileNotFoundError(f"Stable Diffusion model directory not found at: {sd_model_path}. Please ensure it exists and contains the model files.")

        print(f"Loading CLIP model from: {clip_model_path}")
        # Load CLIP from local path
        self.clip_model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
        # Add use_fast=True to address the warning
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path, use_fast=False)
        print("CLIP model loaded.")
        
        print(f"Loading Stable Diffusion model from: {sd_model_path}")
        # Determine torch dtype based on device
        torch_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32

        # Load Stable Diffusion Pipeline from local path
        # Remove revision='fp16' when loading locally, it's for hub downloads
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            sd_model_path,
            torch_dtype=torch_dtype,
        ).to(self.device)
        print("Stable Diffusion model loaded.")
        # Enable memory-efficient attention if CUDA is available (optional, can help with memory)
        # if self.device.type == 'cuda':
        #    try:
        #        self.pipe.enable_xformers_memory_efficient_attention()
        #        print("Enabled xformers memory efficient attention.")
        #    except ImportError:
        #        print("xformers not installed. Memory efficient attention not enabled.")

    def extract_image_features(self, images):
        """
        Extract CLIP features from multiple images

        Args:
            images (List[PIL.Image]): List of input images (already preprocessed)

        Returns:
            torch.Tensor: Aggregated image features
        """
        if not images:
             return torch.zeros((1, self.clip_model.config.projection_dim)).to(self.device) # Return zero tensor if no images

        # Preprocess images using the CLIP processor
        # Note: Ensure images are PIL objects here
        try:
             processed_inputs = self.clip_processor(
                 images=images,
                 return_tensors="pt",
                 padding=True
             )
             pixel_values = processed_inputs.pixel_values.to(self.device)

             # Handle potential dtype mismatch if not using CUDA/float16
             if self.device.type != 'cuda':
                 pixel_values = pixel_values.to(torch.float32)
             else:
                 pixel_values = pixel_values.to(torch.float16)


        except Exception as e:
            print(f"Error during CLIP processing: {e}")
            raise

        # Compute image features
        with torch.no_grad():
            try:
                image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
            except Exception as e:
                print(f"Error getting image features from CLIP model: {e}")
                raise


        # Aggregate features (simple mean average)
        # The shape is likely (num_images, feature_dim), so mean over dim 0
        if image_features.shape[0] > 1:
            aggregated_features = torch.mean(image_features, dim=0, keepdim=True) # Keep batch dim for consistency
        else:
            aggregated_features = image_features

        return F.normalize(aggregated_features, p=2, dim=-1)

    # This simple prompt generation might be less effective than desired.
    # Consider replacing with a fixed, good prompt for jewelry or enhancing this.
    def generate_enhanced_style_prompt(self, image_features, num_input_images):
        """
        Generate a descriptive prompt (simplified version).

        Args:
            image_features (torch.Tensor): Aggregated image features (currently unused in this simple version)
            num_input_images (int): Number of input images used.

        Returns:
            str: Descriptive style prompt
        """
        style_keywords = [
            "luxurious", "intricate", "elegant",
            "detailed", "professional", "high-end",
            "artisanal", "handcrafted", "premium",
            "exquisite", "refined", "sophisticated",
            "modern", "vintage", "minimalist", "baroque",
            "geometric", "organic", "textured"
        ]

        # Select a few random keywords
        num_keywords = min(3 + num_input_images // 2, len(style_keywords)) # Slightly more keywords if more inputs
        selected_styles = random.sample(style_keywords, num_keywords)

        base_prompt = f"A professional product photograph of a {' '.join(selected_styles)} jewelry piece"
        if num_input_images > 1:
             base_prompt += f" inspired by {num_input_images} different designs"

        # Add quality/setting modifiers
        full_prompt = f"{base_prompt}, studio lighting, high resolution, sharp focus, detailed craftsmanship"
        # print(f"Generated Prompt: {full_prompt}") # For debugging
        return full_prompt

    def preprocess_images(self, image_paths, target_size=(768, 768)):
        """
        Load, resize, and pad images.

        Args:
            image_paths (List[str]): Paths to input images.
            target_size (Tuple[int, int]): Target size for the model (should match SD model expectations).

        Returns:
            List[PIL.Image]: List of preprocessed PIL images.
        """
        preprocessed_images = []
        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")

                # Resize while maintaining aspect ratio (thumbnail)
                image.thumbnail(target_size, Image.Resampling.LANCZOS) # Use LANCZOS for better quality downscaling

                # Create a new image with white background and paste the resized image in the center
                new_image = Image.new("RGB", target_size, (255, 255, 255))
                paste_x = (target_size[0] - image.width) // 2
                paste_y = (target_size[1] - image.height) // 2
                new_image.paste(image, (paste_x, paste_y))

                preprocessed_images.append(new_image)
            except FileNotFoundError:
                print(f"Warning: Image file not found at {path}. Skipping.")
            except Exception as e:
                print(f"Warning: Could not process image {path}. Error: {e}. Skipping.")

        return preprocessed_images

    def generate_single_variation(self, input_images_paths, strength=0.7, use_rotation=False, rotation_index=0, guidance_scale=7.5, num_inference_steps=30, progress_callback=None):
        """
        Generate a single variation with optional rotation

        Args:
            input_images_paths (List[str]): Paths to input images.
            strength (float): Image-to-image strength (0-1).
            use_rotation (bool): If True, use the image at rotation_index as base.
            rotation_index (int): Index of the image to use as base if use_rotation is True.
            guidance_scale (float): Controls how much the prompt guides generation.
            num_inference_steps (int): Number of denoising steps.
            progress_callback (Callable, optional): Callback function for progress updates.
                                                   It will be called by the diffusers pipeline with `(step, timestep, latents)`.
                                                   `step` is the current diffusion step, `timestep` is the scheduler's current timestep, `latents` are the current latents.

        Returns:
            PIL.Image: Generated variation
        """
        # Preprocess input images
        input_images = self.preprocess_images(input_images_paths)
        
        if not input_images:
            print("Error: No valid input images could be preprocessed.")
            return None
            
        # Extract features from all images
        image_features = self.extract_image_features(input_images)
        
        # Generate prompt
        prompt = self.generate_enhanced_style_prompt(image_features, len(input_images))
        negative_prompt = "low quality, blurry, distorted, deformed, unrealistic, amateurish, text, signature, watermark, multiple pieces, collage"
        
        # Random seed for variation
        seed = random.randint(1, 1000000)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Select the base image
        if use_rotation and len(input_images) > 1:
            # Use the specified image as base
            base_image_index = rotation_index % len(input_images)
            base_image = input_images[base_image_index]
        else:
            # Use the first image as base
            base_image = input_images[0]
            
        try:
            # Ensure base_image is a PIL Image
            if not isinstance(base_image, Image.Image):
                raise TypeError("Base image must be a PIL Image.")
                
            # Generate the variation
            variation = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=base_image,
                strength=strength,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                callback_steps=1 if progress_callback else 0, # Call callback at each step if provided
                callback=progress_callback
            ).images[0]
            
            return variation
            
        except Exception as e:
            print(f"Error during single variation generation: {e}")
            return None

    def generate_multi_style_variations(
        self,
        input_image_paths,
        num_variations=3,
        strength=0.7,
        guidance_scale=7.5, # Added guidance scale parameter
        num_inference_steps=30, # Added inference steps parameter
        use_rotation=True,
        progress_callback_per_image=None # Callback for each image's generation progress
    ):
        """
        Generate variations using multi-image style transfer.

        Args:
            input_image_paths (List[str]): Paths to input images.
            num_variations (int): Number of variations to generate.
            strength (float): Image-to-image strength (0-1). Higher values preserve less of the original structure.
            guidance_scale (float): Controls how much the prompt guides generation.
            num_inference_steps (int): Number of denoising steps.
            use_rotation (bool): If True, rotate through input images as the base image for generation. If False, use only the first image as base.
            progress_callback_per_image (Callable, optional): Callback function for progress updates for each generated image.
                                                              It will be called with `(variation_loop_idx, total_variations_for_this_base, pipe_step, pipe_total_steps, pipe_timestep, pipe_latents)`.
                                                              `variation_loop_idx` is the index of the current variation being generated for a base.
                                                              `total_variations_for_this_base` is the num_variations for the current base.
                                                              `pipe_step`, `pipe_total_steps` (same as num_inference_steps), `pipe_timestep`, `pipe_latents` are from the diffusion pipeline.


        Returns:
            List[PIL.Image]: Generated variations.
        """
        # Preprocess input images (load, resize, pad)
        input_images = self.preprocess_images(input_image_paths)

        if not input_images:
            print("Error: No valid input images could be preprocessed.")
            return [] # Return empty list if no images loaded

        # Extract aggregated features from all valid preprocessed images
        image_features = self.extract_image_features(input_images)

        # Generate a prompt based on the features / number of images
        prompt = self.generate_enhanced_style_prompt(image_features, len(input_images))
        negative_prompt = "low quality, blurry, distorted, deformed, unrealistic, amateurish, text, signature, watermark, multiple pieces, collage"

        generated_variations = []
        num_base_images = len(input_images)

        for i in range(num_variations):
            # Random seed for each variation for diversity
            seed = random.randint(1, 1000000)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Generating variation {i+1}/{num_variations} with seed {seed}")

            # Select the base image for img2img
            if use_rotation and num_base_images > 1:
                # Rotate through available input images
                base_image_index = i % num_base_images
                base_image = input_images[base_image_index]
                # print(f"Using base image index: {base_image_index}")
            else:
                # Always use the first image as the base
                base_image = input_images[0]
                # print("Using first image as base.")

            # Adjust strength slightly based on number of input images - subtle effect
            # adaptive_strength = min(strength * (1 + 0.02 * num_base_images), 0.95)
            adaptive_strength = strength # Using fixed strength might be more predictable

            # Generate variation using the pipeline
            try:
                 # Ensure base_image is a PIL Image
                 if not isinstance(base_image, Image.Image):
                     raise TypeError("Base image must be a PIL Image.")

                 # Wrapper for the progress_callback_per_image to include image_idx and total_images
                 current_image_callback = None
                 if progress_callback_per_image:
                     def wrapped_callback(step, timestep, latents):
                         # The diffuser callback might send different number of arguments,
                         # we are interested in the step and the total_steps (which is num_inference_steps)
                         progress_callback_per_image(i, num_variations, step, num_inference_steps, timestep, latents)
                     current_image_callback = wrapped_callback

                 # Run the img2img pipeline
                 variation = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=base_image, # Pass the selected PIL image
                    strength=adaptive_strength,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    callback_steps=1 if progress_callback_per_image else 0,
                    callback=current_image_callback
                 ).images[0] # Get the first (and usually only) image from the output list

                 generated_variations.append(variation)
                 print(f"Variation {i+1} generated successfully.")

            except Exception as e:
                 print(f"Error during pipeline generation for variation {i+1}: {e}")
                 # Decide if you want to skip or stop on error
                 # continue # Skip this variation
                 # raise # Stop generation

        return generated_variations

    # The composite base image function might not be needed if using rotation or single base logic,
    # but keeping it here for potential future use.
    def create_composite_base_image(self, input_images):
        """
        Create a composite base image by averaging multiple input images (simple averaging).

        Args:
            input_images (List[PIL.Image]): List of preprocessed input images.

        Returns:
            PIL.Image or None: Composite image or None if no input.
        """
        if not input_images:
            return None

        if len(input_images) == 1:
            return input_images[0]

        # Ensure all images are NumPy arrays for calculation
        img_arrays = [np.array(img).astype(float) for img in input_images]

        # Calculate the average
        result_array = np.mean(np.stack(img_arrays), axis=0)

        # Convert back to PIL Image
        composite_image = Image.fromarray(result_array.astype(np.uint8))
        return composite_image
