import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import t5
from torch.nn.utils.rnn import pad_sequence

from PIL import Image

from datasets import load_dataset

from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from typing import List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor


# Collator adjusted for local dataset
class Collator:
    def __init__(self, image_size, text_label, image_label, name, channels):
        self.text_label = text_label
        self.image_label = image_label
        self.name = name
        self.channels = channels
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        
    def __call__(self, batch):
        texts = []
        masks = []
        images = []
        for item in batch:
            try:
                # Load image from local file
                image_path = 'data/'+item[self.image_label]  # Assuming this is a path to the image file
                with Image.open(image_path) as img:
                    image = self.transform(img.convert(self.channels))
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                continue

            # Encode the text
            text, mask = t5.t5_encode_text(
                [item[self.text_label]], 
                name=self.name, 
                return_attn_mask=True
                )
            texts.append(torch.squeeze(text))
            masks.append(torch.squeeze(mask))
            images.append(image)

        if len(texts) == 0:
            return None
        
        # Are these strictly necessary?
        texts = pad_sequence(texts, True)
        masks = pad_sequence(masks, True)

        newbatch = []
        for i in range(len(texts)):
            newbatch.append((images[i], texts[i], masks[i]))

        return torch.utils.data.dataloader.default_collate(newbatch)


class GlyffuserPipeline(DiffusionPipeline):
    r'''
    Pipeline for text-to-image generation from the glyffuser model

    Parameters:
        unet (['UNet2DConditionModel'])
        scheduler (['SchedulerMixin'])
        text_encoder (['TextEncoder']) - T5 small
    '''
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(
            unet=unet, 
            scheduler=scheduler,
            )

    @torch.no_grad()
    def __call__(
        self,
        texts: List[str],
        text_encoder: str = "google-t5/t5-small",
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        '''
        Docstring
        '''        
        # Get text embeddings
        # Encode the text
        # text_embeddings = []
        # for text in texts:
        #     embedding = t5.t5_encode_text(text, name=text_encoder)
        #     text_embeddings.append(torch.squeeze(embedding))
        # text_embeddings = pad_sequence(text_embeddings, True)

        batch_size = len(texts)

        text_embeddings, masks = t5.t5_encode_text(texts, name=text_encoder, return_attn_mask=True)

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        
        # if self.device.type == "mps": # MPS is apple silicon
        #     # randn does not work reproducibly on mps
        #     image = randn_tensor(image_shape, generator=generator)
        #     image = image.to(self.device)
        # else:
        image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(
                image, 
                t,
                encoder_hidden_states=text_embeddings, # Add text encoding input
                encoder_attention_mask=masks, # Add attention mask
                return_dict=False
                )[0] # <-- sample is an attribute of the BaseOutClass of type torch.FloatTensor

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator, return_dict=False)[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
    
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, texts, pipeline):
    images = pipeline(
        texts,
        batch_size = config.eval_batch_size, 
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Generator must be on CPU for sampling during training
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")