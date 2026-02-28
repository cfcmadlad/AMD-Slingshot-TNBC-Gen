import torch
import os
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# --- Configuration ---
MODEL_ID   = "runwayml/stable-diffusion-v1-5"
DATA_DIR   = "/content/drive/MyDrive/AMD_TNBC/data"
OUTPUT_DIR = "/content/drive/MyDrive/AMD_TNBC/checkpoints"
LORA_RANK  = 32
LR         = 1e-4
BATCH_SIZE = 1
MAX_STEPS  = 1000
SAVE_EVERY = 200
DEVICE     = "cuda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Dataset ---
class TNBCDataset(Dataset):
    def __init__(self, data_dir, tokenizer, size=512):
        self.data_dir  = data_dir
        self.tokenizer = tokenizer
        self.size      = size
        self.samples   = sorted([f for f in os.listdir(data_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        txt_name = img_name.replace(".png", ".txt")

        img = Image.open(os.path.join(self.data_dir, img_name)).convert("RGB").resize((self.size, self.size))
        img_tensor = (torch.tensor(list(img.getdata()), dtype=torch.float32)
                      .reshape(self.size, self.size, 3).permute(2, 0, 1) / 127.5 - 1.0)

        with open(os.path.join(self.data_dir, txt_name)) as f:
            caption = f.read().strip()

        tokens = self.tokenizer(
            caption, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        return {"pixel_values": img_tensor, "input_ids": tokens.input_ids.squeeze()}

# --- Load Models ---
print("Loading models...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_enc  = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(DEVICE)
vae       = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(DEVICE)
unet      = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(DEVICE)
noise_sch = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

vae.requires_grad_(False)
text_enc.requires_grad_(False)

# --- Apply LoRA via PEFT ---
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=["to_q", "to_v", "to_k", "to_out.0"],
    lora_dropout=0.1,
    bias="none"
)
unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

optimizer  = torch.optim.AdamW(unet.parameters(), lr=LR)
dataset    = TNBCDataset(DATA_DIR, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Training Loop ---
print(f"Starting LoRA training on {len(dataset)} samples...")
unet.train()
step      = 0
data_iter = iter(dataloader)

while step < MAX_STEPS:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)

    pixel_values = batch["pixel_values"].to(DEVICE)
    input_ids    = batch["input_ids"].to(DEVICE)

    with torch.no_grad():
        latents    = vae.encode(pixel_values).latent_dist.sample() * 0.18215
        enc_hidden = text_enc(input_ids)[0]

    noise     = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_sch.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
    noisy_lat = noise_sch.add_noise(latents, noise, timesteps)

    pred = unet(noisy_lat, timesteps, enc_hidden).sample
    loss = torch.nn.functional.mse_loss(pred, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    step += 1

    if step % 50 == 0:
        print(f"Step {step}/{MAX_STEPS} | Loss: {loss.item():.4f}")

    if step % SAVE_EVERY == 0:
        ckpt = os.path.join(OUTPUT_DIR, f"lora_tnbc_step{step}.pt")
        torch.save(unet.state_dict(), ckpt)
        print(f"Checkpoint saved: {ckpt}")

print("Training complete.")
torch.save(unet.state_dict(), os.path.join(OUTPUT_DIR, "lora_tnbc_final.pt"))
print("Final weights saved.")