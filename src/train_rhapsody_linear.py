import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig
import torch.nn as nn
from tqdm import tqdm
import wandb

from preprocessData import filterHighlights

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Configuration
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR = "../rhapsody/data"
TRAIN_FILE = "train-00000-of-00001.parquet"
VAL_FILE = "val-00000-of-00001.parquet"
HF_TOKEN_FILE = "../huggingFaceToken.txt"

# Hyperparameters
BATCH_SIZE = 1
GRAD_ACCUMULATION_STEPS = 8
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.01
LORA_DROPOUT = 0.3
LORA_RANK = 8
LORA_ALPHA = 16
EPOCHS = 1 
VALIDATION_STEPS = 1000 # Run validation every 1000 steps

class RhapsodyDataset(Dataset):
    def __init__(self, parquet_file, tokenizer):
        self.df = pd.read_parquet(parquet_file)
        self.tokenizer = tokenizer
        allReplayData = self.df['gt'].to_numpy()
        self.allGTHighlights = filterHighlights(allReplayData)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        summaries = row['segment_summaries']
        gtHighlights = self.allGTHighlights[idx]
        
        # Create binary labels
        labels = [0.0] * 100
        for highlight in gtHighlights:
            labels[highlight] = 1
        
        # Tokenize and concatenate
        input_ids = []
        segment_end_indices = []
        
        # Use the tokenizer's eos_token_id as the separator
        sep_token_id = self.tokenizer.eos_token_id
        
        for summary in summaries:
            tokens = self.tokenizer.encode(summary, add_special_tokens=False)
            input_ids.extend(tokens)
            input_ids.append(sep_token_id)
            segment_end_indices.append(len(input_ids) - 1)
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'segment_end_indices': torch.tensor(segment_end_indices, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

class SegmentClassificationLLM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Classification head: Linear + Sigmoid
        # Input dim is hidden size of LLM.
        self.head = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, segment_end_indices):
        # input_ids: (batch_size, seq_len)
        # segment_end_indices: (batch_size, 100)
        
        outputs = self.base_model(input_ids=input_ids)
        hidden_states = outputs.last_hidden_state # (batch_size, seq_len, hidden_size)
        
        # Extract hidden states at segment_end_indices
        batch_size = input_ids.size(0)
        num_segments = segment_end_indices.size(1)
        
        # Create batch indices
        batch_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(1).expand(-1, num_segments)

        # Gather
        selected_hidden_states = hidden_states[batch_indices, segment_end_indices, :] # (batch_size, 100, hidden_size)

        # print("\nPrinting sizes")
        # print(num_segments)
        # print(batch_indices.size())
        # print(selected_hidden_states.size())
        
        # Apply head
        probs = self.head(selected_hidden_states.float()) # (batch_size, 100, 1)
        return probs.squeeze(-1) # (batch_size, 100)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Base Model with QLoRA config if possible
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        print("Using QLoRA (4-bit quantization)")
        base_model = AutoModel.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Could not load with quantization: {e}")
        print("Falling back to standard loading")
        base_model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16
        )
        base_model.to(device)

    # Apply PEFT
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=None, 
        target_modules=["q_proj", "v_proj"] 
    )
    
    base_model = get_peft_model(base_model, peft_config)
    base_model.print_trainable_parameters()
    
    model = SegmentClassificationLLM(base_model)
    model.head.to(device)
    
    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()
    
    # Dataset
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    if not os.path.exists(train_path):
        print(f"Training file not found at {train_path}")
        return

    dataset = RhapsodyDataset(train_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Training Loop
    model.train()
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    total_loss = 0
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        segment_end_indices = batch['segment_end_indices'].to(device)
        labels = batch['labels'].to(device)
        
        probs = model(input_ids, segment_end_indices)
        
        loss = criterion(probs, labels)
        loss = loss / GRAD_ACCUMULATION_STEPS
        loss.backward()
        
        if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRAD_ACCUMULATION_STEPS
        progress_bar.set_postfix({'loss': total_loss / (step + 1)})
        
    print("Training finished.")
    
    # Save model
    output_dir = "rhapsody_model_linear"
    os.makedirs(output_dir, exist_ok=True)
    model.base_model.save_pretrained(output_dir)
    torch.save(model.head.state_dict(), os.path.join(output_dir, "head.pt"))
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
