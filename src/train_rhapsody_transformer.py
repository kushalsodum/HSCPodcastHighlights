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

from preprocessData import filterHighlights, calculateMetrics

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
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
LORA_DROPOUT = 0.3
LORA_RANK = 8
LORA_ALPHA = 16
EPOCHS = 1 
VALIDATION_STEPS = 1000

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
        
        hidden_size = base_model.config.hidden_size
        
        # Transformer Encoder
        # d_model=hidden_size, nhead=16, num_layers=2
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=16, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Positional Embeddings (max 100 segments)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, hidden_size))
        
        # Classification head: Linear (output logits)
        self.classifier = nn.Linear(hidden_size, 1)
        
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
        
        # Add positional embeddings and cast to float
        x = selected_hidden_states.float() + self.pos_embedding[:, :num_segments, :].to(selected_hidden_states.device)
        
        # Pass through Transformer
        transformer_out = self.transformer(x) # (batch_size, 100, hidden_size)
        
        # Apply classifier to each time step
        logits = self.classifier(transformer_out) # (batch_size, 100, 1)
        
        return logits.squeeze(-1) # (batch_size, 100)

def run_validation(model, val_dataloader, criterion, device, step):
    if not val_dataloader:
        return
        
    print(f"\nStarting validation at step {step}...")
    model.eval()
    val_loss = 0
    all_metrics = {
        'hit_rate': [],
        'precision': [],
        'recall': [],
        'f1': [],
    }
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            segment_end_indices = batch['segment_end_indices'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, segment_end_indices)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            # Calculate metrics
            probs = torch.sigmoid(logits)
            probs_np = probs.cpu().numpy()[0]
            labels_np = labels.cpu().numpy()[0]
            
            # Get predicted indices > Threshold (0.3)
            pred_indices = np.where(probs_np > 0.3)[0].tolist()
            gt_indices = np.where(labels_np > 0.5)[0].tolist()
            
            metrics = calculateMetrics(pred_indices, gt_indices)
            for k, v in metrics.items():
                all_metrics[k].append(v)
    
    avg_val_loss = val_loss / len(val_dataloader)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    print(f"Validation Loss: {avg_val_loss:.4f}")
    for k, v in avg_metrics.items():
        print(f"Val {k}: {v:.4f}")
        
    wandb.log({
        "Val Loss": avg_val_loss,
        **{f"Val {k}": v for k, v in avg_metrics.items()}
    }, step=step)
    
    model.train()

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
    model.transformer.to(device)
    model.classifier.to(device)
    model.pos_embedding.data = model.pos_embedding.data.to(device)
    
    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Dataset
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    if not os.path.exists(train_path):
        print(f"Training file not found at {train_path}")
        return

    dataset = RhapsodyDataset(train_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Calculate pos_weight
    # num_positives = sum(len(x) for x in dataset.allGTHighlights)
    # total_segments = len(dataset) * 100
    # pos_weight_val = (total_segments - num_positives) / num_positives
    # pos_weight = torch.tensor([pos_weight_val]).to(device)
    # print(f"Calculated pos_weight: {pos_weight.item():.2f}")

    pos_weight = torch.tensor([5.0]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Validation Dataset
    val_path = os.path.join(DATA_DIR, VAL_FILE)
    val_dataset = RhapsodyDataset(val_path, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    wandb.init(entity="ksodum-university-of-michigan",
                project="final project Transformer testing")
    
    # Scheduler

    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    # Training Loop
    model.train()
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    total_loss = 0
    plot_loss = 0
    global_step = 0

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        segment_end_indices = batch['segment_end_indices'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, segment_end_indices)
        
        loss = criterion(logits, labels)
        loss = loss / GRAD_ACCUMULATION_STEPS
        loss.backward()

        plot_loss += loss.item()            
        
        if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            global_step += 1
            wandb.log({"Train loss": plot_loss, "Train step": global_step})
            plot_loss = 0
            
        total_loss += loss.item() * GRAD_ACCUMULATION_STEPS
        progress_bar.set_postfix({'loss': total_loss / (step + 1)})
        
        # Validation
        # if (step + 1) % VALIDATION_STEPS == 0:
        #     run_validation(model, val_dataloader, criterion, device, step + 1)
        
    print("Training finished.")
    
    # Save model
    output_dir = "rhapsody_model_transformer"
    os.makedirs(output_dir, exist_ok=True)
    model.base_model.save_pretrained(output_dir)
    torch.save(model.transformer.state_dict(), os.path.join(output_dir, "transformer.pt"))
    torch.save(model.classifier.state_dict(), os.path.join(output_dir, "classifier.pt"))
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
