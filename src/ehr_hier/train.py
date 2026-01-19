import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- Import your modules ---
from transformer.model import AdaptiveEpisodicTransformer
from transformer.loss import AETLossModule
from transformer.collator import AETHierarchicalCollator


# from data.dataset import MEDSDataset
# from data.vocabulary import GlobalVocabulary

# --- Configuration ---
class TrainConfig:
    # Model Params
    d_model = 768
    num_heads = 12
    d_ff = 3072
    num_local_layers = 4
    num_global_layers = 6
    rope_max_period = 10000.0
    dropout = 0.1

    # Training Params
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 0.01
    epochs = 10
    grad_clip = 1.0
    log_interval = 100
    save_path = "./checkpoints"

    # Vocab Config (Must match GlobalVocabulary offsets)
    vocab_config = {
        'total_size': 400000,
        'size_special': 1000,
        'size_rvq': 4000,
        'size_meas_labels': 20000,
        'size_meds': 370000,
        # Offsets needed for Loss calculation
        'offsets': {
            'SPECIAL': 0,
            'RVQ': 1000,
            'MEAS': 10000,
            'MED': 30000
        }
    }


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler):
    model.train()
    total_loss = 0.0

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(pbar):
        # 1. Move Batch to Device
        # The collator returns a dict of tensors
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # 2. Forward Pass (Mixed Precision)
        with torch.amp.autocast('cuda'):
            # Note: We pass the inputs as kwargs to match model signature
            # Input: ids, times, values, types, mask
            head_outputs, _ = model(
                input_ids=inputs['input_ids'],
                time_ids=inputs['time_ids'],
                numeric_values=inputs['numeric_values'],
                token_type_ids=inputs['token_type_ids'],
                attention_mask=inputs['attention_mask'],
                window_start_times=inputs.get('window_start_times', None),
                window_mask=inputs.get('window_mask', None),
                window_type_ids=inputs.get('window_type_ids', None),
                # prev_global_state=None (Assuming independent segments for now)
            )

            # 3. Calculate Loss
            # We pass the outputs AND the targets (inputs) to the loss module
            loss, loss_logs = criterion(head_outputs, inputs)

        # 4. Backward Pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # 5. Gradient Clipping (Unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainConfig.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # 6. Logging
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']

        # Update progress bar with specific head losses
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'struct': f"{loss_logs.get('loss_struct', 0):.3f}",
            'med': f"{loss_logs.get('loss_med', 0):.3f}",
            'lr': f"{current_lr:.2e}"
        })


    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            with torch.amp.autocast('cuda'):
                head_outputs, _ = model(
                    input_ids=inputs['input_ids'],
                    time_ids=inputs['time_ids'],
                    numeric_values=inputs['numeric_values'],
                    token_type_ids=inputs['token_type_ids'],
                    attention_mask=inputs['attention_mask'],
                    window_start_times=inputs.get('window_start_times', None),
                    window_mask=inputs.get('window_mask', None),
                    window_type_ids=inputs.get('window_type_ids', None),
                )
                loss, _ = criterion(head_outputs, inputs)

            total_val_loss += loss.item()

    return total_val_loss / len(dataloader)


def main():
    # 0. Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    if not os.path.exists(TrainConfig.save_path):
        os.makedirs(TrainConfig.save_path)

    # 1. Prepare Data
    # vocab = GlobalVocabulary()
    # train_dataset = MEDSDataset(..., split='train')
    # val_dataset = MEDSDataset(..., split='val')

    # collator = AETHierarchicalCollator(vocab, max_windows=64, max_len_per_window=128)

    # train_loader = DataLoader(train_dataset, batch_size=TrainConfig.batch_size,
    #                           shuffle=True, collate_fn=collator, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=TrainConfig.batch_size,
    #                         shuffle=False, collate_fn=collator, num_workers=4)

    # --- MOCK DATA LOADER (For Testing the Loop) ---
    print("WARNING: Using Mock Data. Uncomment real data loading above.")
    from transformer.collator import AETHierarchicalCollator  # Just for class ref
    train_loader = [
        {
            'input_ids': torch.randint(0, 400000, (32, 4, 128)),
            'time_ids': torch.rand(32, 4, 128),
            'numeric_values': torch.randn(32, 4, 128, 1),
            'token_type_ids': torch.randint(0, 5, (32, 4, 128)),
            'attention_mask': torch.ones(32, 4, 128)
        }
    ]  # List of 1 batch
    val_loader = train_loader
    # -----------------------------------------------

    # 2. Initialize Model
    model = AdaptiveEpisodicTransformer(TrainConfig, TrainConfig.vocab_config).to(device)

    # 3. Initialize Loss
    # Define custom weights if needed (e.g., Structure is 5x more important)
    loss_weights = {'struct': 5.0, 'rvq': 1.0, 'meas': 1.0, 'med': 1.0, 'val': 1.0}
    criterion = AETLossModule(vocab_config=TrainConfig.vocab_config, weights=loss_weights).to(device)

    # 4. Optimizer
    # Separate weight decay for embeddings/weights vs biases/layernorms
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n],
         'weight_decay': TrainConfig.weight_decay},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(param_groups, lr=TrainConfig.learning_rate)

    # Gradient Scaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    # 5. Training Loop
    for epoch in range(1, TrainConfig.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, f"{TrainConfig.save_path}/aet_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
