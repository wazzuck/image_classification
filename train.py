# Purpose: Implement the training loop for the Encoder-Decoder Transformer
#          for predicting digit sequences from tiled MNIST images.

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import TiledMNISTSeqDataset, vocab_size, pad_token_idx # Import dataset class and vocab info
from model import EncoderDecoderTransformer, generate_square_subsequent_mask # Need to import model
import torch.optim as optim # Need to import optimizer
from tqdm import tqdm # Import tqdm
from safetensors.torch import save_model # Import safetensors
import wandb # Import wandb

if __name__ == "__main__":
    # --- Configuration --- #
    num_encoder_layers = 3
    num_decoder_layers = 3
    num_heads = 4
    # vocab_size is imported from dataset
    img_embedding_dim = 64
    text_embedding_dim = 64
    learning_rate = 0.001
    batch_size = 128
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pad_token_idx is imported from dataset

    # --- Wandb Initialization --- #
    wandb.init(
        project="mnist-transformer-seq", # Replace with your project name
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "num_heads": num_heads,
            "img_embedding_dim": img_embedding_dim,
            "text_embedding_dim": text_embedding_dim,
            "vocab_size": vocab_size,
            "architecture": "EncoderDecoderTransformer",
            "dataset": "TiledMNISTSeqDataset",
            "device": device,
        }
    )
    # -------------------------- #

    print(f"Using device: {device}")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Padding Token Index: {pad_token_idx}")

    # --- Data Loading --- #
    print("\nLoading datasets...")
    train_dataset = TiledMNISTSeqDataset(split="train")
    test_dataset = TiledMNISTSeqDataset(split="test")

    # Consider using num_workers > 0 for faster loading if not on Windows/certain setups
    # Adjust based on your system's capabilities
    num_workers = 0 # Start with 0, increase if loading is a bottleneck
    if device == "cuda":
      pin_memory = True
      # Example: num_workers = 4 # Often a good starting point on Linux with GPUs
    else:
      pin_memory = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    print(f"  Train dataset size: {len(train_dataset)}")
    print(f"  Test dataset size: {len(test_dataset)}")
    print(f"  Train loader batches: {len(train_loader)}")
    print(f"  Test loader batches: {len(test_loader)}")
    print("Datasets loaded.")
    # -------------------- #

    # --- Model, Loss, Optimizer --- #
    print("\nInitializing model, loss, and optimizer...")
    # 1. Instantiate model: model = EncoderDecoderTransformer(...hyperparameters...)
    # --- Your model instantiation code here ---
    # model = EncoderDecoderTransformer(...
    #                             ).to(device)
    # model = None # Placeholder - REMOVE THIS WHEN MODEL IS IMPLEMENTED
    model = EncoderDecoderTransformer(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        vocab_size=vocab_size,
        img_embedding_dim=img_embedding_dim,
        text_embedding_dim=text_embedding_dim,
        max_seq_len=train_dataset.max_seq_len # Use max_seq_len from dataset
    ).to(device)

    # 2. Define loss function, ignoring padding: criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)

    # 3. Define optimizer.
    # --- Your optimizer definition code here ---
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = None # Placeholder - REMOVE THIS WHEN MODEL IS IMPLEMENTED
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Initialization complete.")
    # ------------------------------ #

    # --- Watch Model (Optional) --- #
    # Logs gradients and parameters to wandb. Can slightly slow down training.
    wandb.watch(model, log="gradients", log_freq=100) # Log gradients every 100 batches
    # ------------------------------ #

    # --- Training Loop ---
    print("\nStarting training loop...")
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        total_train_loss = 0.0
        # Wrap train_loader with tqdm for a progress bar
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_iterator:
            # Ensure model and optimizer are implemented before running this loop
            if model is None or optimizer is None:
                 print("ERROR: Model or optimizer not implemented. Skipping training loop.")
                 break # Exit inner loop

            # a. Get image, input_seq, target_seq from batch.
            image, input_seq, target_seq = batch
            # b. Move data to device.
            image, input_seq, target_seq = image.to(device), input_seq.to(device), target_seq.to(device)

            # c. Generate target mask for decoder self-attention:
            #    tgt_mask = generate_square_subsequent_mask(input_seq.size(1), device=device)
            # The mask is generated inside the model's forward pass now.
            # d. Zero gradients.
            optimizer.zero_grad()

            # e. Forward pass: output_logits = model(image, input_seq) # Mask generated internally
            # --- Your forward pass code here ---
            # output_logits = model(...) 
            output_logits = model(image, input_seq)

            # if output_logits is None: # Skip if forward pass not implemented
            #     print("ERROR: Model forward pass not implemented. Skipping batch.")
            #     continue

            # f. Calculate loss:
            #    - Reshape logits: output_logits.view(-1, vocab_size)
            #    - Reshape targets: target_seq.view(-1)
            #    - loss = criterion(output_logits_reshaped, target_seq_reshaped)
            loss = criterion(output_logits.reshape(-1, vocab_size), target_seq.reshape(-1))

            # g. Backward pass.
            loss.backward()
            # h. Optimizer step.
            optimizer.step()
            # i. Accumulate loss.
            total_train_loss += loss.item()

            # Update tqdm progress bar description with current average loss
            train_iterator.set_postfix(loss=total_train_loss / (train_iterator.n + 1))
            # Log batch loss to wandb
            wandb.log({"batch_train_loss": loss.item()})
        
        # Check if loop was broken due to missing model/optimizer
        if model is None or optimizer is None:
            break # Exit outer loop

        avg_train_loss = total_train_loss / len(train_loader)

        # -- Evaluation Phase -- #
        model.eval() # Set model to evaluation mode
        total_test_loss = 0.0
        # Wrap test_loader with tqdm for a progress bar
        test_iterator = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test ]")
        with torch.no_grad(): # Disable gradient calculations
            for batch in test_iterator:
                # a. Get image, input_seq, target_seq.
                image, input_seq, target_seq = batch
                # b. Move to device.
                image, input_seq, target_seq = image.to(device), input_seq.to(device), target_seq.to(device)

                # c. Generate target mask.
                #    tgt_mask = generate_square_subsequent_mask(input_seq.size(1), device=device)
                # The mask is generated inside the model's forward pass now.

                # d. Forward pass: output_logits = model(image, input_seq) # Mask generated internally
                # --- Your forward pass code here ---
                # output_logits = model(...) 
                output_logits = model(image, input_seq)

                # if output_logits is None: # Skip if forward pass not implemented
                #    print("ERROR: Model forward pass not implemented. Skipping eval batch.")
                #    continue

                # e. Calculate loss (using reshaped logits/targets and criterion).
                loss = criterion(output_logits.reshape(-1, vocab_size), target_seq.reshape(-1))
                # f. Accumulate loss.
                total_test_loss += loss.item()
                
                # Update tqdm progress bar description with current average loss
                test_iterator.set_postfix(loss=total_test_loss / (test_iterator.n + 1))

        avg_test_loss = total_test_loss / len(test_loader)

        # -- Log Epoch Metrics to Wandb -- #
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
        })
        # -------------------------------- #

        # -- Epoch Summary -- #
        # Use tqdm.write to print summary without interfering with progress bars
        tqdm.write(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    print("\nTraining finished.")

    # --- (Optional) Save Model --- #
    if model is not None:
        # torch.save(model.state_dict(), "mnist_encoder_decoder.pth") # Original torch save
        save_model(model, "mnist_encoder_decoder.safetensors") # Using safetensors
        print("Model saved to mnist_encoder_decoder.safetensors")

    # --- Finish Wandb Run --- #
    wandb.finish()
    # ------------------------ #