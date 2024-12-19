import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from datasets import load_dataset
from tqdm.auto import tqdm
import random

# --- Configuration ---
MODEL_NAME = "gpt2"  # You can choose a smaller model like "distilgpt2" for faster training
DATASET_NAME = "wikitext"  # Or a dialogue dataset like "daily_dialog"
DATASET_SPLIT = "train"
CONTEXT_LENGTH = 128  # Maximum sequence length
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL_PATH = "trained_model"

# --- Model Definition (Simplified for 1 Million Parameters) ---
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=4, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embedded = self.embedding(input_ids)
        
        # For causal language modeling, we create a mask to prevent attending to future tokens
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).to(DEVICE)
        
        seq_len = input_ids.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len))).to(DEVICE)

        transformer_output = self.transformer(
            src=embedded,
            tgt=embedded,
            src_mask=causal_mask,
            tgt_mask=causal_mask,
            src_key_padding_mask=(1 - attention_mask).bool(),
            tgt_key_padding_mask=(1 - attention_mask).bool()
        )

        logits = self.fc(transformer_output)

        loss = None
        if labels is not None:
            # Shift the labels to the right for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss, logits

    def generate(self, input_ids, attention_mask=None, max_length=50, num_beams=5, temperature=1.0, top_k=50, top_p=0.95):
        """
        Generates text using beam search.

        Args:
            input_ids: Tensor of input token IDs.
            attention_mask: Tensor indicating which tokens are padding (0) and which are not (1).
            max_length: Maximum length of the generated sequence.
            num_beams: Number of beams for beam search.
            temperature: Controls the randomness of the output.
            top_k: Limits the sampling to the top k tokens.
            top_p: Limits the sampling to tokens above a cumulative probability threshold.
        """
        self.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            generated = input_ids
            past_key_values = None

            for _ in range(max_length):
                outputs = self(generated, attention_mask=attention_mask)
                logits = outputs[1][:, -1, :] / temperature  # Apply temperature scaling

                # Apply top-k and top-p filtering
if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')

                # Sample the next token
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

                # Append the next token to the generated sequence
                generated = torch.cat((generated, next_token), dim=1)

                # Update attention mask if provided
                if attention_mask is not None:
                    attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype).to(DEVICE)), dim=1)

                # Check if the end-of-sequence token was generated
                if next_token.item() == tokenizer.eos_token_id:
                    break

            return generated
# --- Data Loading and Preprocessing ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token for consistent sequence lengths

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=CONTEXT_LENGTH)

dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

class TextDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.tokenized_dataset[idx]["input_ids"]),
            "attention_mask": torch.tensor(self.tokenized_dataset[idx]["attention_mask"]),
            "labels": torch.tensor(self.tokenized_dataset[idx]["input_ids"]),
        }

train_dataset = TextDataset(tokenized_datasets)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model Initialization ---
# Calculate approximate vocab size for our simplified model
vocab_size = tokenizer.vocab_size  # Use the tokenizer's vocab size
model = SimpleGPT(vocab_size)

# --- Parameter Counting (for Verification) ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {count_parameters(model):,}")

# --- Optimizer and Scheduler ---
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=num_training_steps,
)

# --- Training Loop ---
model.to(DEVICE)
model.train()

progress_bar = tqdm(range(num_training_steps))

for epoch in range(NUM_EPOCHS):
    for batch in train_dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})

# --- Save the Model ---
torch.save(model.state_dict(), SAVE_MODEL_PATH)
print(f"Model saved to {SAVE_MODEL_PATH}")

# --- Turn-Based Interaction ---
model.eval()

def interact_with_model(model, tokenizer, history=None, max_turns=5):
    if history is None:
        history = []

    for turn in range(max_turns):
        user_input = input("You: ")
        history.append(f"You: {user_input}")

        # Prepare the input for the model
        input_text = "\n".join(history) + "\nAI:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
        attention_mask = torch.ones_like(input_ids).to(DEVICE)

        # Generate a response
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=CONTEXT_LENGTH + input_ids.size(1),
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract only the AI's response from the generated text
        response = response[len(input_text):].strip()

        print(f"AI: {response}")
        history.append(f"AI: {response}")

        # Check if the conversation should end (e.g., based on a keyword or maximum turns)
        if "bye" in user_input.lower() or "goodbye" in user_input.lower():
            print("Ending conversation.")
            break

# Start the interaction
interact_with_model(model, tokenizer)
