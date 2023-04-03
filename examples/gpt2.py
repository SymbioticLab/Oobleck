import rpyc

ft_spec = 0
model_name = "gpt2"
model_size = "xl"  # "small", "medium", "large", "xl"
dataset_path = "wikitext"
dataset_name = "wikitext-103-raw-v1"
if model_size == "small":
    model_args = {
        "n_layers": 12,
        "n_positions": 1024,
        "n_embd": 768,
        "n_head": 12,
    }
elif model_size == "medium":
    model_args = {
        "n_layers": 24,
        "n_positions": 1024,
        "n_embd": 1024,
        "n_head": 16,
    }
elif model_size == "large":
    model_args = {
        "n_layers": 36,
        "n_positions": 1024,
        "n_embd": 1280,
        "n_head": 20,
    }
elif model_size == "xl":
    model_args = {
        "n_layers": 48,
        "n_positions": 1024,
        "n_embd": 1600,
        "n_head": 25,
    }
else:
    raise ValueError("Invalid model size")
training_args = {
    "per_device_train_batch_size": 4,
    "max_steps": 30,
}

client = rpyc.connect("localhost", 27322)
client.root.run_model(
    ft_spec,
    model_name,
    dataset_path,
    dataset_name,
    model_size,
    model_args,
    training_args,
)
