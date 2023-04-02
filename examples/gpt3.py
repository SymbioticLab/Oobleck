import rpyc

ft_spec = 0
model_name = "gpt2"
model_size = "13b"  # "small", "medium", "large", "xl", "2.7b", "6.7b", "13b", "175b"
dataset_path = "wikitext"
dataset_name = "wikitext-103-raw-v1"
if model_size == "small":
    model_args = {
        "n_positions": 2048,
        "n_layers": 12,
        "n_embd": 768,
        "n_head": 12,
    }
elif model_size == "medium":
    model_args = {
        "n_positions": 2048,
        "n_layers": 24,
        "n_embd": 1024,
        "n_head": 16,
    }
elif model_size == "large":
    model_args = {
        "n_positions": 2048,
        "n_layers": 24,
        "n_embd": 1536,
        "n_head": 16,
    }
elif model_size == "xl":
    model_args = {
        "n_positions": 2048,
        "n_layers": 24,
        "n_embd": 2048,
        "n_head": 16,
    }
elif model_size == "2.7b":
    model_args = {
        "n_positions": 2048,
        "n_layers": 32,
        "n_embd": 2560,
        "n_head": 32,
    }
elif model_size == "6.7b":
    model_args = {
        "n_positions": 2048,
        "n_layers": 32,
        "n_embd": 4096,
        "n_head": 32,
    }
elif model_size == "13b":
    model_args = {
        "n_positions": 2048,
        "n_layers": 40,
        "n_embd": 5120,
        "n_head": 40,
    }
elif model_size == "175b":
    model_args = {
        "n_positions": 2048,
        "n_layers": 96,
        "n_embd": 12288,
        "n_head": 96,
    }
else:
    raise ValueError("Invalid model size")
training_args = {
    "per_device_train_batch_size": 4,
    "max_steps": 10,
}

client = rpyc.connect("localhost", 27322)
client.root.run_model(
    ft_spec, model_name, dataset_path, dataset_name, model_args, training_args
)
