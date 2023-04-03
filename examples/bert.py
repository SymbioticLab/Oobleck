import rpyc

ft_spec = 0
model_name = "bert-large-uncased"
dataset_path = "wikitext"
dataset_name = "wikitext-2-raw-v1"
training_args = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 64,
    "max_steps": 30,
}

client = rpyc.connect("localhost", 27322)
client.root.run_model(
    ft_spec, model_name, dataset_path, dataset_name, "large", None, training_args
)
