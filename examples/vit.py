import rpyc

ft_spec = 0
model_name = "google/vit-large-patch16-224"
dataset_path = "Maysee/tiny-imagenet"
dataset_name = None
training_args = {
    "per_device_train_batch_size": 16,
    "max_steps": 30,
}

client = rpyc.connect("localhost", 27322)
client.root.run_model(
    ft_spec, model_name, dataset_path, dataset_name, None, None, training_args
)
