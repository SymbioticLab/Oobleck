from pippy import annotate_split_points, PipeSplitWrapper

# Sharding for the Google's HuggingFace ViT model
# e.g. google/vit-base-patch16-224 (https://huggingface.co/google/vit-base-patch16-224)
def vit_add_split_points(vit):
    for i in range(0, vit.config.num_hidden_layers):
        annotate_split_points(
            vit,
            {f"vit.encoder.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING},
        )
    annotate_split_points(vit, {"vit.layernorm": PipeSplitWrapper.SplitPoint.BEGINNING})


# Sharding for the Microsoft's HuggingFace ResNet model
# e.g. microsoft/resnet-152 (https://huggingface.co/microsoft/resnet-152)
def resnet_add_split_points(resnet):
    for i in range(0, len(resnet.config.depths)):
        for j in range(0, resnet.config.depths[i]):
            for k in range(0, 3):
                annotate_split_points(
                    resnet,
                    {
                        f"resnet.encoder.stages.{i}.layers.{j}.layer.{k}": PipeSplitWrapper.SplitPoint.BEGINNING
                    },
                )

    annotate_split_points(
        resnet, {"resnet.pooler": PipeSplitWrapper.SplitPoint.BEGINNING}
    )
