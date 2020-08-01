from invariances.model.cinn import ConditionalTransformer


def get_model(name):
    _models = {
        "alexnet_conv5_animals": lambda: ConditionalTransformer.from_pretrained("animals"),
    }
    return _models[name]()
