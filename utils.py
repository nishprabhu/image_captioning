""" Utility functions """

import os
import numpy as np


def get_text(outputs, tokenizer):
    """ Function to convert numpy array of token ids to a list of strings  """
    captions = []
    for output in outputs:
        captions.append(tokenizer.decode(output, skip_special_tokens=True))
    return captions


def get_best_model(path):
    """ Return the filename of the model with least validation error """
    best_model = None
    best_loss = np.inf
    models = os.listdir(path)
    for model in models:
        val_loss = model.split("=")[-1].split(".ckpt")[0]
        try:
            val_loss = float(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
        except Exception:
            continue
    if best_model is None:
        print(
            "Not able to detect the path to the best performing model. Please set it manually."
        )
    return os.path.join("models", best_model)


if __name__ == "__main__":
    model = get_best_model("models/")
    print(model)
