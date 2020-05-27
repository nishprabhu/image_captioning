""" Utility functions """

import os
import numpy as np
from transformers import BertTokenizer


def get_text(outputs):
    """ Function to convert numpy array of token ids to a list of strings  """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    captions = []
    for output in outputs:
        captions.append(tokenizer.decode(output))
    return captions


def get_best_model(path):
    """ Return the filename of the model with least validation error """
    best_model = None
    best_loss = np.inf
    models = os.listdir(path)
    for model in models:
        val_loss = model.split("=")[-1].split("c")[0][:-1]
        try:
            val_loss = float(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
        except Exception:
            continue
    return os.path.join("models", best_model)


if __name__ == "__main__":
    model = get_best_model("models/")
    print(model)
