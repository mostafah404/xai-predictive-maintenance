"""
Trains the LSTM and the CNN again and saves ../data/model.pth and ../data/cnn_model.pth.
It runs the data preparation cells, the LSTM training cell and the CNN training cell of data_preprocessing.ipynb one after the other
(nothing else of the notebook), so the code is the one of the notebook. The weights of the best epoch are kept.

    python retrain_lstm_cnn.py

The notebook sets all random seeds in its first cell, so the result is the same on every run on the same machine.
If you run the whole notebook from the top instead, the CNN differs slightly, because other cells use the random generator in between.
"""
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
cells = json.load(open("data_preprocessing.ipynb"))["cells"]


def find(pattern):
    return next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and re.search(pattern, "".join(c["source"])))


last_lstm = find(r"best_val_loss = float")
cnn_class, cnn_train = find(r"class CNNModel"), find(r"best_cnn_loss = float")
order = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and i <= last_lstm] + [cnn_class, cnn_train]
namespace = {"__name__": "__main__"}
for i in order:
    source = "".join(cells[i]["source"])
    source = "\n".join(line for line in source.split("\n") if not line.lstrip().startswith(("%", "!")))
    start = time.time()
    exec(compile(source, f"cell {i}", "exec"), namespace)
    print(f"--- cell {i} done ({time.time() - start:.0f} s)", flush=True)
print("saved ../data/model.pth and ../data/cnn_model.pth")
