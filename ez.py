import torch
import numpy
import random
from asd import LinearRegressionModel, y_preds_new
from asd import MODEL_SAVE_PATH
from asd import x_q
loaded_model= LinearRegressionModel()

print(loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH)))

print(loaded_model.state_dict())

loaded_model.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model(x_q)

print(loaded_model_preds)
print(y_preds_new)
