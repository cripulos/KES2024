import tensorflow as tf
import tensorflow_model_optimization as tfmot
import os
from dotenv import load_dotenv


load_dotenv(os.getenv('ENVIRONMENT_FILE', './.env'), override=True)

global_segmentation_model_name = os.getenv('GLOBAL_SEGMENTATION_MODEL_NAME')
print(global_segmentation_model_name)
model = tf.keras.models.load_model(os.getenv('MODELS_PATH') + global_segmentation_model_name, compile=False)

def apply_pruning_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Dense):
    return tfmot.sparsity.keras.prune_low_magnitude(layer)
  return layer


model_for_pruning = tf.keras.models.clone_model(
    model,
    clone_function=apply_pruning_to_dense,
)

model.summary()
model_for_pruning.summary()
model_for_pruning.save("../Model/prunned_TF_lite")


