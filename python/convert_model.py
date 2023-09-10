import tensorflow as tf
import tensorflow.keras as keras

loaded_model = keras.models.load_model("flowers.h5")
tf.saved_model.save(loaded_model, "model/1")