import tensorflow as tf
import keras
new_model = tf.keras.models.load_model('G:/rauf/STEPBYSTEP/Projects/NLP/Text_Classification_CNN/Text_Classification_CNN/saved_model.pb')
new_model.summary()
