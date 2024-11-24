import tensorflow as tf

# Load model from json and weights files
with open('model.json', 'r') as json_file:
    json_config = json_file.read()

model = tf.keras.models.model_from_json(json_config)
model.load_weights('weights.bin')

# Save the model in .h5 format
model.save('model.h5')
