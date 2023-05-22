import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import MultiHeadAttention, Dropout, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from transformers import TFRobertaModel
import optuna
import os
from sklearn.model_selection import train_test_split

def load_preprocessed_data():
    loaded_normalized_data = np.load('D:\\fliptest\\normalized_osu_data.npy', allow_pickle=True).item()
    timing_points_normalized = loaded_normalized_data['timing_points']
    hit_objects_normalized = loaded_normalized_data['hit_objects']
    
    return timing_points_normalized, hit_objects_normalized

def preprocess_data_for_gan(timing_points_normalized, hit_objects_normalized):
    # Convert the data into the desired format for GAN input
    gan_input_data = []

    for tp_data, ho_data in zip(timing_points_normalized, hit_objects_normalized):
        # Flatten the data and concatenate timing points and hit objects data
        flat_tp_data = tp_data.flatten()
        flat_ho_data = [ho[:-1] for ho in ho_data]  # Exclude the object_type and slider_points
        flat_ho_data = np.array(flat_ho_data).flatten()
        
        # Concatenate and append to the gan_input_data
        combined_data = np.concatenate((flat_tp_data, flat_ho_data))
        gan_input_data.append(combined_data)

    gan_input_data = np.array(gan_input_data)

    return gan_input_data

def main():
    # Load the preprocessed data
    timing_points_normalized, hit_objects_normalized = load_preprocessed_data()

    # Preprocess the data for GAN input
    gan_input_data = preprocess_data_for_gan(timing_points_normalized, hit_objects_normalized)

    # Split the dataset
    X_train, X_val, X_test = train_test_split(gan_input_data, test_size=0.3, random_state=42)

    # The variables X_train, X_val, and X_test now contain the data in a format that can be fed into a GAN model
    print(X_train)
    print(X_val)
    print(X_test)

main()

# Transformer block
def transformer_block(inputs, d_model, num_heads, dff, rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = Dense(dff, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)

    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2

import os

def save_beatmap_to_file(beatmap, metadata, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        # Write metadata to the file (you'll need to adjust this according to your metadata structure)
        file.write(f"Metadata:\n{metadata}\n")

        # Write timing points section
        file.write("[TimingPoints]\n")
        for tp in beatmap["timing_points"]:
            file.write(f"{tp[0]},{tp[1]}\n")

        # Write hit objects section
        file.write("[HitObjects]\n")
        for ho in beatmap["hit_objects"]:
            ho_type = "1" if ho[3] == "circle" else "2"
            slider_points = ",".join([f"{point[0]}:{point[1]}" for point in ho[4]])
            file.write(f"{ho[0]},{ho[1]},{ho[2]},{ho_type},0,{slider_points}\n")

def load_beatmap_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    timing_points = extract_timing_points_from_content(content)
    hit_objects = extract_hit_objects_from_content(content)

    # Extract metadata from the file (you'll need to adjust this according to your metadata structure)
    metadata = content.split('\n')[1]

    return {"timing_points": timing_points, "hit_objects": hit_objects}, metadata

def extract_timing_points_from_content(content):
    timing_points = []
    in_timing_points_section = False

    for line in content.split('\n'):
        if line.startswith('['):
            in_timing_points_section = line.startswith('[TimingPoints]')
            continue

        if not in_timing_points_section:
            continue

        if not line.strip():
            break

        values = line.strip().split(',')
        timing_points.append([float(values[0]), float(values[1])])

    return np.array(timing_points)

def extract_hit_objects_from_content(content):
    hit_objects = []
    in_hit_objects_section = False

    slider_pattern = re.compile(r'slider_points:(\S+)')

    for line in content.split('\n'):
        if line.startswith('['):
            in_hit_objects_section = line.startswith('[HitObjects]')
            continue

        if not in_hit_objects_section:
            continue

        if not line.strip():
            break

        values = line.strip().split(',')
        x, y, time = map(float, (values[0], values[1], values[2]))

        object_type = 'circle'
        slider_points = []

        if '2' in values[3]:
            object_type = 'slider'
            if len(values) > 5:
                slider_points = [tuple(map(float, point.split(':'))) for point in slider_pattern.findall(values[5])]

                hit_objects.append([x, y, time, object_type, slider_points])

    return np.array(hit_objects, dtype=object)

def expert_review(generated_beatmaps, metadata_batch, save_folder="generated_beatmaps", reviewed_folder="reviewed_beatmaps"):
    reviewed_beatmaps = []

    # Step 1: Export generated beatmaps to osu! editor format
    for i, (beatmap, metadata) in enumerate(zip(generated_beatmaps, metadata_batch)):
        save_path = os.path.join(save_folder, f"generated_beatmap_{i}.osu")
        save_beatmap_to_file(beatmap, metadata, save_path)

    print(f"Please review and adjust the generated beatmaps in '{save_folder}', and save the reviewed beatmaps in '{reviewed_folder}'.")

    input("Press Enter when you have finished reviewing the beatmaps...")

    # Step 3: Import reviewed beatmaps
    for i in range(len(generated_beatmaps)):
        reviewed_path = os.path.join(reviewed_folder, f"reviewed_beatmap_{i}.osu")
        reviewed_beatmap, reviewed_metadata = load_beatmap_from_file(reviewed_path)

        # Step 4: Preprocess the reviewed beatmaps
        preprocessed_reviewed_beatmap = preprocess_data_for_gan(reviewed_beatmap["timing_points"], reviewed_beatmap["hit_objects"])
        reviewed_beatmaps.append(preprocessed_reviewed_beatmap)

    reviewed_beatmaps = np.array(reviewed_beatmaps)

    return reviewed_beatmaps, metadata_batch

# Input layers
timesteps = ...
input_features = ...
metadata_features = ...
output_features = ...
review_interval = 10
num_reviewed_beatmaps = 2
expert_weight = 100

input_data = Input(shape=(timesteps, input_features))
metadata_input = Input(shape=(metadata_features,))

# Transformer layers and skip connections
trans1 = transformer_block(input_data, d_model=128, num_heads=4, dff=256)
trans2 = transformer_block(trans1, d_model=128, num_heads=4, dff=256)
skip_connection = Add()([trans1, trans2])

# Concatenate rhythm information (metadata)
concat = Concatenate()([skip_connection, metadata_input])

# Reshape data for 2D convolutions
reshape = Reshape((timesteps, input_features + metadata_features, 1))(concat)

# U-Net-like architecture
conv1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(reshape)
act1 = ReLU()(conv1)
conv2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(act1)
act2 = ReLU()(conv2)

upconv1 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(act2)
upconv2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(upconv1)

# Dense layers with batch normalization and dropout
dense1 = Dense(units=512)(upconv2)
bn1 = BatchNormalization()(dense1)
act3 = ReLU()(bn1)
dropout1 = Dropout(rate=0.5)(act3)

dense2 = Dense(units=256)(dropout1)
bn2 = BatchNormalization()(dense2)
act4 = ReLU()(bn2)
dropout2 = Dropout(rate=0.5)(act4)

# Output layer
output_generator = Dense(units=output_features, activation='tanh')(dropout2)

# Model definition
generator = Model(inputs=[input_data, metadata_input], outputs=output_generator)

# Discriminator
input_data_disc = Input(shape=(timesteps, input_features))
metadata_input_disc = Input(shape=(metadata_features,))
# Expand metadata_input to match input_data shape
expanded_metadata = RepeatVector(timesteps)(metadata_input_disc)

# Concatenate input_data and metadata_input
concat_disc = Concatenate(axis=-1)([input_data_disc, expanded_metadata])

# Temporal Convolutional Network (TCN) layers
tcn = TCN(nb_filters=64, kernel_size=5, nb_stacks=2, dilations=[1, 2, 4, 8, 16], 
          activation='relu', padding='causal', use_skip_connections=True)(concat_disc)
dropout1_disc = Dropout(rate=0.3)(tcn)

# Dense layers
dense1_disc = Dense(units=256, activation='relu')(dropout1_disc)
dropout2_disc = Dropout(rate=0.5)(dense1_disc)

# Flatten layer
flat_disc = Flatten()(dropout2_disc)

# Output layer
output_discriminator = Dense(units=1, activation='sigmoid')(flat_disc)

# Model definition
discriminator = Model(inputs=[input_data_disc, metadata_input_disc], outputs=output_discriminator)

# Combined model
discriminator.trainable = False
combined_output = discriminator([generator([input_data, metadata_input]), metadata_input])
combined_model = Model(inputs=[input_data, metadata_input], outputs=combined_output)

# Custom loss functions
def simple_difference(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def custom_generator_loss(y_true, y_pred, alpha=0.1):
    ce_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    domain_specific_loss = simple_difference(y_true, y_pred)
    total_loss = ce_loss + alpha * domain_specific_loss
    return total_loss

def custom_discriminator_loss(y_true, y_pred, beta=0.1):
    ce_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    domain_specific_loss = simple_difference(y_true, y_pred)
    total_loss = ce_loss + beta * domain_specific_loss
    return total_loss

# Compile the models with custom loss functions and optimizers
generator_optimizer = tf.keras.optimizers.Adam(lr=1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=1e-4)

combined_model.compile(loss=custom_generator_loss, optimizer=generator_optimizer)
discriminator.compile(loss=custom_discriminator_loss, optimizer=discriminator_optimizer)

# Load your training data here
train_data = load_training_data()  # Replace with your specific method to load training data

epochs = 100
batch_size = 64
num_batches = len(train_data) // batch_size

# Save checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

latent_dim = 100

# Training loop
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for epoch in range(epochs):
    for i in range(num_batches):
        idx = np.random.randint(0, train_data.shape[0], batch_size)
        real_beatmaps = train_data[idx]
        metadata_batch = metadata_input[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_beatmaps = generator.predict([noise, metadata_batch])

        beatmaps = np.concatenate((real_beatmaps, fake_beatmaps))
        metadata = np.concatenate((metadata_batch, metadata_batch))

        labels = np.zeros((2 * batch_size, 1))
        labels[:batch_size] = 1

        d_loss = discriminator.train_on_batch([beatmaps, metadata], labels)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined_model.train_on_batch([noise, metadata_batch], np.ones((batch_size, 1)))

        if i % 50 == 0:
            print(f"Epoch: {epoch}, Batch: {i}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

    # Save model checkpoints periodically
    if epoch % 5 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    # Periodically generate a batch of beatmaps for expert review
    if epoch % review_interval == 0:
        noise = np.random.normal(0, 1, (num_reviewed_beatmaps, latent_dim))
        generated_beatmaps = generator.predict([noise, metadata_batch[:num_reviewed_beatmaps]])

        reviewed_beatmaps, reviewed_metadata = expert_review(generated_beatmaps, metadata_batch[:num_reviewed_beatmaps])

        # Compute the differences between generated and reviewed beatmaps and use them as input
        beatmap_differences = reviewed_beatmaps - generated_beatmaps
        g_loss_expert = combined_model.train_on_batch([noise, reviewed_metadata], np.ones((num_reviewed_beatmaps, 1)), sample_weight=np.full((num_reviewed_beatmaps, 1), expert_weight))

        print(f"Epoch: {epoch}, Expert-reviewed Generator Loss: {g_loss_expert}")

    # Use early stopping to prevent overfitting
    val_loss = combined_model.evaluate(val_data, val_labels, verbose=0)
    early_stopping.on_epoch_end(epoch, logs={'val_loss': val_loss})
    if early_stopping.stop_training:
        print("Early stopping triggered. Stopping training.")
        break

if __name__ == '__main__':
    main()
