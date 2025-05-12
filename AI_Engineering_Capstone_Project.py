"""
medical_image_classifier.py

An improved medical image disease detector using transfer learning, tf.data pipelines,
proper splitting, augmentation, callbacks, and evaluation metrics.
"""

import os
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, applications, optimizers, metrics
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def set_seed(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def gather_paths(data_dir):
    """Collect .pt file paths and labels."""
    pos_dir = os.path.join(data_dir, "Positive_tensors")
    neg_dir = os.path.join(data_dir, "Negative_tensors")
    pos_paths = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if f.endswith('.pt')]
    neg_paths = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.endswith('.pt')]
    paths = pos_paths + neg_paths
    labels = [1] * len(pos_paths) + [0] * len(neg_paths)
    return paths, labels

def load_pt(path):
    """Load a .pt file and return a HxWxC float32 numpy array."""
    arr = torch.load(path.decode('utf-8')).numpy()
    # assume arr is CxHxW; transpose to HxWxC
    if arr.shape[0] <= 4:
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)

def preprocess(path, label, img_size):
    """TF wrapper to load and preprocess image, returns (image, label)."""
    img = tf.numpy_function(func=load_pt, inp=[path], Tout=tf.float32)
    img.set_shape([None, None, None])
    # resize to square
    img = tf.image.resize(img, [img_size, img_size])
    # standardize each image
    img = (img - tf.reduce_mean(img)) / (tf.math.reduce_std(img) + 1e-6)
    return img, label

def prepare_dataset(paths, labels, img_size, batch_size, training):
    """Create a tf.data.Dataset from file paths and labels."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.map(lambda p, l: preprocess(p, l, img_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        # on-the-fly augmentation
        aug = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip('horizontal'),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
            layers.experimental.preprocessing.RandomContrast(0.1),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(img_size, base_lr):
    """Build a transfer-learning model for binary classification."""
    inputs = layers.Input(shape=(img_size, img_size, 3))
    # Assume grayscale if single-channel: repeat to 3 channels
    def to_rgb(x):
        if tf.shape(x)[-1] == 1:
            return tf.image.grayscale_to_rgb(x)
        return x
    x = layers.Lambda(to_rgb)(inputs)
    # data augmentation can be included here if desired
    base = applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        pooling=None
    )
    base.trainable = False  # freeze pretrained weights
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=base_lr),
        loss='binary_crossentropy',
        metrics=[
            metrics.BinaryAccuracy(name='accuracy'),
            metrics.AUC(name='auc'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')
        ]
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="Medical Image Disease Detector")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Root directory with Positive_tensors/ and Negative_tensors/")
    parser.add_argument('--img_size', type=int, default=224,
                        help="Image height and width after resizing")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Maximum number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Base learning rate")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--test_size', type=float, default=0.1,
                        help="Fraction of data to reserve for final testing")
    parser.add_argument('--val_size', type=float, default=0.1,
                        help="Fraction of data to reserve for validation")
    parser.add_argument('--model_out', type=str, default='best_model.h5',
                        help="Path to save the best model weights")
    parser.add_argument('--log_dir', type=str, default='logs',
                        help="TensorBoard log directory")
    args = parser.parse_args()

    set_seed(args.seed)

    # gather and split data
    paths, labels = gather_paths(args.data_dir)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=args.test_size, stratify=labels, random_state=args.seed
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=args.val_size / (1 - args.test_size),
        stratify=train_labels, random_state=args.seed
    )

    # prepare datasets
    train_ds = prepare_dataset(train_paths, train_labels, args.img_size,
                               args.batch_size, training=True)
    val_ds   = prepare_dataset(val_paths,   val_labels,   args.img_size,
                               args.batch_size, training=False)
    test_ds  = prepare_dataset(test_paths,  test_labels,  args.img_size,
                               args.batch_size, training=False)

    # build and train model
    model = build_model(args.img_size, args.lr)
    cb_early = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    cb_reduce = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
    )
    cb_ckpt = callbacks.ModelCheckpoint(
        args.model_out, save_best_only=True, monitor='val_accuracy'
    )
    cb_tb = callbacks.TensorBoard(log_dir=args.log_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[cb_early, cb_reduce, cb_ckpt, cb_tb]
    )

    # evaluation
    print("\nEvaluating on test set:")
    results = model.evaluate(test_ds, return_dict=True)
    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    # detailed classification report
    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    y_pred_prob = model.predict(test_ds)
    y_pred = (y_pred_prob.ravel() > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
