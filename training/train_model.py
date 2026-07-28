import json
import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# --- Configuration ---

# Paths
from augment_landmarks import augment_sequence

BASE_DIR = pathlib.Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_CHECKPOINT_DIR = BASE_DIR / "training" / "models_checkpoints"
RESULTS_DIR = BASE_DIR / "training" / "results"
FINAL_MODEL_DIR = BASE_DIR / "models" / "asl_lstm_model"

# Model & Training Hyperparameters
SEQUENCE_LENGTH = 30
NUM_FEATURES = 126
NUM_EPOCHS = 250 # Increased epochs for smaller, regularized model
BATCH_SIZE = 4 # Reduced batch size given the small dataset
RANDOM_STATE = 42


def load_data(processed_dir):
    """
    Loads the pre-processed landmark data and labels.

    Args:
        processed_dir (pathlib.Path): Directory containing X.npy, y.npy, and labels.json.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The feature data (X).
            - np.ndarray: The one-hot encoded labels (y_one_hot).
            - list: A sorted list of unique string labels (class_names).
    """
    print("Loading dataset...")
    X = np.load(processed_dir / "X.npy")
    y_str = np.load(processed_dir / "y.npy")
    with open(processed_dir / "labels.json", "r") as f:
        label_map = json.load(f)

    # Invert the map to go from string label to integer index
    class_names = sorted(label_map.values())
    str_to_int_map = {name: i for i, name in enumerate(class_names)}

    # Encode string labels to integers
    y_int = np.array([str_to_int_map[label] for label in y_str])

    # One-hot encode the integer labels
    num_classes = len(class_names)
    y_one_hot = tf.keras.utils.to_categorical(y_int, num_classes=num_classes)

    print(f"Loaded {len(X)} samples for {num_classes} classes.")
    return X, y_one_hot, class_names


def print_class_sample_counts(y_int, class_names, title):
    """
    Counts occurrences of each class and prints a table, flagging low-sample classes.

    Args:
        y_int (np.ndarray): Array of integer labels for the dataset.
        class_names (list): List of string names corresponding to the integer labels.
        title (str): The title to print for the report.
    """
    unique_labels, counts = np.unique(y_int, return_counts=True)
    class_counts = dict(zip(unique_labels, counts))

    print(f"\n--- {title} ---")
    print(f"{'Class':<20} | {'Count':>5} | Status")
    print("-" * 37)
    for i, class_name in enumerate(class_names):
        count = class_counts.get(i, 0)
        status = "(LOW)" if count < 3 else ""
        print(f"{class_name:<20} | {count:>5} | {status}")
    print("-" * 37)


def build_model(num_classes):
    """
    Builds, compiles, and returns the LSTM model.

    Args:
        num_classes (int): The number of output classes.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    print("\nBuilding smaller, regularized LSTM model...")
    # This architecture is smaller and more regularized to combat overfitting on the small dataset.
    # Total params should be well under 100k.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),

        tf.keras.layers.LSTM(64, return_sequences=True, unroll=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.LSTM(32, return_sequences=False, unroll=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # Slightly lower learning rate
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.summary()
    return model


def train_model(model, X_train, y_train, X_val, y_val, checkpoint_path, class_weight_dict):
    """
    Trains the model with callbacks.

    Returns:
        tf.keras.callbacks.History: The training history object.
    """
    print("\nStarting model training...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), # Ensure filepath is a string
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    return history


def evaluate_model(model, history, X_test, y_test, class_names, results_dir):
    """
    Evaluates the model on the test set and saves results.
    """
    print("\nEvaluating model on the test set...")
    if X_test.shape[0] == 0:
        warnings.warn("WARNING: Test set is empty. Skipping evaluation.")
        print("\n--- Final Summary ---")
        print("No test data to evaluate.")
        return

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # --- Generate and Save Plots and Reports ---
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # --- Generate and Save Plots ---
    # Accuracy and Loss Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['categorical_accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_and_loss_curves.png")
    print(f"Saved accuracy/loss plot to {results_dir / 'accuracy_and_loss_curves.png'}")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png")
    print(f"Saved confusion matrix plot to {results_dir / 'confusion_matrix.png'}")
    plt.close()

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    print(f"Saved classification report to {results_dir / 'classification_report.txt'}")

    # --- Calculate Top-3 Accuracy ---
    top_3_acc = tf.keras.metrics.top_k_categorical_accuracy(y_test, y_pred_probs, k=3).numpy().mean()

    # --- Final Summary ---
    print("\n--- Final Summary ---")
    print(f"Overall Test Accuracy (Top-1): {accuracy:.2%}")
    print(f"Overall Test Accuracy (Top-3): {top_3_acc:.2%}")
    print("\nNOTE: Test set has only 1 sample per class — precision/recall per class should be interpreted cautiously given this sample size.")

    struggling_classes = []
    for class_name, metrics in report.items():
        if class_name in class_names: # Exclude 'accuracy', 'macro avg', etc.
            if metrics['precision'] < 0.5 or metrics['recall'] < 0.5:
                struggling_classes.append(f"- {class_name} (Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f})")

    if struggling_classes:
        print("\nClasses with Precision or Recall below 0.5:")
        for line in struggling_classes:
            print(line)
    else:
        print("\nAll classes achieved at least 0.5 precision and recall. Well done!")


def allocate_split_per_class(X, y_int, class_names, min_train=3, random_state=RANDOM_STATE):
    """
    Manually allocates samples to train, validation, and test sets on a per-class basis.
    This approach is used instead of proportional stratified splits (e.g., from sklearn)
    because with a small number of samples per class (e.g., 5-14 samples), proportional
    splits often fail to guarantee at least one sample per class in each split,
    leading to ValueError. This manual allocation ensures a valid split for each class
    based on predefined minimums.

    Args:
        X (np.ndarray): The feature data.
        y_int (np.ndarray): The integer labels.
        class_names (list): List of string names corresponding to the integer labels.
        min_train (int): Minimum number of samples to guarantee for training if possible.
                         (Not strictly enforced if class size is too small for test/val).
        random_state (int): Seed for shuffling for reproducibility.

    Returns:
        tuple: Contains lists of indices for (train, val, test) sets,
               and counts of classes with zero test/val samples.
    """
    print("\n--- Performing per-class manual data allocation ---")
    train_indices = []
    val_indices = []
    test_indices = []

    rng = np.random.RandomState(random_state)

    classes_with_zero_test = 0
    classes_with_zero_val = 0

    print(f"{'Class':<20} | {'Total':>5} | {'Train':>5} | {'Val':>5} | {'Test':>5}")
    print("-" * 55)

    for i, class_name in enumerate(class_names):
        class_sample_indices = np.where(y_int == i)[0]
        total_samples = len(class_sample_indices)

        # Shuffle indices for the current class
        rng.shuffle(class_sample_indices)

        current_train_indices = []
        current_val_indices = []
        current_test_indices = []

        # Allocate to test
        if total_samples >= 5:
            current_test_indices.append(class_sample_indices[0])
            remaining_indices = class_sample_indices[1:]
        else:
            classes_with_zero_test += 1
            remaining_indices = class_sample_indices

        # Allocate to validation
        # Check remaining after test allocation
        if len(remaining_indices) >= 4:
            current_val_indices.append(remaining_indices[0])
            current_train_indices.extend(remaining_indices[1:])
        else:
            classes_with_zero_val += 1
            current_train_indices.extend(remaining_indices) # All remaining go to train

        # Add to global lists
        train_indices.extend(current_train_indices)
        val_indices.extend(current_val_indices)
        test_indices.extend(current_test_indices)

        print(f"{class_name:<20} | {total_samples:>5} | {len(current_train_indices):>5} | {len(current_val_indices):>5} | {len(current_test_indices):>5}")

    print("-" * 55)
    print(f"Classes with 0 test samples: {classes_with_zero_test}")
    print(f"Classes with 0 validation samples: {classes_with_zero_val}")

    return train_indices, val_indices, test_indices, classes_with_zero_test, classes_with_zero_val


def split_data(X, y_one_hot, y_int, class_names, random_state=RANDOM_STATE):
    """
    Splits data into training, validation, and test sets using a per-class manual allocation strategy.
    This is necessary due to the small number of samples per class, which makes
    proportional stratified splits (e.g., from sklearn) prone to failure.

    Args:
        X (np.ndarray): The feature data.
        y_one_hot (np.ndarray): The one-hot encoded labels.
        y_int (np.ndarray): The integer labels.
        class_names (list): List of string names corresponding to the integer labels.
        random_state (int): Seed for shuffling for reproducibility.

    Returns:
        tuple: X_train_real, y_train_real_one_hot, X_val, y_val_one_hot, X_test, y_test_one_hot
    """
    print("\nSplitting data into training, validation, and test sets using manual per-class allocation...")

    train_indices, val_indices, test_indices, classes_with_zero_test, classes_with_zero_val = \
        allocate_split_per_class(X, y_int, class_names, random_state=random_state)

    # Convert lists of indices to numpy arrays
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    # Use indices to create the actual datasets
    X_train_real = X[train_indices]
    y_train_real_one_hot = y_one_hot[train_indices]

    X_val = X[val_indices]
    y_val_one_hot = y_one_hot[val_indices]

    X_test = X[test_indices]
    y_test_one_hot = y_one_hot[test_indices]

    print(f"\n--- Final Split Totals ---")
    print(f"Total Training Samples: {len(X_train_real)}")
    print(f"Total Validation Samples: {len(X_val)}")
    print(f"Total Test Samples: {len(X_test)}")
    print(f"Classes with 0 test samples: {classes_with_zero_test} out of {len(class_names)}")
    print(f"Classes with 0 validation samples: {classes_with_zero_val} out of {len(class_names)}")

    return X_train_real, y_train_real_one_hot, X_val, y_val_one_hot, X_test, y_test_one_hot


def build_augmented_training_set(X_train_real, y_train_real, num_augmentations_per_sample):
    """
    Builds an augmented training set by applying augment_sequence to each real training sample.

    Args:
        X_train_real (np.ndarray): The real (unaugmented) training feature data.
        y_train_real (np.ndarray): The real (unaugmented) training one-hot encoded labels.
        num_augmentations_per_sample (int): Number of augmented versions to generate for each real sample.

    Returns:
        tuple: X_train_augmented, y_train_augmented
    """
    print(f"\nBuilding augmented training set (generating {num_augmentations_per_sample} augmentations per real sample)...")
    if X_train_real.shape[0] == 0:
        print("  -> Real training set is empty, no augmentations will be generated.")
        # Ensure the returned arrays have the correct shape even if empty
        return np.array([]).reshape(0, SEQUENCE_LENGTH, NUM_FEATURES), np.array([]).reshape(0, y_train_real.shape[1])

    augmented_X = []
    augmented_y = []

    for i in tqdm(range(X_train_real.shape[0]), desc="Augmenting training data"):
        original_sequence = X_train_real[i]
        original_label = y_train_real[i]

        augmented_X.append(original_sequence) # Always include the original sample
        augmented_y.append(original_label)

        for _ in range(num_augmentations_per_sample):
            augmented_seq = augment_sequence(original_sequence)
            augmented_X.append(augmented_seq)
            augmented_y.append(original_label)

    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)

    print(f"  -> Augmented training set size: {augmented_X.shape[0]} samples.")
    return augmented_X, augmented_y


def main():
    """Main function to run the training pipeline."""
    # Create directories if they don't exist
    MODEL_CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    FINAL_MODEL_DIR.mkdir(exist_ok=True, parents=True)

    # 1. Load Data
    X, y_one_hot, class_names = load_data(PROCESSED_DATA_DIR)
    num_classes = len(class_names)
    y_int = np.argmax(y_one_hot, axis=1)

    # Print initial class sample counts
    print_class_sample_counts(y_int, class_names, title="Initial Dataset Class Sample Counts")

    # 2. Split Data
    # X_train_real, y_train_real are the *real* (unaugmented) training samples
    X_train_real, y_train_real, X_val, y_val, X_test, y_test = split_data(X, y_one_hot, y_int, class_names)

    # 3. Compute Class Weights for weighted loss
    # Based on the *real* (unaugmented) training data to counteract imbalance in the original dataset.
    from sklearn.utils import class_weight
    y_train_real_int = np.argmax(y_train_real, axis=1)
    class_weights_array = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train_real_int),
        y=y_train_real_int
    )
    class_weight_dict = dict(enumerate(class_weights_array))
    print("\nComputed class weights for training.")

    # 3. Build Augmented Training Set
    # Augmentation is applied ONLY to the training set to increase its size and variability,
    # helping to combat overfitting with limited real data. This function is defined in this file.
    NUM_AUGMENTATIONS_PER_SAMPLE = 6 # For each real training sample, generate 6 augmented versions
    X_train_augmented, y_train_augmented = build_augmented_training_set(X_train_real, y_train_real, NUM_AUGMENTATIONS_PER_SAMPLE)

    # 4. Build and Train Model
    model = build_model(num_classes)
    history = train_model(model, X_train_augmented, y_train_augmented, X_val, y_val, MODEL_CHECKPOINT_DIR / "best_model.keras", class_weight_dict)

    # 5. Evaluate Model and Save Results
    # Evaluation is performed on the real, unaugmented test set to get an unbiased
    # measure of the model's generalization performance.
    evaluate_model(model, history, X_test, y_test, class_names, RESULTS_DIR)

    # 6. Save Final Model
    print(f"\nSaving final model to {FINAL_MODEL_DIR}...")
    model.export(FINAL_MODEL_DIR)
    print("Model saved successfully in TensorFlow SavedModel format.")
    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()