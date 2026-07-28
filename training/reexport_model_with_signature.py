import pathlib
import tensorflow as tf


class ModelWrapper(tf.Module):
    """
    A wrapper class for the Keras model to define a concrete serving signature.
    This is necessary to fix export issues where Keras fails to create a named
    input placeholder in the SavedModel's 'serving_default' signature.
    """

    def __init__(self, model):
        """
        Initializes the wrapper with the trained Keras model.
        Args:
            model: The trained tf.keras.Model instance.
        """
        self.model = model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 30, 126], dtype=tf.float32, name='input_landmarks')
    ])
    def __call__(self, input_landmarks):
        """
        Defines the forward pass with a specific input signature. This function
        will be traced by TensorFlow to create the graph with a named input.
        The output name will be automatically assigned by TensorFlow.
        """
        return self.model(input_landmarks)


def main():
    """
    Main function to load, wrap, and re-export the model.
    """
    # --- 1. Define Paths ---
    BASE_DIR = pathlib.Path(__file__).parent.parent
    # Path to the existing trained Keras model
    KERAS_MODEL_PATH = BASE_DIR / "training" / "models_checkpoints" / "best_model.keras"
    # Path for the new, correctly-signed SavedModel
    NEW_SAVED_MODEL_PATH = BASE_DIR / "models" / "asl_lstm_model_v2"

    print(f"Loading trained Keras model from: {KERAS_MODEL_PATH}")
    # --- 2. Load the existing Keras model ---
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    print("Model loaded successfully.")

    # --- 3. Wrap the model and save it with an explicit signature ---
    wrapped_model = ModelWrapper(model)
    print(f"\nSaving model with explicit signature to: {NEW_SAVED_MODEL_PATH}")
    tf.saved_model.save(
        wrapped_model,
        str(NEW_SAVED_MODEL_PATH),
        signatures={'serving_default': wrapped_model.__call__}
    )
    print("Model re-exported successfully.")

    # --- 4. Verify the new signature ---
    print("\nVerifying the signature of the newly saved model...")
    reloaded_model = tf.saved_model.load(str(NEW_SAVED_MODEL_PATH))
    print("Available signatures:", list(reloaded_model.signatures.keys()))
    print("\n'serving_default' signature details:")
    print(reloaded_model.signatures['serving_default'])
    print("\nVerification complete. Check the output above for 'input_landmarks'.")

if __name__ == "__main__":
    main()