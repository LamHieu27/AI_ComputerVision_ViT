import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from src.config import ProjectConfig

def get_densenet_keras_model(weights_path=None):
    """
    Reconstructs the Keras DenseNet121 model with custom head layers matching
    the exact structure of the provided pretrained_model.h5 weights.
    """
    if weights_path is None:
        weights_path = ProjectConfig.PRETRAINED_DENSENET_PATH
        
    print("Đang khởi tạo model DenseNet121 Baseline (TensorFlow/Keras)...")
    
    # Layer naming must match the weights inside pretrained_model.h5 exactly
    input_layer = Input(shape=(ProjectConfig.IMAGE_SIZE, ProjectConfig.IMAGE_SIZE, 3), name="input_2")
    
    base_model = tf.keras.applications.DenseNet121(
        include_top=False, 
        weights=None, 
        input_tensor=input_layer
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_average_pooling2d_2")(x)
    output_layer = Dense(ProjectConfig.NUM_CLASSES, activation="sigmoid", name="dense_2")(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"Đã tải thành công trọng số DenseNet121 từ: {weights_path}")
        except Exception as e:
            print(f"Lỗi khi tải trọng số DenseNet121: {e}. Tiến hành bỏ qua.")
    else:
        print(f"Cảnh báo: Không tìm thấy file trọng số tại {weights_path}.")
        
    return model

if __name__ == "__main__":
    # Test compiling and loading
    model = get_densenet_keras_model()
    print("Keras DenseNet121 baseline instantiated.")
