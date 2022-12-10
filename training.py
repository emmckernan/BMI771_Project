# ************* Feature extraction *************

##### https://github.com/lukemelas/EfficientNet-PyTorch#example-classification
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import os

data_folder = r"./data/TCGA-A2-A25B-01Z-00-DX1.58D7BEDE-5558-4A9E-A95E-DDF24C9267EF.svs"

#for file in os.listdir(data_folder):

# model = EfficientNet.from_pretrained('efficientnet-b0')

# img_path = data_folder + "/4255_48922_2127_2100.png"

# tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
# img = tfms(Image.open(img_path).convert('RGB')).unsqueeze(0)

# # ... image preprocessing as in the classification example ...
# print(img.shape) # torch.Size([1, 3, 224, 224])

# features = model.extract_features(img)
# print(features.shape) # torch.Size([1, 1280, 7, 7])

# #************* Training *************

##pre-trained breast cancer model from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5967747/

from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

IMG_SIZE = 224
NUM_CLASSES = 3 # NUM_CLASSES = ds_info.features["label"].num_classes

batch_size = 64

ds_train =  r"./pretrained"
ds_test = data_folder

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


with strategy.scope():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

model.summary()

epochs = 40  # @param {type: "slider", min:10, max:100}
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)
