#TensorFlow Keras is an implementation of the Keras API that uses TensorFlow as a backend.
from keras.applications import EfficientNetB7
from keras.applications import EfficientNetB0
import cv2
import numpy as np
from matplotlib.pyplot import imshow
from matplotlib.pyplot import imread
from keras.applications.imagenet_utils import preprocess_input, decode_predictions



# here we are using per-trained model of EfficientNetB0 for images. which is imagenet and it is trained on 1000 classes.
model = EfficientNetB7(weights='imagenet')
print(model.summary())

img_path = 'b.jpg'
# for EfficientNetB7
img_size = 600

img = cv2.imread(img_path)

img = cv2.resize(img, (img_size, img_size))

# expanding the dimension of image for predictions.
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)

# displaying the image using matplotlib
m_img = imread(img_path)
imshow(m_img)

# making prediction using the model. In output it will give the probability of each 1000 classes.
pred = model.predict(x)
print(pred)
# Here we decoding the predictions and taking only the high value predictions classes.
result = decode_predictions(pred, top=3)

print(result)



