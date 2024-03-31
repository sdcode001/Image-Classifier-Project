import os
import cv2
import numpy as np
from keras import models
from keras.applications.imagenet_utils import preprocess_input


def get_decoded_predictions(preds, label_classes, top=0):
    result = []
    for i in range(len(label_classes)):
        result.append((preds[i], label_classes[i]))
    result = list(result)
    result.sort(reverse=True)
    if top!=0:
        first = result[0:top]
        return first
    return result


# Testing our trained model on unseen data....

# For EfficientNetB0 Model
img_size = 224
test_img_path = 'bed-1303451__340.jpg'

img = cv2.imread(test_img_path)
img = cv2.resize(img, (img_size, img_size))
# expanding the dimension of image for predictions.
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)
# making prediction using the model
model = models.load_model('trained_model.h5')
pred = model.predict(x)
print(pred)

# Here we decoding the predictions and taking only the high value predictions classes.
label_classes = os.listdir('dataset')
results = get_decoded_predictions(preds=pred[0], label_classes=label_classes, top=1)
print(results)


