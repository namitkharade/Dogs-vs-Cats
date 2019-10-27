import cv2
from keras.models import load_model
from keras import backend as K

def swish_activation(x):
    return (K.sigmoid(x) * x)

def load_cv2_image(fileName):
        return cv2.resize(cv2.imread(fileName , cv2.IMREAD_GRAYSCALE), (50, 50)).reshape(1, 1, 50, 50)

def predict_animal(fileName):
        test_image = load_cv2_image(fileName)

        model = load_model("../Model/Cats_and_Dogs.h5", custom_objects = {"swish_activation": swish_activation})
        prediction_result = model.predict([test_image])
        result_animal = "Cat" if prediction_result[0][0] > prediction_result[0][1] else "Dog"
        
        msg = "I think the animal is a {}".format(result_animal)
        print(msg)
        return msg
