# library used for creating web applications and interactive data dashboards with minimal effort.
import streamlit
import cv2
import numpy as np
from PIL import Image

streamlit.image('assets/neural-network.png')
streamlit.title('IMAGE CLASSIFICATION USING DNN');

# loading DNN model
model = cv2.dnn.readNet(
    model='./model/DenseNet_121.caffemodel',
    config='./model/DenseNet_121.prototxt',
    framework='Caffe'
)

# getting classes name
with open('./model/classification_classes_ILSVRC2012.txt', 'r') as file:
    imageNetClasses = file.read().split('\n')

# allow upload of image using streamlit
localFile = streamlit.file_uploader('Insert an image', type=['jpg', 'jpeg', 'png'])

if (localFile is not None):
    img = np.array(Image.open(localFile));
    # showing image
    streamlit.image(img)

    # first convert image to 3 channels, if necessary since dense net only accepts 3 channels
    height, width, channels = img.shape
    if (channels == 4):
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # modifying image to be accepted by the dnn model
    blob = cv2.dnn.blobFromImage(
        image = img,
        size=(224, 244),
        scalefactor=0.017, 
        mean=(104, 117, 123)
    )

    # sending model to be classified
    model.setInput(blob)
    outputs = model.forward()
    # getting logits
    logits = outputs[0].reshape(1000, 1)
    # getting result and reshape it to be used using argmax
    resultId = np.argmax(logits)
    # getting probability for all classes using softmax formula
    softmax = np.exp(logits) / np.sum(np.exp(logits))
    # getting the higher probability
    probability = softmax[resultId] * 100

    # showing result
    streamlit.markdown(
        f'<p>Class: {imageNetClasses[resultId]}, probability: {probability}</p>',
        unsafe_allow_html=True
    )



