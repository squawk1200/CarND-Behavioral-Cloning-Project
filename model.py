import base64
import json
import numpy as np
import cv2

from PIL import Image
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import Adam

DATA_DIR                = "data"
DRIVING_LOG_FILE        = "driving_log.csv"
WEIGHTS_FILE            = "model.h5"
MODEL_FILE              = "model.json"
BATCH_SIZE              = 128
SAMPLES_PER_EPOCH       = 20000
NUMBER_OF_EPOCHS        = 5
STEERING_OFFSET         = 0.25
ORIG_ROW_SIZE           = 160
ORIG_COL_SIZE           = 320
CH                      = 3


def load_driving_log(path_to_file):
    data = np.genfromtxt(path_to_file, dtype=None, delimiter=',', names=True)
    
    array_data = []
    
    for row in data:
        array_row = [row[0], row[1], row[2], row[3], row[4], row[5], row[6]]
        array_data.append(array_row)

    return np.asarray(array_data)

# From Vivek Yadav post
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.8btw7h59y
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

# From Vivek Yadav post
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.8btw7h59y
def trans_image(image, steer, trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x], [0,1,tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (ORIG_COL_SIZE, ORIG_ROW_SIZE))
    
    return image_tr,steer_ang


def process_image(filename, steering_angle):
    image = Image.open(filename)
    image = np.asarray(image)
    # Add random brightness
    image = augment_brightness_camera_images(image)
    # Add random translation (horizontal & vertical)
    image, steering_angle = trans_image(image, steering_angle, 100)
    
    # Flip half of the images horizontally
    if (np.random.randint(2) == 0):
        image = np.fliplr(image)
        steering_angle = -steering_angle

    return image, steering_angle

def augment_image(line_item):

    random_select_3 = np.random.randint(3)
    random_select_2 = np.random.randint(2)
    
    steering_angle = line_item[3].astype(float)
    steering_offset = 0
    
    # Discard steering angle 0 data with 50% probability
    if ((steering_angle == 0.0) and (random_select_2 == 0)):
        return None, None
    
    # Pick image from one of the 3 cameras
    if (random_select_3 == 0):
        # Pick center image
        filename = DATA_DIR + "/" + line_item[0].decode('UTF-8').strip()
        steering_offset = 0.0
    elif (random_select_3 == 1):
        # Pick left image
        filename = DATA_DIR + "/" + line_item[1].decode('UTF-8').strip()
        # Bias the car towards the center by adding a positive offset
        steering_offset = STEERING_OFFSET
    else:
        # Pick right image
        filename = DATA_DIR + "/" + line_item[2].decode('UTF-8').strip()
        # Bias the car towards the center by adding a negative offset
        steering_offset = -STEERING_OFFSET

    steering_angle += steering_offset
    image, steering_angle = process_image(filename, steering_angle)

    return image, steering_angle


def batch_generator(driving_log, batch_size=32):
    batch_images = np.zeros((batch_size, ORIG_ROW_SIZE, ORIG_COL_SIZE, CH))
    batch_steering_angles = np.zeros(batch_size)
    numberOfLogEntries = len(driving_log) - 1
    
    while 1:
        for i in range(batch_size):
            i_line = np.random.randint(numberOfLogEntries)
            line_item = driving_log[i_line]
        
            x, y = augment_image(line_item)
            
            while (x is None):
                i_line = np.random.randint(numberOfLogEntries)
                line_item = driving_log[i_line]
                x, y = augment_image(line_item)
        
            batch_images[i] = x
            batch_steering_angles[i] = y

        yield batch_images, batch_steering_angles


# Nvidia pipeline
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def get_nvidia_model():
    model = Sequential()
    
    # NVIDIA pipeline
    
    # 1. Input 160x320x3
    
    # 2. Normalization
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(ORIG_ROW_SIZE, ORIG_COL_SIZE, CH),
                     output_shape=(ORIG_ROW_SIZE, ORIG_COL_SIZE, CH)))
    print("Normalization layer: ", model.layers[-1].output_shape)
    
    # Crop 25 pixels from the top of the image
    model.add(Cropping2D(cropping=((35, 1), (1, 1)),
                         input_shape=(ORIG_ROW_SIZE, ORIG_COL_SIZE, CH),
                         dim_ordering='tf'))
    print("Cropping layer 1: ", model.layers[-1].output_shape)
    
    # 3. Convolutional layer 1 5x5 kernel 2x2 stride
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Activation('relu'))
    print("Convolution layer 1: ", model.layers[-1].output_shape)
    
    # 4. Convolutional layer 2 5x5 kernel 2x2 stride
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Activation('relu'))
    print("Convolution layer 2: ", model.layers[-1].output_shape)
    
    # 5. Convolutional layer 3 5x5 kernel 2x2 stride
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Activation('relu'))
    print("Convolution layer 3: ", model.layers[-1].output_shape)
    
    # 6. Convolutional layer 4 3x3 kernel
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    print("Convolution layer 4: ", model.layers[-1].output_shape)
    
    # 7. Convolutional layer 5 3x3 kernel 3x12x64
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    print("Convolution layer 5: ", model.layers[-1].output_shape)
    
    # 8. Flatten
    model.add(Flatten())
    print("Flatten: ", model.layers[-1].output_shape)
    
    # 9. Fully connected layer 1 - 100 neurons
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Activation('relu'))
    print("Fully connected layer 1: ", model.layers[-1].output_shape)
    
    # 10. Fully connected layer 2 - 50 neurons
    model.add(Dense(50))
    model.add(Activation('relu'))
    print("Fully connected layer 2: ", model.layers[-1].output_shape)
    
    # 11. Fully connected layer 3 - 10 neurons
    model.add(Dense(10))
    model.add(Activation('relu'))
    print("Fully connected layer 3: ", model.layers[-1].output_shape)
    
    # 12. Output
    model.add(Dense(1))

    model.compile(optimizer=Adam(), loss="mse")
    
    model.summary()

    return model


if __name__ == "__main__":

    # Load driving log
    driving_log = load_driving_log(DATA_DIR + "/" + DRIVING_LOG_FILE)

    # Split into training and validation sets
    msk = np.random.rand(len(driving_log))  < 0.8
    train = driving_log[msk]
    val = driving_log[~msk]
    
    model = get_nvidia_model()
 
    # Use a generator - for both training and validation data
    history = model.fit_generator(batch_generator(train, batch_size=BATCH_SIZE),
                                  samples_per_epoch=SAMPLES_PER_EPOCH,
                                  nb_epoch=NUMBER_OF_EPOCHS,
                                  validation_data=batch_generator(val, batch_size=BATCH_SIZE),
                                  nb_val_samples=len(val))

    # Save weights
    model.save_weights(WEIGHTS_FILE)
                                            
    # Save model
    with open(MODEL_FILE, 'w') as outfile:
        json.dump(model.to_json(), outfile)
