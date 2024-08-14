import os
import zipfile
import cv2 
import numpy as np
import tensorflow as tf
import cv2

from keras.backend import clear_session
from keras.models import Sequential
from keras import layers, Model
from sklearn.model_selection import train_test_split
from keras import Model
from keras.layers import Conv2D, PReLU,BatchNormalization, Flatten
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime


#code for building generator


def res_block(input_dim):
    model = Conv2D(64, (3,3), padding = 'same' )(input_dim)
    model = BatchNormalization()(model)
    model = PReLU(shared_axes = [1,2])(model)
    model = Conv2D(64, (3,3), padding = 'same' )(model)
    model = BatchNormalization()(model)
    return add([input_dim, model])
def upscale_block(input_dim):
    model = Conv2D(256,(3,3), strides=1, padding = 'same')(input_dim)
    model = UpSampling2D(size = (2,2))(model)
    model = PReLU(shared_axes=[1, 2])(model)
    return model
def generator_model(input, res_range = 1,upscale_range=1):
    model = Conv2D(64,(9,9), strides=1, padding = 'same')(input)
    model = PReLU(shared_axes = [1,2])(model)
    model1 = model
    for i in range(res_range):
        model = res_block(model)
    model = Conv2D(64, (3,3), padding = 'same' )(model)
    model = BatchNormalization()(model)
    model = add([model,model1])
    for i in range(upscale_range):
        model  =upscale_block(model)
    output = Conv2D(3, (9,9),  padding='same')(model)
    return Model(input, output)


#code for building discriminator
def discrim_block(input_dim, fmaps = 64, strides = 1):
    model = Conv2D(fmaps, (3,3), padding = 'same', strides  = strides)(input_dim)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    return model
def discriminator_model(input):
    model = Conv2D(64,(3,3),padding='same')(input)
    model = LeakyReLU()(model)
    model = discrim_block(model, strides = 2)
    model = discrim_block(model, fmaps  = 128)
    model = discrim_block(model, fmaps = 128, strides = 2)
    model = discrim_block(model, fmaps=256)
    model = discrim_block(model, fmaps=256, strides=2)
    model = discrim_block(model, fmaps=512)
    model = discrim_block(model, fmaps=512, strides=2)
    model = Flatten()(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    out = Dense(1, activation='sigmoid')(model)
    return Model(input, out)

#introducing vgg19 layer
from tensorflow.keras.applications.vgg19 import VGG19
def build_vgg(hr_shape):
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)

    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)


# Combined model
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)

    gen_features = vgg(gen_img)

    disc_model.trainable = False
    validity = disc_model(gen_img)

    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])


def create_training_data():
    for img in tqdm(list(os.listdir(datadir))):  # iterate over each image per dogs and cats
        try:
            img_array = cv2.imread(datadir+'/'+img ,cv2.IMREAD_COLOR)  # convert to array
            new_array = cv2.resize(img_array, (128, 128))  # resize to normalize data size
            array.append([new_array]) 
            array_small.append([cv2.resize(img_array, (64,64),
                            interpolation=cv2.INTER_AREA)]) # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass


def create_model():

    hr_shape = (y_train.shape[1], y_train.shape[2], y_train.shape[3])
    lr_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    lr_ip = Input(shape=lr_shape)
    hr_ip = Input(shape=hr_shape)

    generator = generator_model(lr_ip, res_range = 16, upscale_range=1)
    # generator.summary()

    discriminator = discriminator_model(hr_ip)
    discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    # discriminator.summary()

    vgg = build_vgg((128,128,3))
    
    vgg.trainable = False

    gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)

    gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")

    return generator, discriminator, gan_model, vgg


def data_generator(X, y, batch_size):
    while True:
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            yield X[start:end], y[start:end]


def data_generator_tf(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def train_model(X_train, y_train, generator, discriminator, gan_model, vgg):
    
    batch_size = 8
    epochs = 10
    steps_per_epoch = len(X_train) // batch_size

    # Initialize the data generator
    #train_gen = data_generator(X_train, y_train, batch_size)
    train_dataset = data_generator(X_train, y_train, batch_size)

    #Enumerate training over epochs
    for e in range(epochs):
        print(f"----------EPOCH {e}----------")
        fake_label = np.zeros((batch_size, 1)) # Assign a label of 0 to all fake (generated images)
        real_label = np.ones((batch_size, 1)) # Assign a label of 1 to all real images.
        
        #Create empty lists to populate gen and disc losses. 
        g_losses = []
        d_losses = []
        
        #Enumerate training over batches. 
        #for b in range(steps_per_epoch): #tqdm(range(len(train_hr_batches))):
        for lr_imgs, hr_imgs in train_dataset:

            #lr_imgs, hr_imgs = next(train_gen)  # Get the next batch from the generator
            
            fake_imgs = generator.predict_on_batch(lr_imgs) #Fake images
            
            #First, train the discriminator on fake and real HR images. 
            discriminator.trainable = True
            d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
            d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)
            
            #Now, train the generator by fixing discriminator as non-trainable
            discriminator.trainable = False
            
            #Average the discriminator loss, just for reporting purposes. 
            d_loss = 0.5 * np.add(d_loss_gen, d_loss_real) 
            
            #Extract VGG features, to be used towards calculating loss
            image_features = vgg.predict(hr_imgs)
        
            #Train the generator via GAN. 
            #Remember that we have 2 losses, adversarial loss and content (VGG) loss
            g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])
            
            #Save losses to a list so we can average and report. 
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            
        #Convert the list of losses to an array to make it easy to average    
        g_losses = np.array(g_losses)
        d_losses = np.array(d_losses)
        
        #Calculate the average losses for generator and discriminator
        g_loss = np.sum(g_losses, axis=0) / len(g_losses)
        d_loss = np.sum(d_losses, axis=0) / len(d_losses)
        
        clear_session()
        #Report the progress during training. 
        print("epoch:", e+1 ,"g_loss:", g_loss, "d_loss:", d_loss)

        if (e+1) % 1 == 0: #Change the frequency for model saving, if needed
            #Save the generator after every n epochs (Usually 10 epochs)
            generator.save(f"{print_time()}_gen_e_" + str(e+1) + ".h5")


def print_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    return formatted_time   


def extract_dataset():
    print("Extracting images...")
    with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
        zip_ref.extractall('.')


def augment_dataset(datadir, X):
    #augmenting the data
    #this generator will save files in a physical format
    print("Augmenting images...")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(        
            rotation_range = 40,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            brightness_range = (0.5, 1.5))

    os.mkdir(datadir)

    for a in X:
        i = 0
        a = a.reshape((1, ) + a.shape)
        for batch in datagen.flow(a, batch_size=1,  save_to_dir= 'Augmented_images_64', save_prefix='dr', save_format='jpeg'):    
            try:
                i += 1   
                if i>= 10:
                    break 
            except Exception:
                print("error")
                pass


if __name__=='__main__':

    datadir = 'qr_dataset'

    currentdir = os.getcwd()
    if not (os.path.exists(os.path.join(currentdir, datadir)) and os.path.isdir(os.path.join(currentdir, datadir))):
        extract_dataset()

    array = []
    array_small = []

    create_training_data()

    X = []
    Xs = []
    for features in array:
        X.append(features)
    for features in array_small:
        Xs.append(features)
    X = np.array(X).reshape(-1, 128, 128, 3)
    Xs = np.array(Xs).reshape(-1, 64, 64, 3)

    array=[]
    array_small=[]
    datadir = 'Augmented_images_64'

    if not (os.path.exists(os.path.join(currentdir, datadir)) and os.path.isdir(os.path.join(currentdir, datadir))):
        augment_dataset(datadir, X)

    create_training_data()

    X1 =  []
    Xs1 = []
    for features in array:
        X1.append(features)
    for features in array_small:
        Xs1.append(features)
    X1 = np.array(X1).reshape(-1, 128, 128, 3)
    Xs1 = np.array(Xs1).reshape(-1, 64, 64, 3)

    X=np.concatenate((X,X1), axis = 0)
    Xs=np.concatenate((Xs,Xs1), axis=0)

    X_train, X_valid, y_train, y_valid = train_test_split(Xs, X, test_size = 0.33, random_state = 42)
    X_train, X_valid, y_train, y_valid = X_train[::2, ...], X_valid[::2, ...], y_train[::2, ...], y_valid[::2, ...]

    generator, discriminator, gan_model, vgg = create_model()

    train_model(X_train, y_train, generator, discriminator, gan_model, vgg)
