import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

DATA_PATH = "data"

train_dir = os.path.join(DATA_PATH, "train")
test_dir  = os.path.join(DATA_PATH, "test")


#image generator boyutlandırma    AUGMENT
train_gen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 10,        #10 derec döndür
    width_shift_range = 0.05,    #yatay kaydır
    height_shift_range = 0.05,   #dikey kaydır
    #shear_range = 0.2,          #kesme
    zoom_range = 0.1,           
    horizontal_flip = True,      #yaty çevirme
    #fill_mode = 'nearest'       #fill missing pixel 
    )

test_gen = ImageDataGenerator(rescale = 1./255)



train_data = train_gen.flow_from_directory(
    train_dir,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = "categorical"
)


test_data = test_gen.flow_from_directory(
    test_dir,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = "categorical",
    shuffle=False
)


#transfer learning MOBILENETV2 BASE
base_model = tf.keras.applications.MobileNetV2(
    input_shape = (224, 224, 3),
    include_top = False,
    weights = 'imagenet'

)

base_model.trainable = False #baslangiç


#CNN
model = models.Sequential([
    base_model,
    
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation = 'relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation = 'softmax')
    
    
    #layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (128, 128, 3)),
    #layers.MaxPooling2D(2, 2),

    #layers.Conv2D(64, (3,3), activation='relu'),
    #layers.MaxPooling2D(2,2),

    #layers.Conv2D(128, (3,3), activation='relu'),
    #layers.MaxPooling2D(2,2),

    #layers.Conv2D(256, (3,3), activation='relu'),
    #layers.MaxPooling2D(2,2),

    #layers.Conv2D(512, (3,3), activation='relu'),
    #layers.MaxPooling2D(2,2),

    #layers.Flatten(),
    #layers.Dropout(0.3),            #overfitting azaltma
    #layers.Dense(256, activation='relu'),
    #layers.Dense(12, activation='softmax')   #train_data.num_classes      kaç class varsa onu alacaktır
])

#overfit kontrol
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]



base_model.trainable = True
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)   #'adam'


model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss = 'categorical_crossentropy',
    metrics = ["accuracy",
               tf.keras.metrics.Precision(name="precision"),
               tf.keras.metrics.Recall(name="recall"),
    ]
)

#model eğitme
history = model.fit(train_data, validation_data = test_data, epochs = 15)


#fine tuning
#optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy',
               tf.keras.metrics.Precision(name="precision"),
               tf.keras.metrics.Recall(name="recall")
    ]
)

history = model.fit(train_data, validation_data=test_data, epochs=10, callbacks=callbacks)


MODEL_PATH = "models" # Klasör adını be0333lirle
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

model_save_path = os.path.join(MODEL_PATH, "food_model.h5")
model.save(model_save_path)
print(f"Model successfully saved to: {model_save_path}")



calorie_table = {
    "0": 442,  # class 0
    "1": 313,
    "2": 165,
    "3": 550,
    "4": 486,
    "5": 473,
    "6": 225,
    "7": 436,
    "8": 311,
    "9": 400,
    "10": 390,
    "11": 500
}






import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

model = load_model(os.path.join(MODEL_PATH, "food_model.h5"))


calorie_table = {
    "0": 180, "1": 250, "2": 320, "3": 400, "4": 220, "5": 520, "6": 450, "7": 310, "8": 150, "9": 600, 
    "10": 275, "11": 100
}

i=1
while(i<5):
    img_path = input("Please load a meal image's file path to calculate all calories:")
    #print(r"Sample file path: 'C:\Users\sevvl\Desktop\0.jpeg'")  #r for escape char
    #img_path = r"C:\Users\sevvl\Desktop\0.jpeg"

    try: 
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)

        class_idx = np.argmax(pred)
        calories = calorie_table[str(class_idx)]
        class_indices = train_data.class_indices
        inv_class_map = {v: k for k, v in class_indices.items()}
        class_name = inv_class_map[class_idx]


        print("Predicted Class ID:", class_idx+1)
        print("Predicted Class:", class_name)
        print("Predicted Calories:", calories)

    except FileNotFoundError:
        print(f"Error: '{img_path}' cannot found. Please check it.")  

    except Exception as e:
        print(f"Unexpected Error: {e}") 





