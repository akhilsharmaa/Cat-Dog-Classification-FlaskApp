# Cat-Dog Classification FlaskApp

<!--- <img alt="Screenshot 2023-03-26 at 4 48 50 PM" height='300px' src="https://user-images.githubusercontent.com/74103314/227773997-88fe9d13-8dfb-4d34-979a-b4abc6b785b1.png"> -->

Cat-Dog Classification **CNN**(Convolutional Neural Networks) Flask web-app. This model takes input as format of **256 x 256 x 3** and output as a sigmoid function. 

**Dataset** : [kaggle.com/competitions/dogs-vs-cats/](https://www.kaggle.com/competitions/dogs-vs-cats/data)

**Video-demo**: [youtu.be/0w5E9YOj_Qg](https://youtu.be/0w5E9YOj_Qg)

### Model ARCHITECTURE 
```
#initializing the cnn
model = Sequential()

model.add(Convolution2D(64,3,3,input_shape=(264,264,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()
```

**Model accuracy on validation data : 0.77**, this accuracy is low because this the very simple achitecture. Accuracy can be improved by tuning the hyperparameters of the network.
