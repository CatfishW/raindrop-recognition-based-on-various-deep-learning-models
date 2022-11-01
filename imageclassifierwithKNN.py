from preprocessing import SimplePreprocessor 
from datasets import SimpleDatasetLoader
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# 全局变量
train_data = 'images/train'
test_data =  'images/test'

# 全局常量
N_NEIGHBOURS = 5
TARGET_IMAGE_WIDTH = 32
TARGET_IMAGE_HEIGHT = 32

# initialize the image preprocessor and datasetloader
sp = SimplePreprocessor(TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT)
sdl = SimpleDatasetLoader(preprocessors=[sp])

# Load images
print("[INFO] loading images...")

train_image_paths = list(paths.list_images(train_data)) # path included
test_image_paths = list(paths.list_images(test_data))

(X_train, y_train) = sdl.load(train_image_paths, verbose=500, grayscale = True)
(X_test, y_test) = sdl.load(test_image_paths, verbose=500, grayscale = True)

# Flatten (reshape the data matrix)
X_train = X_train.reshape((X_train.shape[0], TARGET_IMAGE_WIDTH*TARGET_IMAGE_HEIGHT)) 
X_test = X_test.reshape((X_test.shape[0], TARGET_IMAGE_WIDTH*TARGET_IMAGE_HEIGHT)) 

# Show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(X_train.nbytes / (1024 * 1024.0)))

# Label encoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# 第二部分：训练模型

# Train model
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors= N_NEIGHBOURS, metric = 'minkowski', p = 2)
model.fit(X_train, y_train)


# 第三部分：评估模型

# Evaluate model
y_pred = model.predict(X_test)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Report
print(classification_report(y_test, y_pred, target_names=le.classes_))