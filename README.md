# PNEUMONIA DETECTION USING TRANSFER LEARNING
#OBJECTIVE OF THE STUDY
 The objective of the study is to develop an accurate deep learning model to detect pneumonia using chest X-ray images.
 This involves applying preprocessing techniques, building a transfer learning models to get accurate result and reduce diagnostic errors, supporting earlier disease detection.
 #Preprocessing Techniques
 1. Morphological Image Processing
Performed image enhancement techniques to improve pneumonia detection:
Dilation: Expands bright regions, enhancing lung opacity in pneumonia-affected X-rays.
Erosion: Reduces noise and removes small white spots in normal X-rays.
Gaussian Blur: Reduces image noise while preserving key features.
Canny Edge Detection: Extracts key lung region edges for better feature detection.
2. Data Normalization & Resizing
Pixel Normalization: Scaled pixel values to [0,1] range to improve model training.
Resizing: All images resized to 224×224 to match CNN input requirements.
3. Data Augmentation (Before CNN Training)
Applied augmentation techniques only on the training set to improve generalization:
Horizontal & Vertical Flip → Handles different orientations.
Rotation (0.3°) → Increases diversity.
ZCA Whitening → Highlights key patterns.
Width & Height Shifts (0.25) → Adjusts for X-ray misalignment.
Channel Shift (0.35) → Handles intensity variations.
Shear (0.2) & Zoom (0.4) → Introduces small distortions for better robustness.
#Model Architecture (Custom CNN)
Why Custom CNN?
Designed from scratch to extract relevant patterns from pneumonia images.
No reliance on pretrained weights, allowing the model to fully adapt to the dataset.
Helps establish a baseline accuracy before using Transfer Learning.
Model Layers:
Conv2D (32 filters, 3×3 kernel, ReLU, Same Padding) + MaxPooling
Conv2D (64 filters, 3×3 kernel, ReLU, Same Padding) + MaxPooling
Dropout (0.2) to reduce overfitting
Conv2D (128 filters, 3×3 kernel, ReLU, Same Padding) + MaxPooling
Conv2D (256 filters, 3×3 kernel, ReLU, Same Padding) + MaxPooling
Dropout (0.2) to improve generalization
Flatten layer to convert feature maps into a single vector
Dense (128, 64, 32) Fully Connected Layers with ReLU activation
Output Layer (Sigmoid for binary classification: Pneumonia vs. Normal)
Results from Custom CNN:
Achieved 84% accuracy on the test set.
# Transfer Learning with VGG16
mplementation of VGG16 for Pneumonia Detection
 Step 1: Load the Pretrained VGG16 Model
VGG16 is loaded without the fully connected (top) layers (include_top=False).
We use pretrained weights from ImageNet to leverage its learned features.
Input shape is set to (224, 224, 3) to match our image dataset.
 Step 2: Add Custom Fully Connected Layers for Classification
Flatten Layer → Converts feature maps into a 1D vector.
Dense(128, 64, 32) layers (ReLU activation) → Extracts high-level patterns.
Output Layer (Sigmoid activation) → Classifies images as Pneumonia (1) or Normal (0).
 Step 3: Freeze the VGG16 Layers
The pretrained convolutional layers are frozen (i.e., trainable = False), so only the new layers learn from our dataset.
This ensures we retain the powerful feature extraction capability of VGG16 without modifying its learned filters.
 Step 4: Compile and Train the Model
Optimizer: Adam (efficient for deep learning tasks).
Loss Function: Binary Cross-Entropy (since we have two classes: Normal & Pneumonia).
Evaluation Metric: Accuracy (to measure classification performance).
# Transfer Learning with VGG19
Implementation of VGG19 for Pneumonia Detection
Step 1: Load the Pretrained VGG19 Model
 The VGG19 model is loaded without the top (fully connected) layers by setting include_top=False.
 We use pretrained weights from the ImageNet dataset to utilize the rich feature extraction capabilities learned from millions of images.
 The input shape is defined as (224, 224, 3) to match the dimensions of our chest X-ray images.
Step 2: Add Custom Fully Connected Layers for Classification
 A Flatten layer is added to convert the output feature maps from VGG19 into a 1D vector.
 Three fully connected (Dense) layers with 128, 64, and 32 neurons respectively, all using the ReLU activation function, are stacked to learn high-level, abstract representations of the input data.
 A final Dense layer with 1 neuron and a sigmoid activation function is used to perform binary classification, distinguishing between Pneumonia (1) and Normal (0) images.
Step 3: Freeze the VGG19 Layers
 All layers in the VGG19 base model are frozen by setting trainable=False.
 This prevents the pretrained convolutional layers from being updated during training, preserving the powerful and generalizable visual features they have already learned.
 Only the newly added fully connected layers will be trained on our specific dataset.
Step 4: Compile and Train the Model
 The model is compiled using the Adam optimizer, which adapts learning rates during training and works well for deep learning tasks.
 The Binary Cross-Entropy loss function is chosen as we are dealing with a binary classification problem.
Accuracy is used as the evaluation metric to monitor the model's performance in predicting Pneumonia or Normal cases correctly during training and validation.
# Transfer Learning with InceptionResNetV2
mplementation of InceptionResNetV2 for Pneumonia Detection
Step 1: Load the Pretrained InceptionResNetV2 Model
 The InceptionResNetV2 model is loaded with include_top=False to exclude the original classification head.
 It uses pretrained weights from the ImageNet dataset, which allows the model to start with a solid foundation of feature extraction.
 The input shape is set to (224, 224, 3) to match the dimensions of chest X-ray images used in the dataset.
 Global max pooling is applied to reduce the spatial dimensions before feeding into the dense layers.
Step 2: Add Custom Fully Connected Layers for Classification
 A Flatten layer is used to convert the output of the convolutional base into a 1D vector.
 This is followed by fully connected layers with 128, 64, and 32 neurons respectively, each using the ReLU activation function to capture complex patterns.
 Dropout layers (with a rate of 0.3) are added after the first two dense layers to reduce overfitting by randomly deactivating neurons during training.
 A final dense layer with a single neuron and sigmoid activation is used for binary classification (Pneumonia or Normal).
Step 3: Freeze the InceptionResNetV2 Layers
 All layers in the base model are frozen by setting trainable=False.
 This ensures that the learned convolutional filters from ImageNet are preserved and only the added dense layers are trained on our specific pneumonia dataset.
Step 4: Compile and Train the Model
 The model is compiled using the Adam optimizer, which adapts the learning rate and is effective for deep learning.
 The binary cross-entropy loss function is used since this is a two-class classification task.
Accuracy is chosen as the performance metric to evaluate how well the model distinguishes between Pneumonia and Normal images.
# Transfer Learning with InceptionV3
Implementation of InceptionV3 for Pneumonia Detection
Step 1: Load the Pretrained InceptionV3 Model
 The InceptionV3 model is loaded with include_top=False, meaning the fully connected layers used for ImageNet classification are excluded.
 Pretrained weights from ImageNet are utilized to benefit from the rich feature representations learned on a large and diverse dataset.
 The input shape is set to (224, 224, 3), compatible with typical chest X-ray image preprocessing.
pooling="max" applies global max pooling to summarize the spatial features into a single vector per feature map.
Step 2: Add Custom Fully Connected Layers for Pneumonia Classification
 A Flatten layer is used to reshape the pooled output into a one-dimensional vector.
 This is followed by fully connected layers with 128, 64, and 32 neurons respectively, each activated using ReLU to introduce non-linearity and enable learning of complex patterns.
 Two Dropout layers with a rate of 0.3 are included after the first two dense layers to prevent overfitting by randomly disabling neurons during training.
 The final output layer uses a single neuron with a sigmoid activation function to output a probability for binary classification: Pneumonia (1) or Normal (0).
Step 3: Freeze the InceptionV3 Layers
 All layers in the base InceptionV3 model are frozen by setting trainable=False, so their pretrained weights remain unchanged.
 This ensures that the powerful feature extraction capabilities of InceptionV3 are retained, while only the new classification head is trained on the pneumonia dataset.
Step 4: Compile and Train the Model
 The model is compiled using the Adam optimizer, known for its adaptive learning capabilities and efficiency in deep learning tasks.
 The binary cross-entropy loss function is used as the task involves two classes.
Accuracy is set as the evaluation metric to monitor the model’s performance during training and validation.
# Transfer Learning with ResNet50V2
Implementation of ResNet50V2 for Pneumonia Detection
Step 1: Load the Pretrained ResNet50V2 Model
 The ResNet50V2 model is loaded with include_top=False, which removes the final classification layers that were originally trained on ImageNet.
 Pretrained ImageNet weights are used to take advantage of the deep residual learning features developed from a vast image dataset.
 The input shape is set to (224, 224, 3) to match standard image preprocessing requirements, and pooling="max" applies global max pooling, compressing feature maps into a single vector per map.
Step 2: Add Custom Fully Connected Layers for Classification
 The output from the base model is passed through a Flatten layer, transforming the pooled features into a 1D vector.
 This is followed by three dense layers with 128, 64, and 32 units respectively, each activated by ReLU to learn abstract and high-level patterns specific to pneumonia classification.
 The final layer is a single-node dense layer with a sigmoid activation function, producing output in the range [0,1], indicating the probability of pneumonia (1) or normal (0).
Step 3: Freeze the ResNet50V2 Layers
 All layers in the ResNet50V2 base are frozen (i.e., trainable=False), which means their weights will not be updated during training.
 This allows the model to retain the general-purpose feature extraction capabilities learned from ImageNet while training only the new classifier head on the pneumonia dataset.
Step 4: Compile and Train the Model
 The model is compiled using the Adam optimizer, which adapts learning rates during training and is well-suited for deep learning.
 The binary cross-entropy loss function is ideal for binary classification tasks like detecting pneumonia.
Accuracy is used as the evaluation metric to assess how well the model is distinguishing between normal and pneumonia X-rays.












