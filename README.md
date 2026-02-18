# Aurora-Detection
Use 'train_model.py' and 'test_model.py' for actual cases.
Use 'train_model_MNIST.py' and 'test_model_MNIST.py' as test cases to execute with the publicly available MNIST dataset.

## Training procedure
1. Put aurora images in /aurora_images.
2. Prepare CSV files for training and validation (contains only training and validation data, respectively) with two columns: "filename_img" and "label_detection"(labels indicating the presence of auroras).
   *Typically, 60-80% of the data is used for training, with the remainder reserved for validation and test.
3. Update the path to the training and validation CSV file, and to the directory containing images.
4. Update the path to save the CSV file of the history and the trained model weights.
5. Excute.
   *Select your favorite base model, and adjust epochs, learning rate, optimizer, and scheduler as needed.

## Test procedure
1. Prepare CSV files for test (contains only test data) with two columns: "filename_img" and "label_detection"(labels indicating the presence of auroras).
2. Updata the path to the trained model weights.
3. Update the path to the test CSV file, and to the directory containing images.
4. Update the path to save the CSV file of the estimated labels.
5. Excute.
