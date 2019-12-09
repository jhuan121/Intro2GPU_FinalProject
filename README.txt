Goal:
Develop an image classifier that can use Tensorflow's CUDA GPU Acceleration, to identify CT medical images
that are of the abdominal region of the body and images that are not of the abdominal region of the body.

Main Code Files:
code/preprocessing.py - turn raw images from the NIH dataset into scaled intensity images to more clearly show features in the image
code/classification.py - take the preprocessed images and build a classifier model that uses GPU acceleration

Dataset: NIH DeepLesion CT Image Dataset
https://nihcc.app.box.com/v/DeepLesion/file/307763142914
https://nihcc.app.box.com/v/DeepLesion/file/307764249542
https://nihcc.app.box.com/v/DeepLesion/file/307764129006

Trial Run Screenshots and Standard Out:
gpu_trial_3/*