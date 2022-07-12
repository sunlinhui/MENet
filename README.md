# Pytorch Implementation of MENet
## Environment
The proposed MENet is implemented on CUDA 10.1, Pytorch 1.1, and Python 3.6.
## Data Preparation
### Dataset
* DVS128 Gesture Dataset [[Link]](https://research.ibm.com/interactive/dvsgesture/)

* MNIST-DVS [[Link]](http://www2.imse-cnm.csic.es/caviar/MNISTDVS.html)

* N-Cars [[Link]](https://www.prophesee.ai/2018/03/13/dataset-n-cars/)

* CIFAR10-DVS [[Link]](https://paperswithcode.com/dataset/cifar10-dvs)
### Data processing
* DVS128 Gesture Dataset

1. Event streams contained in the dataset are divided into training set and testing set according to the ratio of 70% and 30%, respectively. 
2. The data of each event stream is converted into the txt format, in which each line records the (x,y,p,t) of an event. 
3. Each stream is split into multiple sliding windows with a fixed time interval of 0.5s and a step size set as 0.25s.
4. Two txt files for training and testing are generated, in which each line records the storage location and category of each sliding window:
  ```sh
  /01_user27_natural.txt 11	
 ```
   >where 01 means the first sliding window of the event stream user27_natural, and 11 means corresponding category of gesture.

* N-MNIST and N-Cars
1. The data of each event stream is converted into the txt format, in which each line records the (x,y,p,t) of an event.
2. Two txt files for training and testing are generated, in which each line records the storage location and category of each event stream:
  ```sh
  /n-cars_test/cars/obj_000000_td.txt 1	
 ```
 ## Run
 You can run different modes with following codes.
 ```sh
 ##Training MENet for DVS128 Gesture Dataset
python train.py –num_points=512 --outf='/MENet/weights/menet_dvs128' --numclass=11 --data_path='/DVS128/train_sliding_window.txt' --phase='train_MENet'

##Training MENet_single for DVS128 Gesture Dataset
python train.py –num_points=512 --outf='/MENet/weights/menet_single_dvs128' --numclass=11 --data_path='/DVS128/train_sliding_window.txt' --phase='train_MENet_single'

##Testing MENet for DVS128 Gesture Dataset
python train.py –num_points=512 --numclass=11 --data_path='/DVS128/test_sliding_window.txt' --model='/weights/menet_dvs.pth' --phase='test_MENet'
	
##Testing MENet_single for DVS128 Gesture Dataset
python train.py –num_points=512 --numclass=11 --data_path='/DVS128/test_sliding_window.txt' --model='/weights/menet_single_dvs.pth' --phase='test_MENet_single'
 ```

 ```sh
##Training MENet_single for N-MNIST
python train.py –num_points=512 --outf='/MENet/weights/menet_single_mnist' --numclass=10 --data_path='/N-MNIST/train.txt' --phase='train_MENet_single'

##Testing MENet_single for N-MNIST
python train.py –num_points=512 --numclass=10 --data_path='/N-MNIST/test.txt'  --model='/weights/menet_single_mnist.pth' --phase='test_MENet_single'
 ```
 ```sh
##Training MENet_single for N-Cars
python train.py –num_points=1024 --outf='/MENet/weights/menet_single_cars' --numclass=2 --data_path='/N-Cars/train.txt' --phase='train_MENet_single'

##Testing MENet_single for N-Cars
python train.py –num_points=1024 --numclass=2 --data_path='/N-Cars/test.txt'  --model='/weights/menet_single_cars.pth' --phase='test_MENet_single'

  ```
