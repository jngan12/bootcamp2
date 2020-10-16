#### To Implement Behavior Cloning for Line Following:

##### Dependencies:

* Tensorflow 1.14
* Keras 2.2.4
* CUDA 10.0



**Steps To Train:**

The first step in training would be data collection. When working with image data, especially when using CNNs, the best training data is usually real world data, where images have defined and differentiable textures. This training data has been provided to you, fully parsed. Download the training data from the [drive link](https://drive.google.com/drive/folders/1V4SbIxo7ohxFTo1Z0IslWOiT2gmk3DQ9). Copy the `/output` folder from the drive into the `/bag2csv` folder. 

To train natively on your PC, set the file-path correctly in  `model_train.py` and execute. (only recommended if you have a powerful enough GPU)

To train on [Google Colab](https://colab.research.google.com/), use the iPython notebook, `behavior_cloning.ipynb`. 

<u>Create your own training data:</u>

Collect training data in the real world as a bag file. Collect image, throttle and angle information. Modify `record_training.sh` with the correct topic names to record. Modify topic names in the `bagutils` file. 

Use `rosbag_to_csv.py` to convert to csv file:

```bash
$ python rosbag_to_csv.py -i bag_files/ -o output/
```



**Testing Trained Models:**

A number of trained models have already been provided to you. These models have been trained with the above script by changing different parameters:

![Screenshot from 2020-08-29 22-36-03](/home/adhitir/Pictures/Screenshot from 2020-08-29 22-36-03.png)

Three kinds of parameters have been altered -- *learning rate*, *batch size*, and the *number of epochs* the model is trained for. It is often hard to guess how these parameters affect the end performance of a DL model, but there are some evaluation metrics that allow us to visualize it. 

For each of these models, print a plot of

1. MSE vs number of epochs
   - Use MSE to calculate only validation loss if working with pre-trained models provided above
   - Use MSE to calculate validation and training loss if training your own models based on the parameters given above. 
2. Prediction vs true value

Based on what you see, can you describe which of these models will give you a good behavior? In this example, how would you describe MSE as an indicator of performance? Which other evaluation metrics can be used?

Use the `drive_model.py` script to actually test the model in Gazebo. As before, you will need to set the correct model paths within the script. Use the `line_follow_DL` world to bring up the vehicle in Gazebo.



**Submission**

Submit a report detailing your efforts in running behavior cloning with responses to the above questions. 

---

<u>Reference</u>: Wil Selby