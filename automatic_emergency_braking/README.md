
# Automatic Emergency Braking Implementation
Automatic emergency braking is an important safety feature for modern cars. It helps to reduce and avert accidents under circumstances such as unexpected objects in driving path or distracted drivers. In this project we implement this feature on a F110 racecar model using a proportional controller. The controller takes either the distance error or time-to-collide (TTC) error as input, and outputs the car velocity proportional to the error input. Since error input reaches 0 at the specified threshold, the car will stop as expected at that point to avoid collision.

## Team: Intel Phoenix
**Members:**
* Atul Hatalkar
* Robert Aguirre
* Vinodkumar Mudaliar
* Zhijian Hua

## Note:
This is my individual submission for record. For Intel Phoenix team submission, please see Robert Aguirre's git submission.

## How to run the code
Please follow the steps below to run the code:
1. Clone the code repository into ROS WS on host: **git clone  https://github.com/SAE-Robotics-Bootcamp/bootcamp-assignments.git**
2. Build ROS projects if not already done so: **catkin_make**
3. Run launch file: **roslaunch automatic_emergency_braking aeb.launch**
4. Default controller is dist_control. To switch to using ttc_control, edit  the **~/sae_ws/ros_ws/src/bootcamp-assignments/automatic_emergency_braking/launch/aeb.launch** file. Change  ** dist_ctrl** inside this line **param name="control" value="dist_ctrl"** to **ttc_ctrl**. Then rerun the launch file above to view the ttc controller scenario.

## Design Considerations
1. P controller takes distance error and ttc error as input. Both inputs are normalized into the same range as VELOCITY prior to multiplied by coefficient k.
2. The velocity is clamped at 0 before being fed to telop controller to prevent possible overshoot and oscillation.
3. Based on my understanding of the assignment, both DIST_THRESHOLD and TTC_THRESHOLD are points where the error value equals to zero. By using p controller it follows that velocity at this point is 0, i.e. car stops here. I then set an arbitrary trigger value for engaging emergency brake. The value for this trigger is set as 1 in my code, meaning 1m before DIST_THRESHOLD in dist_control case, and 1s before TTC_THRESHOLD in ttc_control case.

## Results
A video recording od both dist_control and ttc_control can be found here:
**./video/sae_robotics_bootcamp_aeb_assignment.mp4**
<!-- blank line -->
<figure class="video_container">
  <video controls="true" allowfullscreen="true" poster="path/to/poster_image.png">
    <source src="./video/sae_robotics_bootcamp_aeb_assignment.mp4" type="video/mp4">
  </video>
</figure>
<!-- blank line -->

## Controller Evaluation
Distance controller takes only one variable, which is the distance from the obstacle, into consideration. TTC control combines both the distance and current vehicle velocity and offers a metric that reflects realistically the state of the vehicle, thus providing better knowledge for decision making.

## Reflections and Remaining Questions
1. Hokuyo lidar datasheet shows returning 270 degree ranges array containing precisely 1080 elements (0-1079), whereas in simulation the ranges array contains 1081 elements. Where does the extra 1 element come from?
2. Averaging the distance of the wall over Lidar range points to estimate distance works for this project. But it  may not work properly in some real-world scenarios. For example: a wall with curvature or a wall having trees standing in front of it.
3. The simulator sometimes display car veering off randomly, even when steering angle is fixed at 0. It usually goes away with multiple runs of the program.




