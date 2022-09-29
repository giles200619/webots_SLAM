# webots_SLAM
An implementation of EKF-SLAM and wall-follower in Webots.

Gaussian noise is added to the control signal and measurement signal. The measurement is represented as the relative position from the landmarks to the
robot. Mahalanobis distance check is applied to distinguish landmarks. 

<p align="center">
  <img src="/images/environment.png" height="200">
  <br>Webots environment layout<br>
  <br><br>
  <img src="/images/result.png" height="400">
  <br>SLAM result. Ground truth trajectory v.s. estimated trajectory and keypoint locations. <br>
  <br><br>
  
</p>
