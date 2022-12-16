# How to run this project

Most file can be run on their own, and their name describes what they do.
The majority of the files simply create the models used in this project, the
culmination of which can be found in `drone_controls/fruit_fly.py`.
This script uses all trained models to control a Tello drone and detect fruit's rotten
and ripe states.
It is currently set up to use a computer's webcam instead of a drone, but this can be
changed by uncommenting line 235 and commenting line 236
To only run the robot with movement (skipping classification), run
`drone_controls/drone_controller.py`.

# Resources

Link to training
video: <https://drive.google.com/file/d/1rUllqH4ub0sCb3yRRxXu7i9uTqhxln_W/view?ts=63653393>

## Tello Documentation

Repo: <https://github.com/DIRECTLab/robots/tree/master/tello-drones>

SDK
Guide: <https://terra-1-g.djicdn.com/2d4dce68897a46b19fc717f3576b7c6a/Tello%20%E7%BC%96%E7%A8%8B%E7%9B%B8%E5%85%B3/For%20Tello/Tello%20SDK%20Documentation%20EN_1.3_1122.pdf>

User
Guide: <https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf>
