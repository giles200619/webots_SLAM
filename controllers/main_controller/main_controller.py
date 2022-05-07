"""main_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor
from controller import CameraRecognitionObject
from scipy.stats.distributions import chi2
import numpy as np
import matplotlib.pyplot as plt


robot = Supervisor()

camera = robot.getDevice('camera')
camera.enable(1)

lidar = robot.getDevice('lidar')
lidar.enable(1)
lidar.enablePointCloud()

if camera.hasRecognition():
    camera.recognitionEnable(1)
    camera.enableRecognitionSegmentation()
else:
    print("Your camera does not have recognition")

timestep = int(robot.getBasicTimeStep())

leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0)
rightMotor.setVelocity(0)
#params
coef = 0.6
max_speed = 0.1 * coef
wheelRadius = 0.0205
axleLength = 0.0568 
updateFreq = 1
plotFreq = 50
max_omega = (6.28 * wheelRadius - max_speed) / (0.5*axleLength) *coef*0.4

# plot settings
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 10) 
## utils
def plot_cov(cov_mat, prob=0.95, num_pts=50):
    conf = chi2.ppf(0.95, df=2)
    L, V = np.linalg.eig(cov_mat)
    s1 = np.sqrt(conf*L[0])
    s2 = np.sqrt(conf*L[1])
    
    thetas = np.linspace(0, 2*np.pi, num_pts)
    xs = np.cos(thetas)
    ys = np.sin(thetas)

    standard_norm = np.vstack((xs, ys))
    S = np.array([[s1, 0],[0, s2]])
    scaled = np.matmul(S, standard_norm)
    R= V
    rotated = np.matmul(R, scaled)
    
    return(rotated)
    
def omegaToWheelSpeeds(omega, v):
    wd = omega * axleLength * 0.5
    return v - wd, v + wd
    
def lidar_scan_xy(data):
    # x-> right, y-> up
    d_theta = np.pi/(data.shape[0]-1)
    pts = []
    for i in range(data.shape[0]):
        dis = data[i]
        if dis == float("inf"):
            continue
        theta = d_theta * i
        x = -np.cos(theta)*dis
        y = np.sin(theta)*dis
        pts.append([-y,x])
    return np.asarray(pts)

def lidar_pcd(data):
    pts = []
    for i in range(len(data)):
        pts.append([-data[i].y,data[i].x])
    return np.asarray(pts)

def plt_show(x_hat_t, Sigma_x_t):
    plt.scatter(x_hat_t[3::2],x_hat_t[4::2],color="r")
    plt.show()
        
 # pts = lidar_scan_xy(np.array(lidar.getRangeImage()))
    # plt.plot(np.array(lidar.getRangeImage()))
    # plt.show()
    # pts = lidar_pcd(lidar.getPointCloud())
    # plt.scatter(pts[:,0],pts[:,1])
    # plt.show()
def stage_0_stop(camera):
    recObjs = camera.getRecognitionObjects()
    recObjsNum = camera.getRecognitionNumberOfObjects()
    for i in range(recObjsNum):
        if (recObjs[0].get_colors() == np.array([0,1,0])).all():
            return True
    return False
def stage_1_stop(camera):
    recObjs = camera.getRecognitionObjects()
    recObjsNum = camera.getRecognitionNumberOfObjects()
    for i in range(recObjsNum):
        if (recObjs[0].get_colors() == np.array([0,0,1])).all():
            return True
    return False
##############################################################
# wall follower
wall_threshold = 0.15
def wall_follow_step(lidar_scan):
    left = lidar_scan[0]
    mid = lidar_scan[255]
    right = lidar_scan[-1]
    # print(f"left:{left}",f"mid:{mid}",f"right:{right}")
    left_wall = left < wall_threshold
    front_wall = mid < wall_threshold*1
    right_wall = right < wall_threshold
    if front_wall or lidar_scan[125]<wall_threshold:
        # Go Right
        #print("Right")
        return -max_omega*0.5, max_speed*0.1
    elif (not left_wall) and (not front_wall):
        # Go Left
        #print("Left")
        return max_omega*0.5, max_speed
    elif left_wall and not front_wall:
        # Go Straight
        #print("Straight")
        return 0,max_speed
    else:
        return 0,max_speed

def open_space_step(lidar_scan,x_hat_t,Goal_pos):
    left = lidar_scan[255-125]
    mid = lidar_scan[255]
    right = lidar_scan[255+125]
    #print(f"left:{left}",f"mid:{mid}",f"right:{right}")
    left_wall = left < wall_threshold*2
    front_wall = mid < wall_threshold*2
    right_wall = right < wall_threshold*2
    if not front_wall and not right_wall:
        #d = ((Goal_pos[0][0]-x_hat_t[0])**2 + (Goal_pos[1][0]-x_hat_t[1])**2)**0.5
        #theta = -np.pi + np.arcsin((Goal_pos[1]-x_hat_t[1])/d)
        #theta -= x_hat_t[2]
        r = Goal_pos[1][0]-x_hat_t[1]
        theta = max_omega * 0.8
        return -r * theta, max_speed
    elif not front_wall and right_wall:
        return max_omega*0.5, max_speed
    else:
        if left_wall:
            return -max_omega, 0
        elif right_wall:
            return max_omega, 0
        else: 
            return 0, max_speed
            
def find_goal(camera):
    recObjs = camera.getRecognitionObjects()
    recObjsNum = camera.getRecognitionNumberOfObjects()
    for i in range(recObjsNum):
        if (recObjs[i].get_colors() == np.array([1,0,0])).all():
            landmark = robot.getFromId(recObjs[i].get_id())
            #G_p_L = landmark.getPosition()
            rel_lm_trans = landmark.getPose(robotNode)
            z = np.zeros((2,1))
            x = rel_lm_trans[3]
            y = rel_lm_trans[7]
            if x < 0.1:
                return 0,0
            else:
                return y*max_omega,max_speed
            
    return False
# SLAM
# Robot state
robotNode = robot.getSelf()
G_p_R = robotNode.getPosition()
G_ori_R = robotNode.getOrientation()
Goal_pos = np.array([[-2.9],[-0.01]])
dt = timestep / 1000.0
x_hat_t = np.array([G_p_R[0], G_p_R[1], 0])
Sigma_x_t = np.zeros((3,3))
#Sigma_x_t[0,0], Sigma_x_t[1,1], Sigma_x_t[2,2] = 0.01, 0.01, np.pi/90
Sigma_n = np.zeros((2,2))
std_n_v = max_speed*0.01
std_n_omega = max_omega*0.01
Sigma_n[0,0] = std_n_v * std_n_v
Sigma_n[1,1] = std_n_omega * std_n_omega
std_m = 0.05
Sigma_m = [[std_m*std_m, 0], [0,std_m*std_m]]

def SLAMPropagate(x_hat_t, # robot position and orientation
                 Sigma_x_t, # estimation uncertainty
                 u, # control signals
                 Sigma_n, # uncertainty in control signals
                 dt): # timestep
    x_hat_t[0] += u[0]*dt*np.cos(x_hat_t[2])
    x_hat_t[1] += u[0]*dt*np.sin(x_hat_t[2])
    x_hat_t[2] += u[1]*dt
    A = np.array([[1,0,-dt*u[0]*np.sin(x_hat_t[2])],
                  [0,1,dt*u[0]*np.cos(x_hat_t[2])],
                  [0,0,1]])
    N = np.array([[dt*np.cos(x_hat_t[2]),0],
                  [dt*np.sin(x_hat_t[2]),0],
                  [0,dt]])
   
    Sigma_x_t[:3,:3] = A @  Sigma_x_t[:3,:3] @ A.T + N @ Sigma_n @ N.T
    if Sigma_x_t.shape[0]>3:
        for i in range(int((Sigma_x_t.shape[0]-3)/2)):
            Sigma_x_t[(3+i*2):(3+i*2)+2,:3] = Sigma_x_t[(3+i*2):(3+i*2)+2,:3] @ A.T
            Sigma_x_t[:3,(3+i*2):(3+i*2)+2] = A @ Sigma_x_t[:3,(3+i*2):(3+i*2)+2]
    return x_hat_t, Sigma_x_t

d_threshold1 = 10
d_threshold2 = 100
def SLAMUpdate(x_hat_t, # robot position and orientation
                Sigma_x_t, # estimation uncertainty
                zs, # measurements
                Sigma_ms, # measurements' uncertainty
                dt): # timestep
    h = np.zeros((2,1))
    M = np.eye(2)
    H = np.zeros((2,3))
    new_z = []
    z_pair = [] # index of existing landmarks correspond to each z in zs
    # check if measurements already exist
    for i in range(len(zs)):
        z = zs[i]
        d_list = []
        # match each measurement with existing landmarks
        for j in range(3,x_hat_t.shape[0],2):
            h[0,0] = np.cos(x_hat_t[2])*(x_hat_t[j]-x_hat_t[0])+np.sin(x_hat_t[2])*(x_hat_t[j+1]-x_hat_t[1])
            h[1,0] = -np.sin(x_hat_t[2])*(x_hat_t[j]-x_hat_t[0])+np.cos(x_hat_t[2])*(x_hat_t[j+1]-x_hat_t[1])
             
            H[0,0] = -np.cos(x_hat_t[2])
            H[1,1] = -np.cos(x_hat_t[2]) 
            H[1,0] = np.sin(x_hat_t[2]) 
            H[0,1] = -np.sin(x_hat_t[2])
            H[0,2] = -np.sin(x_hat_t[2])*(x_hat_t[j]-x_hat_t[0])+np.cos(x_hat_t[2])*(x_hat_t[j+1]-x_hat_t[1])
            H[1,2] = -np.cos(x_hat_t[2])*(x_hat_t[j]-x_hat_t[0])-np.sin(x_hat_t[2])*(x_hat_t[j+1]-x_hat_t[1])
            S = H @ Sigma_x_t[:3,:3] @ H.T + M @ Sigma_ms[i] @ M.T
            d = (z-h).T @ np.linalg.inv(S) @ (z-h) # Mahalanobis Distance ^ 2
            d_list.append(d)
        # determine if z is new measurement
        if x_hat_t.shape[0] > 3:
            min_idx = np.argmin(d_list)
            #print(f'dlist:{d_list}')
            if d_list[min_idx] < d_threshold1:
                z_pair.append(min_idx)
            elif d_list[min_idx] < d_threshold2:
                continue
            else:
                new_z.append(z)
                z_pair.append(-1)
        else:
            new_z.append(z)
        
    # update
    n_land = int((x_hat_t.shape[0]-3)/2)
    
    for i in range(n_land):
        # check if each landmark has corresponding measurement
        try:
            z_idx = z_pair.index(i)
            z = zs[z_idx]
        except:
            continue
        HR = np.zeros((2,3))
        HR[0,0] = -np.cos(x_hat_t[2])
        HR[1,1] = -np.cos(x_hat_t[2]) 
        HR[1,0] = np.sin(x_hat_t[2]) 
        HR[0,1] = -np.sin(x_hat_t[2])
        HR[0,2] = -np.sin(x_hat_t[2])*(x_hat_t[2*i+3]-x_hat_t[0])+np.cos(x_hat_t[2])*(x_hat_t[2*i+4]-x_hat_t[1])
        HR[1,2] = -np.cos(x_hat_t[2])*(x_hat_t[2*i+3]-x_hat_t[0])-np.sin(x_hat_t[2])*(x_hat_t[2*i+4]-x_hat_t[1])
        HL = np.zeros((2,2))
        HL[0,0] = np.cos(x_hat_t[2])
        HL[1,1] = np.cos(x_hat_t[2])
        HL[1,0] = -np.sin(x_hat_t[2])
        HL[0,1] = np.sin(x_hat_t[2])
        h[0,0] = np.cos(x_hat_t[2])*(x_hat_t[2*i+3]-x_hat_t[0])+np.sin(x_hat_t[2])*(x_hat_t[2*i+4]-x_hat_t[1])
        h[1,0] = -np.sin(x_hat_t[2])*(x_hat_t[2*i+3]-x_hat_t[0])+np.cos(x_hat_t[2])*(x_hat_t[2*i+4]-x_hat_t[1])
        H_full = np.zeros((2,Sigma_x_t.shape[1]))
        H_full[:2,:3] = HR
        H_full[:2,2*i+3:2*i+5] = HL
        S = H_full @ Sigma_x_t @ H_full.T + M @ Sigma_ms[z_idx] @ M.T
        K = Sigma_x_t @ H_full.T @ np.linalg.inv(S) 
        x_hat_t += (K @ (z-h)).T[0]
        
        Sigma_x_t = Sigma_x_t-Sigma_x_t @ H_full.T @ np.linalg.inv(S) @ H_full @ Sigma_x_t
        
    # add new landmarks with new measurements
    sig_rows = Sigma_x_t.shape[0]
    sig_cols = Sigma_x_t.shape[1]
    x_hat_len = x_hat_t.shape[0]
    new_x_hat_t = np.zeros((x_hat_t.shape[0]+2*len(new_z)))
    new_Sigma_x_t = np.zeros((Sigma_x_t.shape[0]+2*len(new_z), Sigma_x_t.shape[1]+2*len(new_z)))
    new_x_hat_t[:x_hat_len] = x_hat_t
    new_Sigma_x_t[:sig_rows,:sig_cols] = Sigma_x_t
    
    c, s = np.cos(new_x_hat_t[2]), np.sin(new_x_hat_t[2])
    C = np.array([[c, -s], [s, c]])
    
    sigma_RR = new_Sigma_x_t[:3,:3]
    HL = np.zeros((2,2))
    HL[0,0] = np.cos(new_x_hat_t[2])
    HL[1,1] = np.cos(new_x_hat_t[2])
    HL[1,0] = -np.sin(new_x_hat_t[2])
    HL[0,1] = np.sin(new_x_hat_t[2])
    HR = np.zeros((2,3))
    HR[0,0] = -np.cos(new_x_hat_t[2])
    HR[1,1] = -np.cos(new_x_hat_t[2]) 
    HR[1,0] = np.sin(new_x_hat_t[2]) 
    HR[0,1] = -np.sin(new_x_hat_t[2])
    
    for i in range(len(new_z)):
        dRL = C @ new_z[i]
        for k in range(len(zs)):
            if (zs[k] == new_z[i]).all():
                idx_z = k
        new_x_hat_t[x_hat_t.shape[0]+i*2] = new_x_hat_t[0] + dRL[0,0]
        new_x_hat_t[x_hat_t.shape[0]+i*2+1] = new_x_hat_t[1] + dRL[1,0]

        HR[0,2] = -np.sin(new_x_hat_t[2])*(new_x_hat_t[0] + dRL[0,0]-new_x_hat_t[0])+np.cos(new_x_hat_t[2])*(new_x_hat_t[1] + dRL[1,0]-new_x_hat_t[1])
        HR[1,2] = -np.cos(new_x_hat_t[2])*(new_x_hat_t[0] + dRL[0,0]-new_x_hat_t[0])-np.sin(new_x_hat_t[2])*(new_x_hat_t[1] + dRL[1,0]-new_x_hat_t[1])
        
        new_Sigma_x_t[sig_rows+2*i:sig_rows+2*i+2,sig_cols+2*i:sig_cols+2*i+2] = \
        np.linalg.inv(HL) @ (HR @ sigma_RR @ HR.T + M @ Sigma_ms[idx_z] @ M.T) @ np.linalg.inv(HL).T
        new_Sigma_x_t[sig_rows+2*i:sig_rows+2*i+2,:3] = -np.linalg.inv(HL) @ HR @ sigma_RR 
        new_Sigma_x_t[:3,sig_rows+2*i:sig_rows+2*i+2] = -sigma_RR @ HR.T @ np.linalg.inv(HL).T
        
        for j in range(int((new_Sigma_x_t.shape[0]-5)/2)):
            sigma_RL = new_Sigma_x_t[:3,3+2*j:3+2*j+2]
            sigma_LR = new_Sigma_x_t[3+2*j:3+2*j+2,:3]
            
            new_Sigma_x_t[sig_rows+2*i:sig_rows+2*i+2, j*2+3:j*2+5] = -np.linalg.inv(HL) @ HR @ sigma_RL
            new_Sigma_x_t[j*2+3:j*2+5, sig_cols+2*i:sig_cols+2*i+2] = -sigma_LR @ HR.T @ np.linalg.inv(HL).T
    
    return new_x_hat_t, new_Sigma_x_t
                
               
# Main loop:
# - perform simulation steps until Webots is stopping the controller
steps = 0
stage = 0 # 0 -> wall follwer, 1 -> open space, 2 -> find goal
while robot.step(timestep) != -1:
    # moving strategy
    lidar_scan = np.array(lidar.getRangeImage())
    if stage == 0:
        # check stop 
        omega, v = wall_follow_step(lidar_scan)
        if stage_0_stop(camera):
            stage = 1
            omega = 0
            v = 0
            print('Starting at open space')
            #plt_show(x_hat_t, Sigma_x_t)
        
    elif stage == 1:
        omega, v = open_space_step(lidar_scan,x_hat_t,Goal_pos)
        if stage_1_stop(camera):
            stage = 2
            omega = 0
            v = 0
            print('Entering second maze')
    elif stage == 2:
        try:
            omega, v = find_goal(camera)
        except:
            omega, v = wall_follow_step(lidar_scan)
        
    else: 
        omega = 0
        v = max_speed
    # get GT, not used for SLAM, for verifying only
    G_p_R = robotNode.getPosition()
    G_ori_R = robotNode.getOrientation()
    # Control signals
    v_L, v_R = omegaToWheelSpeeds(omega+np.random.normal(0,std_n_omega), v+np.random.normal(0,std_n_v))
    #v_L, v_R = omegaToWheelSpeeds(omega, v)
    leftMotor.setVelocity(v_L/wheelRadius)
    rightMotor.setVelocity(v_R/wheelRadius)
    u = np.array([v, omega])
    # localization
    
    # Propergate
    x_hat_t, Sigma_x_t = SLAMPropagate(x_hat_t, Sigma_x_t, u, Sigma_n, dt)

    # Update
    recObjs = camera.getRecognitionObjects()
    recObjsNum = camera.getRecognitionNumberOfObjects()
    z_pos = np.zeros((recObjsNum, 2)) # relative position measurements   
    zs = []
    Sigma_ms = []
    if steps % updateFreq == 0:
        for i in range(0, recObjsNum):
            landmark = robot.getFromId(recObjs[i].get_id())
            #G_p_L = landmark.getPosition()
            rel_lm_trans = landmark.getPose(robotNode)
            z = np.zeros((2,1))
            z[0][0] = rel_lm_trans[3]+np.random.normal(0,std_m)
            z[1][0] = rel_lm_trans[7]+np.random.normal(0,std_m)
            zs.append(z)    
            Sigma_ms.append(np.asarray(Sigma_m))         

        x_hat_t, Sigma_x_t = SLAMUpdate(x_hat_t, Sigma_x_t, zs, Sigma_ms, dt)
    # print("GT Pos:",G_p_R,"EST", x_hat_t)
    if steps % plotFreq == 0:
        # pts = plot_cov(Sigma_x_t[0:2,0:2])
        # pts[0] += x_hat_t[0]
        # pts[1] += x_hat_t[1]
        # plt.scatter([pts[0,:]], [pts[1,:]],color="b")
        plt.scatter(x_hat_t[0],x_hat_t[1],color="b")
        plt.scatter(G_p_R[0],G_p_R[1],color="g")
        plt.axis('equal')
    if G_p_R[0]<=-2.9 and G_p_R[1]>-0.15 :
        plt_show(x_hat_t, Sigma_x_t)
        leftMotor.setVelocity(0)
        rightMotor.setVelocity(0)
        print("Reached Goal")
        break
        
    steps = steps + 1

    

# Enter here exit cleanup code.
