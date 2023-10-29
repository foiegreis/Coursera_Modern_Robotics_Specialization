function [nextState] = compute_next_youbot_state(currState, controlInput, duration, controlLimits)
%% Compute next state of youbot based on kinematic model (in SI unit)
%% 
%% Inputs:
%% currState      12 dimensional state vector as [chassis configuration(theta, x, y), arm joint angles(theta1 - theta5), wheel angles(theta1 - theta4)] 
%% controlInput   9 dimensional control vector as [wheel speeds(u1 - u4), arm joint speeds(dtheta1 - dtheta5)]
%% duration       time duration
%% controlLimits  a single or 9 dimensional vector indicating joint limits 

%% Outputs:
%% nextState      12 dimensional state vector as [chassis configuration(theta, x, y), arm joint angles(theta1 - theta5), wheel angles(theta1 - theta4)] 

if length(controlLimits) != 1 && length(controlLimits) != length(controlInput)
  error("Control limit dimension has to be consistent with control input dimension!");
end
% relevant constant parameters
armStartIndex = 4;
armEndIndex = 8;
wheelStartIndex = 9;
wheelEndIndex = 12;
armSpeedStartIndex = 5;
armSpeedEndIndex = 9;
wheelSpeedStartIndex = 1;
wheelSpeedEndIndex = 4;

% apply input saturation
controlInput = max(controlInput, -controlLimits);
controlInput = min(controlInput, controlLimits);

% using forward Euler integration method 
nextState = currState; 
nextState(armStartIndex : armEndIndex) += duration * controlInput(armSpeedStartIndex : armSpeedEndIndex);
nextState(wheelStartIndex : wheelEndIndex) += duration * controlInput(wheelSpeedStartIndex : wheelSpeedEndIndex); 

% compute planar twist
r = 0.0475;
l = 0.47 / 2;
w = 0.3 / 2;
H0 = 1/r * [-l-w, 1, -1;
             l+w, 1, 1;
             l+w, 1, -1;
            -l-w, 1, 1]; 
Vb = H0\[controlInput(1);
         controlInput(2);
         controlInput(3);
         controlInput(4)]; 

% compute the velocity tilting angle
omega = Vb(1);
vx = Vb(2);
vy = Vb(3);
theta = currState(1);
x = currState(2);
y = currState(3);
    
% integrate forward using matrix exponential 
T = [cos(theta), -sin(theta), 0, x;
     sin(theta), cos(theta), 0, y;
     0, 0, 1, 0;
     0, 0, 0, 1];
Tb = MatrixExp6(VecTose3([0; 0; omega; vx; vy; 0]) * duration);
Tn = T * Tb; 

% extract state as the new state 
nextState(1) = atan2(Tn(2, 1), Tn(1, 1)); 
nextState(2) = Tn(1, 4);
nextState(3) = Tn(2, 4);

end
