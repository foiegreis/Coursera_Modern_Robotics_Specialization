% start generating reference trajectory
clc; close all; clear all; 
addpath("./mr");

% load in paremeters based on testing scenarios 
load('newTask.mat');
%load('best.mat');
%load('overshoot.mat'); 
fprintf("Control Proportional Gain: %d\n", gains.Kp);
fprintf("Control Integral Gain: %d\n", gains.Ki);

% generate trajectory 
refTraj = generate_ref_trajectory(Tse0, Tsc0, TscG, TceG, TceS, k);
trajNum = size(refTraj, 1); 
states = zeros(trajNum - 1, 13);
errors = zeros(trajNum - 1, 6);

% simulate trajectory
disp("Simulating controlled process");
for idx = 1 : trajNum - 1
  % compute current configuration and references 
  X = compute_endeffector_configuration(currState);
  Xd = [refTraj(idx, 1), refTraj(idx, 2), refTraj(idx, 3), refTraj(idx, 10); ...
        refTraj(idx, 4), refTraj(idx, 5), refTraj(idx, 6), refTraj(idx, 11); ...
        refTraj(idx, 7), refTraj(idx, 8), refTraj(idx, 9), refTraj(idx, 12); ...
        0, 0, 0, 1];
  Xdn =[refTraj(idx+1, 1), refTraj(idx+1, 2), refTraj(idx+1, 3), refTraj(idx+1, 10); ...
        refTraj(idx+1, 4), refTraj(idx+1, 5), refTraj(idx+1, 6), refTraj(idx+1, 11); ...
        refTraj(idx+1, 7), refTraj(idx+1, 8), refTraj(idx+1, 9), refTraj(idx+1, 12); ...
        0, 0, 0, 1];
  % compute current control input 
  [Vb, Xerr] = generate_control_body_twist(X, Xd, Xdn, gains, deltaT);
  Jb = compute_youbot_jacobian(currState);
  u = Jb \ Vb; 
  % record current state and error 
  states(idx, :) = [currState', refTraj(idx, end)]; 
  errors(idx, :) = Xerr';
  % integrate forward 
  nextState = compute_next_youbot_state(currState, u, deltaT, 100);
  currState = nextState; 
end

% postprocessing: saving data and ploting out data 
disp("Generating animation csv file");
csvwrite("CoppeliaSim.csv", states);
disp("Writing and plotting out error data");
csvwrite("Errors.csv", errors);

% plot out error data figure
hfig = figure(); 
hold on;
t = [1:trajNum-1] * deltaT;
plot(t, errors, 'LineWidth', 2.5);
grid on; 
box on;
axis tight; 
xlabel('Time(second)', 'FontSize', 18);
ylabel('Error(meter for position, radian for angle)', 'FontSize', 18);
hleg = legend('eomega_x', 'eomega_y', 'eomega_z', 'ep_x', 'ep_y', 'ep_z');
set(gca, 'FontSize', 18);
set(hleg, 'FontSize', 18);
hold off; 
disp("Done");