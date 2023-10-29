function [stateTraj] = generate_ref_trajectory(Tse0, Tsc0, TscG, TceG, TceS, k)
%% get a continuous trajectory connecting each segments from start to the goal trajectory 
%% 
%% Input:
%% Tse0       Initial endeffector configuration 
%% Tsc0       Initial cube configuration 
%% TscG       Goal cube configuration 
%% TceG       grasping endeffector configuration w.r.t cube
%% TceS       standoff endeffector configuration w.r.t cube
%% k          sampling rate w.r.t 0.01s
%%
%% Output: 
%% stateTraj  a N by 13 array where each row is given as [r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper state] 

stateTraj = []; 

% compute key endeffector frames
TsePickupStandoff = Tsc0 * TceS; 
TsePickup = Tsc0 * TceG; 
TseReleaseStandoff = TscG * TceS;
TseRelease = TscG * TceG; 

% initialize parameters 
transVel = 0.4; 
rotVel = 20.0 / 180 * pi; 
deltaT = 0.01 / k; 
gripperDuration = 1.3; % 1 second, 0.625s operation time 
keyFrames = { Tse0; 
              TsePickupStandoff;
              TsePickup;
              TsePickup; 
              TsePickupStandoff;
              TseReleaseStandoff; 
              TseRelease; 
              TseRelease; 
              TseReleaseStandoff }; 
keyGrippers = [ 0;
                0;
                0;
                1; 
                1;
                1;
                1;
                0;
                0]; 
segNum = length(keyFrames) - 1; 
segSamples = zeros(segNum, 1);                 
                
% allocate time duration for each segment and generate trajectory accordingly 
for segIdx = 1 : segNum
  gripperStatus = keyGrippers(segIdx + 1);
  startFrame = keyFrames{segIdx};
  goalFrame = keyFrames{segIdx+1};
  currDuration = get_rotation_traj(startFrame,
                                   goalFrame,
                                   rotVel,
                                   transVel); 
  if currDuration <= deltaT 
    % the overall duration is too small, indicate gripper operation
    segSamples(segIdx) = gripperDuration / deltaT;
    
    % keep configuration fixed for a default duration
    stateTraj = [
            stateTraj;
            repmat([startFrame(1,1), ...
                    startFrame(1,2), ...
                    startFrame(1,3), ...
                    startFrame(2,1), ...
                    startFrame(2,2), ...
                    startFrame(2,3), ...
                    startFrame(3,1), ...
                    startFrame(3,2), ...
                    startFrame(3,3), ...
                    startFrame(1,4), ...
                    startFrame(2,4), ...
                    startFrame(3,4), ...
                    gripperStatus], 
                   [segSamples(segIdx), 1])]; 
  else 
    % get sample numbers
    segSamples(segIdx) = ceil(currDuration / deltaT);
    sampleDuration = segSamples(segIdx) * deltaT; 
    
    % generate a trajectory using Cartesian interpolation 
    matTraj = CartesianTrajectory(startFrame, 
                                  goalFrame, 
                                  sampleDuration, 
                                  segSamples(segIdx), 
                                  3);
    % concatenate trajectory 
    matTraj = cell2mat(matTraj);
    stateTraj = [
            stateTraj;[
            matTraj(1, 1:4:end)', ...
            matTraj(1, 2:4:end)', ...
            matTraj(1, 3:4:end)', ...
            matTraj(2, 1:4:end)', ...
            matTraj(2, 2:4:end)', ...
            matTraj(2, 3:4:end)', ...
            matTraj(3, 1:4:end)', ...
            matTraj(3, 2:4:end)', ...
            matTraj(3, 3:4:end)', ...
            matTraj(1, 4:4:end)', ...
            matTraj(2, 4:4:end)', ...
            matTraj(3, 4:4:end)', ...
            ones(segSamples(segIdx), 1) * gripperStatus]]; 
  end
end
  
end


function [duration] = get_rotation_traj(T0, T1, omega, v)
  % get time duration based on translation and rotation difference 
  posDiff = norm(T0(1:3, 4) - T1(1:3, 4)); 
  rotDiff = norm(MatrixLog3(T0(1:3, 1:3)' * T1(1:3, 1:3)), 'fro') / sqrt(2); 
  duration = max(posDiff / v, rotDiff / omega); 
end
