function [Vb, Xerr] = generate_control_body_twist(X, Xd, Xdn, gains, deltaT)
%% compute current control inputs as a body twist in the endeffector frame
%% 
%% Input:
%% X      current end effector configuration 
%% Xd     desired end effector configuration 
%% Xdn    desired end effector configuration at the next time step
%% gains  control gains 
%% deltaT time step 
%% 
%% Output:
%% Vb     desired body twist as control inputs 

% compute feedforward term
Vfd = se3ToVec(MatrixLog6(Xd\Xdn)) / deltaT;
ErrorMat = X\Xd;
Vfb = Adjoint(ErrorMat) * Vfd;
Xerr = se3ToVec(MatrixLog6(ErrorMat));

% use persistent as integration error 
persistent Xi;
if isempty(Xi)
  Xi = 0;
else 
  Xi += Xerr * deltaT;
endif

% get the overall control inputs
Vb = Vfb + gains.Kp * Xerr + gains.Ki * Xi;
  
end
