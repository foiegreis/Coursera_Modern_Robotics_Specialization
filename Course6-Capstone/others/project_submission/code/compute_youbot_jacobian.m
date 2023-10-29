function [Jb] = compute_youbot_jacobian(currState)
%% compute body Jacobian of youbot w.r.t control input
%% 
%% Input:
%% currState    12 dimensional state vector as [chassis configuration(theta, x, y), arm joint angles(theta1 - theta5), wheel angles(theta1 - theta4)] 
%% 
%% Output:
%% Jb           6 x 9 matrix representing Jacobian w.r.t wheel and arm joint speed
%%

% compute initial configuration 
Tb0 = [1, 0, 0, 0.1662;
       0, 1, 0, 0;
       0, 0, 1, 0.0026;
       0, 0, 0, 1]; 
T0e = [1, 0, 0, 0.033;
       0, 1, 0, 0;
       0, 0, 1, 0.6546;
       0, 0, 0, 1];
Tbe = Tb0 * T0e;

% compute current endeffector configuration w.r.t base
Blist = [[0; 0; 1; 0; 0.033; 0], ...
         [0;-1; 0;-0.5076;0; 0], ...
         [0;-1; 0;-0.3526;0; 0], ...
         [0;-1; 0;-0.2176;0; 0], ...
         [0; 0; 1; 0; 0; 0]];
thetaList = currState(4:8);
Tbec = FKinBody(Tbe, Blist, thetaList); 

% compute jacobian of wheel control inputs 
r = 0.0475;
l = 0.47 / 2;
w = 0.3 / 2;
H0 = 1/r * [-l-w, 1, -1;
             l+w, 1, 1;
             l+w, 1, -1;
            -l-w, 1, 1]; 
Jb1 = [zeros(2, 4);
       pinv(H0);
       zeros(1, 4)]; 
Jb1 = Adjoint(inv(Tbec)) * Jb1; 

% compute jacobian of arm part 
Jb = [Jb1, JacobianBody(Blist, thetaList)];
  
end
