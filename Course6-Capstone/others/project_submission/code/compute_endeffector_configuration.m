function [X] = compute_endeffector_configuration(currState)
%% Compute endeffector configuration based on state vector 
ctheta = cos(currState(1));
stheta = sin(currState(1));
Tsb = [ctheta, -stheta, 0, currState(2);
       stheta, ctheta, 0, currState(3); 
       0, 0, 1, 0.0963;
       0, 0, 0, 1];
Tb0 = [1, 0, 0, 0.1662;
       0, 1, 0, 0;
       0, 0, 1, 0.0026;
       0, 0, 0, 1]; 
T0e = [1, 0, 0, 0.033;
       0, 1, 0, 0;
       0, 0, 1, 0.6546;
       0, 0, 0, 1];
Tse = Tsb * Tb0 * T0e;

% compute current endeffector configuration w.r.t base
Blist = [[0; 0; 1; 0; 0.033; 0], ...
         [0;-1; 0;-0.5076;0; 0], ...
         [0;-1; 0;-0.3526;0; 0], ...
         [0;-1; 0;-0.2176;0; 0], ...
         [0; 0; 1; 0; 0; 0]];
thetaList = currState(4:8);
X = FKinBody(Tse, Blist, thetaList); 

end
