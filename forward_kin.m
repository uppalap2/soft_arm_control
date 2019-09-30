function [P_O_end] = forward_kin(Pb,Pr1,theta,L)
% Default values for testing
EL = [0;0;30e-3]*10; %load at tip of soft arm
% L = 24e-2; % full length (can vary from 1e-2 to 24e-2)
% theta (in degrees) = 0(default)

% OUTPUTS:

% P_end = end position [x;y;z] (3x1)
% O_end = end orientation [tx;ty;tz] (3x1) tangent direction at end 

% INPUTS:

% Pb = benidng pressure in 11 psi to 40 psi range
% Pr1 = rotating pressure range(-44 to +44)
%       (0 to 44 then clockwise, -44 to
%       0 then counter clockwise rotation)

load('Control valves_2/EI_fine_h.mat');
load('Control valves_2/GJ_fine_h.mat');
load('Control valves_2/kappa_fine_h.mat');
load('Control valves_2/tau_fine_h.mat');

P_b_q = 11:1:40;
P_r_q = 0:1:44;

n_t     = 101;%51 for < 20 % 61 for 20
% init_g_value = 9.81;
WpL = .1194;
F_e = EL;
% n = 15;
% L = 31e-2;
gravity_on = 0;

k = interp2(P_b_q,P_r_q,kappaq,Pb,abs(Pr1));
t = interp2(P_b_q,P_r_q,tauq,Pb,abs(Pr1));
EI = interp2(P_b_q,P_r_q,EIq,Pb,abs(Pr1));
GJ = interp2(P_b_q,P_r_q,GJq,Pb,abs(Pr1));
% EI = .013;
% EI = .0232;

if Pr1 >= 0 % clockwise rotation
    t = -t;
end

% remains +ve value for other direction
    

shape = cosserat_full_mod([EI GJ k t],WpL,F_e,L,gravity_on,n_t);

P_end = shape(end,1:3)';
O_end = reshape(shape(end,4:12),3,3)';
O_end = O_end(:,3);

RotZ = [cosd(theta) -sind(theta) 0;
    sind(theta) cosd(theta) 0;
    0          0          1];

P_end_n = RotZ*P_end;
O_end_n = RotZ*O_end;

% % plotting shape and end orientation
% plot3((shape(:,1)),(shape(:,2)),(shape(:,3)),'r:');
% axis equal
% grid on
% hold on
% a = P_end;
% b = O_end;
% 
% plot3([a(1) a(1)+b(1)/100],[a(2) a(2) + b(2)/100],[a(3) a(3) + b(3)/100],'k.')


P_O_end = [P_end_n'*100];% ,O_end_n'];

end
