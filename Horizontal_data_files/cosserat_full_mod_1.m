function [shape,C] = cosserat_full_mod_1(x0_par,WpL,EC_F,L,gravity_on,n_t,varargin)


%%% example :
% without pre known shape and home orientation
%[shape] = kirchoff_full([.0138 .0022 5 2], .0863, [0;0;0],40e-2,0,100)

%%% with varargin
%[shape] = kirchoff_full([.0138 .0022 5 2], .0863,
%[0;0;0],40e-2,0,100,[zeros(1,3) 1.0000 zeros(1,3) 0.8660 0.5 0 -0.5
%0.8660])



% x0_par is [EI GJ kappa tau];
% EC_F = [3x1] matrix with global direction forces at the end
% WpL = Weight per unit length
% L = Length of the section
% gravity_on = 0 (off) and 1 (on)
% n_t = no of point used by the solver (this will be output also)
%
%  varargin(1} = init_pos = starting position and orientation of the section [12x1]
% varargin(2} = shape_guess = guess of all the position(x,y,z) (position should be of size [n_t x 3]

if nargin == 6
    start_guess = [zeros(n_t,3)   ones(n_t,1)           zeros(n_t,3) ...
        ones(n_t,1)    zeros(n_t,3)           ones(n_t,1) ...
        x0_par(3)*ones(n_t,1)   zeros(n_t,1) x0_par(4)*ones(n_t,1)   zeros(n_t,3)];
    R = eye(3);
    init_pos = [0;0;0; R(:)];
elseif nargin == 7
    init_pos = [varargin{1}];% starting position of base and orientation
    start_guess = [repmat(init_pos,n_t,1)  ...
        x0_par(3)*ones(n_t,1)   zeros(n_t,1) x0_par(4)*ones(n_t,1)   zeros(n_t,3)];
else
    init_pos = [varargin{1}];
    start_guess = [varargin{2}    ones(n_t,1)           zeros(n_t,3) ...
        ones(n_t,1)    zeros(n_t,3)           ones(n_t,1) ...
        x0_par(3)*ones(n_t,1)   zeros(n_t,1) x0_par(4)*ones(n_t,1)   zeros(n_t,3)];
    
    
end


g   = 9.80665;
% eg   = [0;0; gravity_on];           % global direction of gravity
eg   = [0;gravity_on; 0];           % global direction of gravity in +y direction for horizontal operation of arm
rhoA = WpL;                       % mass per unit length
C    = [x0_par(1); x0_par(1); x0_par(2)]  ;
u_p  = [x0_par(3)                   % precurved tube u's
    0
    x0_par(4)];

% dont need cyl_center, r_cyl,n_t(what?)

params = struct('u_p'          , u_p,...
    'g'            , g,...
    'C'            , C,...
    'eg'           , eg,...
    'rhoA'         , rhoA,...
    'EC_F'         , EC_F,...
    'R'            , eye(3),...
    'gravity_flag' , gravity_on,...
    'L'            , L,...
    'WpL'          , WpL,...
    'n_t'          , n_t,...
    'initial_shape', [],...
    'init_pos', init_pos);



% Compute  shape
x0 = 0*ones(n_t,1); % ( uniform local forces)


params.initial_shape = start_guess;
warning(''); % Clear last warning message
shape = final_shape(x0, params);

%
%     hold on
%     plot3(shape(:,1) ,shape(:,2) ,shape(:,3),'b');
%
%     axis equal
%     grid on


end

function sxint = final_shape(f_ly, params)

sxint = common_parts(f_ly, params);

end

function [sxint,dist,ode_range] = common_parts(x,params)




init_shape = params.initial_shape;
ode_step_size = params.L / (params.n_t - 1);
ode_range     = 0 : ode_step_size : params.L;
% Solve the BVP
options = bvpset('Vectorized', 'on','NMax',2000,'FJacobian', @(s,p) odeJac(s,p, x, params) );

solinit = bvpinit(ode_range, @(r) init_shape(round(r/ode_step_size+1), :));
warning('');
sol = bvp4c(@( s, p)  ode_main(s,p, x, params),...
    @(xa,xb) BoundCond(xa,xb, params),...
    solinit,...
    options);

[warnMsg,warnID] = lastwarn;
if ~isempty(warnMsg)
    n_t = params.n_t;
    init_shape = [zeros(n_t,3)   ones(n_t,1)           zeros(n_t,3) ...
        ones(n_t,1)    zeros(n_t,3)   ones(n_t,1) ...
        zeros(n_t,6)];
    
    u_p_orig = params.u_p;
    
    sol = bvpinit(ode_range, @(r) init_shape(round(r/ode_step_size+1), :));
    
    for i = 1:1:5
        params.u_p = u_p_orig/5*i;
        current_u = params.u_p;
        
        current_C = params.C;
        
        [current_C current_u];
        warning('');
        sol = bvp4c(@( s, p)  ode_main(s,p, x, params),...
            @(xa,xb) BoundCond(xa,xb, params),...
            sol,...
            options);
        [warnMsg,warnID] = lastwarn;
    end
    
    if ~isempty(warnMsg)
        sol.y = sol.y*0;
    end
end


% Compute dense output
sxint = deval(sol, ode_range)';

% xy distances (ignoring Z)
dist = sqrt(sum(sxint(:,1:2).^2,2));
end

function dxds = ode_main(s,...
    x,...
    f_ly,...
    params)

% BOTTLENECK: spline() took up most computation time. But it was evaluated very often
% for the same input [s] or equal spline representation [splines]. This means a lot of
% work can be recycled between consecutive calls:
persistent splines  f_y_prev  f_ly_prev  s_prev
%
% Recompute cubic splines, only when needed
reeval = isempty(f_y_prev);
if isempty(splines) || ~isequal(f_ly, f_ly_prev)
    splines = spline(0 : params.L/(numel(f_ly)-1) : params.L, f_ly);
    reeval  = true;
end

% Recompute value, only when needed
f_y = f_y_prev;
if reeval || ~isequal(s, s_prev)
    f_y = ppval_quick(splines, s); end

f_y_prev  = f_y;
s_prev    = s;
f_ly_prev = f_ly;

% look at syms_equations.m and symbolic_equations.m for jacobian and
% vectorization

C   = params.C;
u_p = params.u_p;

x4  = x( 4,:);   x5  = x( 5,:);    x6  = x( 6,:);     x7 = x( 7,:);
x8  = x( 8,:);   x9  = x( 9,:);    x10 = x(10,:);    x11 = x(11,:);
x12 = x(12,:);   x13 = x(13,:);    x14 = x(14,:);    x15 = x(15,:);
x16 = x(16,:);   x17 = x(17,:);    x18 = x(18,:);

f0  = params.rhoA  * params.g * (params.L - s);
f1  = params.eg(1) * f0;
f2  = params.eg(2) * f0;
f3  = params.eg(3) * f0;

f_yC = conj(f_y);

G1 = params.EC_F(1) + conj(x16) + f1;      G4 = conj(u_p(3)) - conj(x15);
G2 = params.EC_F(2) + conj(x17) + f2;      G5 = conj(u_p(2)) - conj(x14);
G3 = params.EC_F(3) + conj(x18) + f3;      G6 = conj(u_p(1)) - conj(x13);

dxds = [+x6
    +x9
    +x12
    +x5 .*x15 - x6 .*x14
    -x4 .*x15 + x6 .*x13
    +x4 .*x14 - x5 .*x13
    +x8 .*x15 - x9 .*x14
    -x7 .*x15 + x9 .*x13
    +x7 .*x14 - x8 .*x13
    +x11.*x15 - x12.*x14
    -x10.*x15 + x12.*x13
    +x10.*x14 - x11.*x13
    +(conj(x5).*G1 + conj(x8).*G2 + conj(x11).*G3 - C(2).*x15.*G5 + C(3).*x14.*G4)/C(1)
    -(conj(x4).*G1 + conj(x7).*G2 + conj(x10).*G3 - C(1).*x15.*G6 + C(3).*x13.*G4)/C(2)
    -(C(1)*x14.*G6 - C(2)*x13.*G5)/C(3)
    -x5 .*f_yC;
    -x8 .*f_yC;
    -x11.*f_yC];

end


function res = BoundCond(xa,...
    xb,...
    params)

res = [ xa( 1:12) -  params.init_pos(:)
    xb(13:18) - [params.u_p; 0; 0; 0]
    ];
end



function dfdx = odeJac(s,...
    x,...
    f_ly,...
    params)

% BOTTLENECK: spline() took up most computation time. But it was evaluated very often
% for the same input [s] or equal spline representation [splines]. This means a lot of
% work can be recycled between consecutive calls:
%     reduced_basis = 0:params.L/(numel(f_ly)-1):params.L;
%     f_y = interp1(reduced_basis,f_ly,s);
persistent splines  f_y_prev  f_ly_prev  s_prev

reeval = isempty(f_y_prev);

% Recompute cubic splines, only when needed
if isempty(splines) || ~isequal(f_ly, f_ly_prev)
    splines = spline(0 : params.L/(numel(f_ly)-1) : params.L, f_ly);
    reeval  = true;
end

% Recompute value, only when needed
f_y = f_y_prev;
if reeval || ~isequal(s, s_prev)
    f_y = ppval_quick(splines, s); end

f_y_prev  = f_y;
s_prev    = s;
f_ly_prev = f_ly;




x4  = x( 4);       x5  = x( 5);    x6  = x( 6);     x7  = x( 7);
x8  = x( 8);       x9  = x( 9);    x10 = x(10);     x11 = x(11);
x12 = x(12);       x13 = x(13);    x14 = x(14);     x15 = x(15);
x16 = x(16);       x17 = x(17);    x18 = x(18);

c1 = params.C(1);
c2 = params.C(2);
c3 = params.C(3);

f0 = params.g * params.rhoA * (params.L - s);
f1 = params.eg(1)*f0;
f2 = params.eg(2)*f0;
f3 = params.eg(3)*f0;

G1 = params.EC_F(1) + conj(x16) + f1;      G4 = c3 * (conj(params.u_p(3)) - conj(x15));
G2 = params.EC_F(2) + conj(x17) + f2;      G5 = c1 * (conj(params.u_p(1)) - conj(x13));
G3 = params.EC_F(3) + conj(x18) + f3;      G6 = c2 * (conj(params.u_p(2)) - conj(x14));

f_yC = conj(f_y);


dfdx = [0 0 0      0,       0,    1,       0,        0,    0,       0,       0,     0,                 0,                  0,                 0,            0,            0,             0
    0 0 0      0,       0,    0,       0,        0,    1,       0,       0,     0,                 0,                  0,                 0,            0,            0,             0
    0 0 0      0,       0,    0,       0,        0,    0,       0,       0,     1,                 0,                  0,                 0,            0,            0,             0
    0 0 0      0,     x15, -x14,       0,        0,    0,       0,       0,     0,                 0,                -x6,                x5,            0,            0,             0
    0 0 0   -x15,       0,  x13,       0,        0,    0,       0,       0,     0,                x6,                  0,               -x4,            0,            0,             0
    0 0 0    x14,    -x13,    0,       0,        0,    0,       0,       0,     0,               -x5,                 x4,                 0,            0,            0,             0
    0 0 0      0,       0,    0,       0,      x15, -x14,       0,       0,     0,                 0,                -x9,                x8,            0,            0,             0
    0 0 0      0,       0,    0,    -x15,        0,  x13,       0,       0,     0,                x9,                  0,               -x7,            0,            0,             0
    0 0 0      0,       0,    0,     x14,     -x13,    0,       0,       0,     0,               -x8,                 x7,                 0,            0,            0,             0
    0 0 0      0,       0,    0,       0,        0,    0,       0,     x15,  -x14,                 0,               -x12,               x11,            0,            0,             0
    0 0 0      0,       0,    0,       0,        0,    0,    -x15,       0,   x13,               x12,                  0,              -x10,            0,            0,             0
    0 0 0      0,       0,    0,       0,        0,    0,     x14,    -x13,     0,              -x11,                x10,                 0,            0,            0,             0
    0 0 0      0,  +G1/c1,    0,       0,   +G2/c1,    0,       0,  +G3/c1,     0,                 0,   (G4 + c2*x15)/c1, -(G6 + c3*x14)/c1,  conj(x5)/c1,  conj(x8)/c1,  conj(x11)/c1
    0 0 0 -G1/c2,       0,    0,  -G2/c2,        0,    0,  -G3/c2,       0,     0, -(G4 + c1*x15)/c2,                  0, +(G5 + c3*x13)/c2, -conj(x4)/c2, -conj(x7)/c2, -conj(x10)/c2
    0 0 0      0,       0,    0,       0,        0,    0,       0,       0,     0,  (G6 + c1*x14)/c3,  -(G5 + c2*x13)/c3,                 0,            0,            0,             0
    0 0 0      0,   -f_yC,    0,       0,        0,    0,       0,       0,     0,                 0,                  0,                 0,            0,            0,             0
    0 0 0      0,       0,    0,       0,    -f_yC,    0,       0,       0,     0,                 0,                  0,                 0,            0,            0,             0
    0 0 0      0,       0,    0,       0,        0,    0,       0,   -f_yC,     0,                 0,                  0,                 0,            0,            0,             0];

end


function X = ppval_quick(S, X)
% PPVAL   Compute the value of a cubic splines interpolant,
%         but then a bit faster than ppval() does it.
%
% See also spline.


if isscalar(X) %(faster)
    % Find breakpoint
    br = find(X > S.breaks, 1, 'last');
    if isempty(br), br = 1; end
    
    % Compute spline interpolant
    X(:) = S.coefs(br,:) * (X - S.breaks(br)).^[3; 2; 1; 0];
    
else % (more flexible)
    % Find breakpoint
    br = sum(bsxfun(@gt, X(:), S.breaks), 2);
    br(br==0) = 1;
    
    % Compute spline interpolant
    X(:) = sum(S.coefs(br,:) .* bsxfun(@power, X(:) - S.breaks(br).', [3 2 1 0]), 2);
end

end % subfunction
