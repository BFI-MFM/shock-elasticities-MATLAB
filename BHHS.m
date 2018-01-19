%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%BHHS Example%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 0 Setting up the model

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%Parameters%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma_g_norm = 0.00034; sigma_g = [0 sigma_g_norm 0]; 
sigma_s_norm = 0.038;   sigma_s = [0 0 sigma_s_norm]; 
sigma_A_norm = 0.0078;  sigma_A = [sigma_A_norm 0 0]; 

phi      = 1; 
delta    = 0.0050; 
rho      = 0.0042; 
gamma    = 10; 
a        = 0.0117;
lambda_g = 0.0210; 
lambda_s = 0.0130; 
g_bar    = 0.0; 
s_bar    = 1.0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Intermediate computations%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha_g = 1.0 / (rho + lambda_g);
sigmaGsigmaA = [alpha_g * sigma_g(1) + sigma_A(1) 
    alpha_g * sigma_g(2) + sigma_A(2) 
    alpha_g * sigma_g(3) + sigma_A(3)];

A = (1.0 - gamma) / 2.0 * sigma_s_norm^2;
B = ( (1.0 - gamma) *  ( sigma_s(1) * sigmaGsigmaA(1)  + sigma_s(2) * sigmaGsigmaA(2) + sigma_s(3) * sigmaGsigmaA(3) ) - (rho + lambda_s)  ) ;
C = (1.0 - gamma) / 2.0 *  dot(sigmaGsigmaA,sigmaGsigmaA) -sigma_A_norm^2 / 2.0;

alpha_s1 = (-B + sqrt(B^2 - 4 * A * C ) ) / (2*A);
alpha_s2 = (-B - sqrt(B^2 - 4 * A * C ) ) / (2*A);
alpha_s = max(alpha_s1,alpha_s2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Creating state space%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[g, s] = ndgrid(linspace(-0.3972,0.3972, 200),linspace(0.00000001,5.7264, 200)) ;

grid= [g(:) s(:)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Finishing the rest %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Pi = zeros(size(grid,1),3);
Pi(:,1) = (gamma - 1) * (alpha_g * sigma_g(1) * sqrt(grid(:,2)) + ...
    alpha_s * sigma_s(1) * sqrt(grid(:,2)) ) + ...
    gamma * sqrt(grid(:,2)) * sigma_A(1);
Pi(:,2) = (gamma - 1) * (alpha_g * sigma_g(2) * sqrt(grid(:,2)) + ...
    alpha_s * sigma_s(2) * sqrt(grid(:,2)) ) + ...
    gamma * sqrt(grid(:,2)) * sigma_A(2);
Pi(:,3) = (gamma - 1) * (alpha_g * sigma_g(3) * sqrt(grid(:,2)) + ...
    alpha_s * sigma_s(3) * sqrt(grid(:,2)) ) + ...
    gamma * sqrt(grid(:,2)) * sigma_A(3);

q = (a + 1/phi) / (rho + 1/phi) * ones(size(Pi,1),1);

alpha_0 = (rho * log(rho) + log(q)/phi - delta + lambda_g * g_bar * alpha_g + lambda_s * s_bar * alpha_s) / (rho);

r = rho + log(q)/phi + grid(:,1) - delta - sqrt(grid(:,2)) .* sigma_A_norm .* Pi(:,1);


muK = grid(:,1) + log(q)/phi - delta;
sigmaK = sqrt(grid(:,2)) .* repmat([1 0 0], size(grid,1),1) .* sigma_A_norm;


%% Step 1: Setting up the stochastic processes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Setting up struct model%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%muC, sigmaC, muS, sigmaS
model.muC    =  muK - 0.5 * dot(sigmaK', sigmaK')';  %%%consumption process
model.sigmaC =  sigmaK; 
model.sigmaS =  -Pi;  %%stochastic discount factor
model.muS    =  -r - 0.5 * dot(model.sigmaS', model.sigmaS')';

%%%muX, sigmaX
model.muX        = [ (lambda_g * (g_bar - grid(:,1)))  (lambda_s * (s_bar - grid(:,2)))];
sigmaX1          = sqrt(grid(:,2)) .* sigma_g;
sigmaX2          = sqrt(grid(:,2)) .* sigma_s;
model.sigmaX = {sigmaX1, sigmaX2};

%%%T, dt
model.T = 200; model.dt = 1;

%%%%%%%%%%
%%% x0 %%%
%%%%%%%%%%
x0 = [ -0.001183088,0.999882;
     -0.000784298,0.999882;
      0.00266582,0.999882;
     0.000770936,0.7670016;
    0.000770936, 0.984711;
    0.000770936, 1.25215];

%%%%%%%%%%%%%%%%%%%%%%%
%%Boundary conditions%%
%%%%%%%%%%%%%%%%%%%%%%%
bc = struct; bc.a0 = 0; bc.first = [1 1]; bc.second = [0 0]; bc.third = [0 0]; bc.level = [0 0];
bc.natural = false;

%%%%%%%%%%%%%%%%%%%%%%%%
%%Optional Arguments%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
optArgs.usePardiso = false;
optArgs.priceElas = true;


%% Step 2: Compute Elasticities
[expoElas, priceElas] = computeElas(grid, model, bc, x0, optArgs); 

%% Step 3: Plot elasticities

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Plot exposure elasticities%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('pos',[10 10 900 900])

subplot(2,3,1)
plot( linspace(1,40,model.T),[expoElas{1}.firstType(:,1) expoElas{2}.firstType(:,1) expoElas{3}.firstType(:,1)] * sqrt(12) , ...
    'LineWidth', 2);
legend( cellstr({strcat('g =', " ", num2str( round(x0(1,1),3) )), ...
    strcat('g =', " ", num2str( round(x0(2,1),3) )), ...
    strcat('g =', " ", num2str( round(x0(3,1),3) ))} ), 'FontSize',8 );
title("First Shock");

subplot(2,3,2)
plot( linspace(1,40,model.T), [expoElas{1}.firstType(:,2) expoElas{2}.firstType(:,2) expoElas{3}.firstType(:,2)] * sqrt(12), ...
    'LineWidth', 2);
title("Second Shock");

subplot(2,3,3)
plot(linspace(1,40,model.T), [expoElas{1}.firstType(:,3) expoElas{2}.firstType(:,3) expoElas{3}.firstType(:,3)] * sqrt(12), ...
    'LineWidth', 2);
title("Third Shock");


subplot(2,3,4)
plot(linspace(1,40,model.T), [expoElas{4}.firstType(:,1) expoElas{5}.firstType(:,1) expoElas{6}.firstType(:,1)] * sqrt(12), ...
    'LineWidth', 2);
legend( cellstr({strcat('s =', " ", num2str( round(x0(1,2),3) )), ...
    strcat('s =', " ", num2str( round(x0(2,2),3) )), ...
    strcat('s =', " ", num2str( round(x0(3,2),3) ))}), 'FontSize',8 );
title("First Shock");

subplot(2,3,5)
plot(linspace(1,40,model.T),  [expoElas{4}.firstType(:,2) expoElas{5}.firstType(:,2) expoElas{6}.firstType(:,2)] * sqrt(12), ...
    'LineWidth', 2);
title("Second Shock");

subplot(2,3,6)
plot(linspace(1,40,model.T), [expoElas{4}.firstType(:,3) expoElas{5}.firstType(:,3) expoElas{6}.firstType(:,3)] * sqrt(12), ...
    'LineWidth', 2);
title("Third Shock");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Plot price elasticities%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('pos',[10 10 900 900])

subplot(2,3,1)
plot( linspace(1,40,model.T),[priceElas{1}.firstType(:,1) priceElas{2}.firstType(:,1) priceElas{3}.firstType(:,1)] * sqrt(12) , ...
    'LineWidth', 2);
legend( cellstr({strcat('g =', " ", num2str( round(x0(1,1),3) )), ...
    strcat('g =', " ", num2str( round(x0(2,1),3) )), ...
    strcat('g =', " ", num2str( round(x0(3,1),3) ))}), 'FontSize',8 );
title("First Shock");

subplot(2,3,2)
plot( linspace(1,40,model.T), [priceElas{1}.firstType(:,2) priceElas{2}.firstType(:,2) priceElas{3}.firstType(:,2)] * sqrt(12), ...
    'LineWidth', 2);
title("Second Shock");

subplot(2,3,3)
plot(linspace(1,40,model.T), [priceElas{1}.firstType(:,3) priceElas{2}.firstType(:,3) priceElas{3}.firstType(:,3)] * sqrt(12), ...
    'LineWidth', 2);
title("Third Shock");

subplot(2,3,4)
plot(linspace(1,40,model.T), [priceElas{4}.firstType(:,1) priceElas{5}.firstType(:,1) priceElas{6}.firstType(:,1)] * sqrt(12), ...
    'LineWidth', 2);
legend( cellstr({strcat('s =', " ", num2str( round(x0(1,2),3) )), ...
    strcat('s =', " ", num2str( round(x0(2,2),3) )), ...
    strcat('s =', " ", num2str( round(x0(3,2),3) ))}), 'FontSize',8 );
title("First Shock");

subplot(2,3,5)
plot(linspace(1,40,model.T),  [priceElas{4}.firstType(:,2) priceElas{5}.firstType(:,2) priceElas{6}.firstType(:,2)] * sqrt(12), ...
    'LineWidth', 2);
title("Second Shock");

subplot(2,3,6)
plot(linspace(1,40,model.T), [priceElas{4}.firstType(:,3) priceElas{5}.firstType(:,3) priceElas{6}.firstType(:,3)] * sqrt(12), ...
    'LineWidth', 2);
title("Third Shock");
