%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%He & Krushnamurthy example%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 0: Setting up inputs

%%%%%%%%%%%%%%
%%Parameters%%
%%%%%%%%%%%%%%

muY = 0.02; sigmaY = 0.09; m = 4; lambda = 0.6; rho = 0.04;
l = 1.84;
x_star = (1-lambda) / (1-lambda + m);

alpha = @(x) (1 - lambda * (ones(size(x,2),1)' - x) ).^(-1) .* (x > x_star) ...
    + (1+m)^(-1) * x.^(-1) .* (x <= x_star) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%Initialize state%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = linspace(0, 1 - 0,10000);

model.muX = (x .* (- l / (1+l)*rho + (alpha(x) - ones(size(x,2),1)' ).^(2)*(sigmaY^2)))';
model.sigmaX = { fillmissing( (x .* (alpha(x) - ones(size(x,2),1)' ) * sigmaY)', 'nearest') };
model.muS = (rho / (1+l) + muY - alpha(x) *(sigmaY^2) + 0.5 * (alpha(x)).^(2)*(sigmaY^2))';
model.sigmaS = (alpha(x) * sigmaY)';

model.muS = -model.muS; model.sigmaS = -model.sigmaS;

%%%%Fill nans%%%%%
model.muX = fillmissing(model.muX, 'nearest'); model.muS = fillmissing(model.muS, 'nearest');
model.sigmaS = fillmissing(model.sigmaS, 'nearest');

%%boundary conditions
bc = struct; bc.a0 = 0; bc.first = [1]; bc.second = [0]; bc.third = [0]; bc.level = [0];
bc.natural =  false;

%%Time settings
model.T = 100;
model.dt = 1;

%%x0
x0 = [0.0471, 0.0822, 0.1266]';

%% Step 1: Compute shock elasticities for aggregate consumption

%%%%Aggregate consumption
model.muC = repmat(muY - 0.5 * sigmaY^2, size(x,2),1);
model.muC = fillmissing(model.muC, 'nearest');
model.sigmaC = repmat(sigmaY, size(x,2),1);
model.sigmaC = fillmissing(model.sigmaC, 'nearest');

%%%compute elasticities for aggregate consumption
[expoElas, priceElas] = computeElas( x', model, bc, x0);

%%%Plot
figure()
subplot(1,2,1)
plot([round(expoElas{1}.firstType,2) round(expoElas{2}.firstType,2) round(expoElas{3}.firstType,2)])
title("Exposure Elasticities First Type (G1)");
legend( { num2str(x0(1)), num2str(x0(2)), num2str(x0(3)) }, 'FontSize',8 );

subplot(1,2,2)

plot([round(expoElas{1}.secondType,2) round(expoElas{2}.secondType,2) round(expoElas{3}.secondType,2)])
title("Exposure Elasticities Second Type (G1)");
legend( { num2str(x0(1)), num2str(x0(2)), num2str(x0(3)) }, 'FontSize',8 );

saveas(gcf,'HK_shockExpoG1.png')


figure()
subplot(1,2,1)
plot([priceElas{1}.firstType priceElas{2}.firstType priceElas{3}.firstType])
legend( { num2str(x0(1)), num2str(x0(2)), num2str(x0(3)) }, 'FontSize',8 );

title("Price Elasticities First Type (G1)");

subplot(1,2,2)

plot([priceElas{1}.secondType priceElas{2}.secondType priceElas{3}.secondType])
title("Price Elasticities Second Type (G1)");
legend( { num2str(x0(1)), num2str(x0(2)), num2str(x0(3)) }, 'FontSize',8 );

saveas(gcf,'HK_shockPriceG1.png')

%% Step 2: Compute Shock Elasticities for Expert Consumption

%%%%Expert Consumption
model.muC = -model.muS - rho;
model.sigmaC = -model.sigmaS;


%%%compute elasticities
[expoElas, priceElas] = computeElas( x', model, bc, x0);

%%%Plot
figure()
subplot(1,2,1)
plot([expoElas{1}.firstType expoElas{2}.firstType expoElas{3}.firstType])
legend( { num2str(x0(1)), num2str(x0(2)), num2str(x0(3)) }, 'FontSize',8 );

title("Exposure Elasticities First Type (G2)");


subplot(1,2,2)
plot([expoElas{1}.secondType expoElas{2}.secondType expoElas{3}.secondType])
legend( { num2str(x0(1)), num2str(x0(2)), num2str(x0(3)) }, 'FontSize',8 );

title("Exposure Elasticities Second Type (G2)");

saveas(gcf,'HK_shockExpoG2.png')

figure()
subplot(1,2,1)
plot([priceElas{1}.firstType priceElas{2}.firstType priceElas{3}.firstType])
legend( { num2str(x0(1)), num2str(x0(2)), num2str(x0(3)) }, 'FontSize',8 );

title("Price Elasticities First Type (G2)");

subplot(1,2,2)

plot([priceElas{1}.secondType priceElas{2}.secondType priceElas{3}.secondType])
legend( { num2str(x0(1)), num2str(x0(2)), num2str(x0(3)) }, 'FontSize',8 );
title("Price Elasticities Second Type (G2)");
saveas(gcf,'HK_shockPriceG2.png')

%% Step 3: Compute Stationary Density

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Compute stationary density%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt = 1/255; T = 100;
drifts = {model.muX}; vols = model.sigmaX;
hists = simStatDent({x}, rand(10,1), dt, T, drifts, vols);