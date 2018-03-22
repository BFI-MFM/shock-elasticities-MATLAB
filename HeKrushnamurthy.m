%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%He & Krushnamurthy example%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 0: Setting up inputs
clear all
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

model.muX = @(x) (x .* (- l / (1+l)*rho + (alpha(x) - ones(size(x,2),1)' ).^(2)*(sigmaY^2)))';
model.sigmaX = @(x)  (fillmissing( (x .* (alpha(x) - ones(size(x,2),1)' ) * sigmaY)', 'nearest'))' ;
model.sigmaX = {model.sigmaX};
model.muS = @(x) -(rho / (1+l) + muY - alpha(x) *(sigmaY^2) + 0.5 * (alpha(x)).^(2)*(sigmaY^2))';
model.sigmaS = @(x) -(alpha(x) * sigmaY)';


%%%%Fill nans%%%%%
model.muX = @(x) fillmissing(model.muX(x)', 'nearest'); model.muS = @(x) fillmissing(model.muS(x)', 'nearest');
model.sigmaS = @(x) fillmissing(model.sigmaS(x)', 'nearest');

%%boundary conditions
bc = struct; bc.a0 = 0; bc.first = [1]; bc.second = [0]; bc.third = [0]; bc.level = [0];
bc.natural =  false;

%%Time settings
model.T = 100;
model.dt = 1;

%% Step 1: Compute Stationary Density

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Compute stationary density%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt = 1/255; T = round(3000 / 10 );
drifts = {model.muX}; vols = model.sigmaX;
hists = simStatDent({x}, rand(10,1), dt, T, drifts, vols);
burnRate = 0.2;
burned = cellfun(@(x) x(round( burnRate * size(x,1)):end,:), hists, 'UniformOutput', false);
alldata = cat(1, burned{:});
x0 = prctile(alldata,[10, 50, 90])';

%% Step 2: Compute shock elasticities for aggregate consumption

%%%%Aggregate consumption
model.muC = @(x) fillmissing(repmat(muY - 0.5 * sigmaY^2, size(x',2),1), 'nearest');
model.sigmaC = @(x) fillmissing(repmat(sigmaY, size(x',2),1), 'nearest');

%%%compute elasticities for aggregate consumption
[expoElas, priceElas] = computeElas( x', model, bc, x0);

%%%Plot
figure('pos', [10 10 1000 400])
subplot(1,2,1)
plot([round(expoElas{1}.firstType,2) round(expoElas{2}.firstType,2) round(expoElas{3}.firstType,2)], 'LineWidth',1.5)
title("Exposure Elasticities First Type (G1)");

subplot(1,2,2)

plot([round(expoElas{1}.secondType,2) round(expoElas{2}.secondType,2) round(expoElas{3}.secondType,2)], 'LineWidth',1.5)
title("Exposure Elasticities Second Type (G1)");
legend( { '10th Pct', '50th Pct', '90th Pct' }, 'FontSize',8 );

saveas(gcf,'HK_shockExpoG1.png')


figure('pos', [10 10 1000 400])
subplot(1,2,1)
plot([priceElas{1}.firstType priceElas{2}.firstType priceElas{3}.firstType],'LineWidth',1.5)

title("Price Elasticities First Type (G1)");

subplot(1,2,2)

plot([priceElas{1}.secondType priceElas{2}.secondType priceElas{3}.secondType],'LineWidth',1.5)
title("Price Elasticities Second Type (G1)");
legend( { '10th Pct', '50th Pct', '90th Pct' }, 'FontSize',8 );

saveas(gcf,'HK_shockPriceG1.png')

%% Step 2: Compute Shock Elasticities for Expert Consumption

%%%%Expert Consumption
model.muC = @(x) -model.muS(x) - rho;
model.sigmaC = @(x) -model.sigmaS(x);

%%%compute elasticities
[expoElas, priceElas] = computeElas( x', model, bc, x0);

%%%Plot
figure('pos', [10 10 1000 400])
subplot(1,2,1)
plot([expoElas{1}.firstType expoElas{2}.firstType expoElas{3}.firstType], 'LineWidth',1.5)

title("Exposure Elasticities First Type (G2)");


subplot(1,2,2)
plot([expoElas{1}.secondType expoElas{2}.secondType expoElas{3}.secondType], 'LineWidth',1.5)
legend( { '10th Pct', '50th Pct', '90th Pct' }, 'FontSize',8 );

title("Exposure Elasticities Second Type (G2)");

saveas(gcf,'HK_shockExpoG2.png')

figure('pos', [10 10 1000 400])
subplot(1,2,1)
plot([priceElas{1}.firstType priceElas{2}.firstType priceElas{3}.firstType],'LineWidth',1.5)
title("Price Elasticities First Type (G2)");

subplot(1,2,2)

plot([priceElas{1}.secondType priceElas{2}.secondType priceElas{3}.secondType], 'LineWidth',1.5)
legend( { '10th Pct', '50th Pct', '90th Pct' }, 'FontSize',8 );
title("Price Elasticities Second Type (G2)");
saveas(gcf,'HK_shockPriceG2.png')
