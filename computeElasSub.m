%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%Function to compute shock elasticities%%%%%%
%%%%%%%%%%%%%%%%after solving PDE%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [elasCell] = computeElasSub(out,stateSpace,model,x0) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%Inputs%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% out - solution to Feynmann Kac PDE
% stateSpace - grid for state variables
% model - struct that has muC, sigmaC, muX, sigmaX, T, dt
% bc - struct that has a0, level, first, and second
% x0 - starting points
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 0 Preliminaries
eps = .000001; %%epsilon
numStarts = size(x0,1);

%% Step 1 Compute Shock Elasticities (First Type)

%%%% Step 1.1 Interpolate for starting points
%%%%%Create vector to interpolate
%%x_eval = zeros(2*stateSpace.N + 1, stateSpace.N); //to be removed
x_evals = cell(numStarts,1);
for i = 1:numStarts
    x_evals{i} = zeros(2*stateSpace.N + 1, stateSpace.N);
    for n = 0:2:2*(stateSpace.N - 1)
        x_evals{i}(n + 1,:) = x0(i,:);
        x_evals{i}(n + 2,:) = x0(i,:);
        x_evals{i}(n + 1, floor(n / 2) +1 ) = x0(i,  floor(n / 2)+1  ) + eps;
        x_evals{i}(n + 2,  floor(n / 2)+1  ) = x0(i,  floor(n / 2)+1  ) - eps;
    end
    x_evals{i}(end,:) = x0(i,:);
end


%%%%Load phiAll into interpolator
phiAll = out{1}; gridVectors = arrayfun(@(x) unique(stateSpace.space(:,x)), 1:stateSpace.N, 'UniformOutput', false);
gridCell = cell(1,stateSpace.N);
[gridCell{:}] = ndgrid(gridVectors{1:stateSpace.N});
res = cell(numStarts,1);
exp1s = cell(numStarts,1);

for i = 1:numStarts
    res{i} = zeros(size(x_evals{1},1), model.T);
    for t = 1:model.T
        F = griddedInterpolant(gridCell{:}, reshape(phiAll(:,t),size(gridCell{1})) );
        res{i}(:,t) = F(x_evals{i});
    end
    exp1s{i} = res{i}(end,:);
end

%%%
disp('Finished interpolating...')

%%%%Step1.2 Compute shock exposure elasticities

%%%%%%Compute derivs
derivsCell = cell(numStarts,1);
for i = 1:numStarts
    derivsCell{i} = zeros(stateSpace.N, model.T);
    for n = 1:2:size(x_evals{1},1)-1
        derivsCell{i}( (n +1) / 2 , :) = ( ( res{i}(n, :) - res{i}(n + 1, :) )  / (2*eps) ) ./ exp1s{i};
    end
end

%%%%%Compute vols at x0
elasCell = cell(1,numStarts);
elas = struct;
allSig = [model.sigmaX{:}];
res = zeros(numStarts, size(allSig,2) + size(model.sigmaC,2));
vals = [allSig model.sigmaC];
vals(~isfinite(vals)) = sign( vals(~isfinite(vals)) ) * 999999;
vals = fillmissing(vals, 'nearest');

for j = 1:size(allSig,2) + size(model.sigmaC,2)
    F = griddedInterpolant(gridCell{:}, reshape(vals(:,j),size(gridCell{1})) );
    for i = 1:numStarts
        res(i,j) = F(x0(i,:));
    end
end
    
for i = 1:numStarts
    elasCell{i} = elas;
    res1 = res(i,1:size(allSig,2));
    res2 = res(i,size(allSig,2)+1:end);
    elasCell{i}.firstType = (reshape(res1, size(model.sigmaX{1},2),stateSpace.N) * derivsCell{i})' + repmat(res2,model.T,1);
end

disp('Finished computing first type.')
%% Step 2 Compute Shock Elasticities (Second Type)

%%%%Step 2.1 Solve Feynmann Kac given new phi0
disp('Computing second type...')

%%%%Step 2.2 Compute shock elasticities
exp2s = cell(numStarts,1);

for i = 1:numStarts
    exp2s{i} = zeros(size(model.sigmaC,2), model.T);
end

for s = 1:size(model.sigmaC,2)
    tmp = out{s + 1};
    for t = 1:model.T
        F = griddedInterpolant(gridCell{:}, reshape(tmp(:,t),size(gridCell{1})) );
        for i = 1:numStarts
            exp2s{i}(s,t) = F(x_evals{i}(end,:));
        end
    end
end

for i = 1:numStarts
    elasCell{i}.secondType  = exp2s{i}' ./ exp1s{i}';
end

disp('Finished computing second type.')

end
