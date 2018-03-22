%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%Function to compute shock elasticities%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [varargout] = computeElas(domain,modelInput,bc,x0,optArgs) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%Inputs%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% domain - grid for state variables
% modelInput - struct that has muC, sigmaC, muX, sigmaX, T, dt
% bc - struct that has a0, level, first, and second
% x0 - starting points
% optArgs - struct that has optional arguments
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Step 0 Preliminaries

%%%Convert function handles to numerical values
model.muX = modelInput.muX(domain);
model.muC = modelInput.muC(domain);
model.muS = modelInput.muS(domain);

model.sigmaX = cell(1, size(model.muX,2));
for i = 1:size(model.sigmaX,2)
    %%
    model.sigmaX{i} = modelInput.sigmaX{i}(domain);
end
model.sigmaC = modelInput.sigmaC(domain);
model.sigmaS = modelInput.sigmaS(domain);

model.T = modelInput.T; model.dt = modelInput.dt;

%%%Check inputs
checkInputs(domain,model,bc,x0);

%%Create state space
stateSpace = struct;
stateSpace.space = domain;
stateSpace.S = size(domain,1);
stateSpace.N = size(domain,2);

%%Preallocate for outputs
totalStart = size(x0,1);
priceElas = cell(1,totalStart);
out = cell(1, size(model.sigmaC,2) + 3 );

%%Parse optional arguments
%%%%Default

if nargin > 4
    %%User has input optional argument
    if ~isfield(optArgs, 'usePardiso')
        %%if user didn't specify usePardiso, set it to default (false)
        optArgs.usePardiso = false;
    end
    if ~isfield(optArgs, 'priceElas')
        %%if user didn't specify usePardiso, set it to default (false)
        optArgs.priceElas = true;
    end
else
    %%User did not input optional argument; set to default
    optArgs.usePardiso = false;
    optArgs.priceElas = true;
end


%% Step 1 Compute Shock Exposure Elasticities (Both Types)

%%%% Step 1.1 Solve Feynman Kac equation
disp('COMPUTING SHOCK EXPOSURE ELASTICITIES')

phi0 = ones(stateSpace.S, 1); %%%RHS for first type
model.RHS = [phi0 model.sigmaC]; %%%RHSs for secodn type

if optArgs.usePardiso
    [out{:}] = constructSolvePardiso(domain, model, bc);
else
    [out{:}] = constructSolve(domain, model, bc);
end


disp('Finished solving Feynman Kac equation...')

stateSpace.dVec = out{end-1}; stateSpace.increVec = out{end};


expoElas = computeElasSub(out, stateSpace, model, x0);
varargout{1} = expoElas;

%% Step 2 Compute Shock Price Elasticities (Both Types)
if optArgs.priceElas
    disp('COMPUTING SHOCK PRICE ELASTICITIES')

    %%%%Add in SDF
    model.muC = model.muS + model.muC;
    model.sigmaC = model.sigmaS + model.sigmaC;
    model.RHS = [phi0 model.sigmaC];

    %%%%Construct linear system
    if optArgs.usePardiso
        [out{:}] = constructSolvePardiso(domain, model, bc);
    else
        [out{:}] = constructSolve(domain, model, bc);
    end

    disp('Finished solving Feynman Kac equation...')

    costElas = computeElasSub(out, stateSpace, model, x0);


    %%%%Compute shock price
    for s = 1:totalStart
        priceElas{s}.firstType = expoElas{s}.firstType - costElas{s}.firstType;
        priceElas{s}.secondType = expoElas{s}.secondType - costElas{s}.secondType;
    end
    varargout{2} = priceElas;
end

end
