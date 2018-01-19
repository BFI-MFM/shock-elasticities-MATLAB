function [hists] = simStatDent(stateSpace, hist0, dt, T, drifts, vols)

%%%Determine dimensions, number of shocks, etc
nDims = size(stateSpace,2); gridCell = cell(1,nDims);
[gridCell{:}]= ndgrid(stateSpace{1:nDims});

if isa(vols{1},'double')
    nShocks = size(vols{1},2);
else
    nShocks = size(vols{1}(hist0(1,:)),2);
end

%%Determine endogenous state variables and create interpolants;
endoDrifts = []; exoDrifts = [];
endoVols = []; exoVols = [];
driftsFuncs = cell(nDims,1); volsFuncs = cell(nDims,1);cellInterpolants = cell(1, nShocks);
for i = 1:nDims
    
    %%%Handling drifts
    if isa(drifts{i},'function_handle')
        exoDrifts = [exoDrifts i];
        driftsFuncs{i} = drifts{i};
    else
        endoDrifts = [endoDrifts i];
        F = griddedInterpolant(gridCell{:}, reshape(drifts{i},size(gridCell{i})));
        driftsFuncs{i} = F;
    end
    
    %%%Handling vols
    if isa(vols{i}, 'function_handle')
        exoVols = [exoVols i];
        volsFuncs{i} = vols{i};
    else
        endoVols = [endoVols i]; 
        for j = 1:nShocks
            cellInterpolants{j} = griddedInterpolant(gridCell{:}, reshape(vols{i}(:,j),size(gridCell{i})));
        end
        volsFuncs{i} = eval(['@(x)[',sprintf('cellInterpolants{%i}(x) ',1:length(cellInterpolants)),']']);
    end
end

%%Initialize drift cell and 
driftsAll = eval(['@(x)[',sprintf('driftsFuncs{%i}(x) ',1:length(driftsFuncs)),']']);
volsAll = eval(['@(x)[',sprintf('volsFuncs{%i}(x); ',1:length(volsFuncs)),']']);





%%%set bounds
upperBounds = cellfun(@(x) max(x(:)), gridCell);
lowerBounds = cellfun(@(x) min(x(:)), gridCell);

%%%set up cell to store simulation data
hists = cell(size(hist0,1),1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%Start Simulations%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parfor i = 1:size(hist0,1)
    
    %%%Preallocate for history
    hist = zeros(T,nDims);

    %%%Fill in first point
    hist(1,:) = hist0(i,:);
    
    for t = 2:ceil(T/dt)
        %%create shock
        shock = normrnd(0,sqrt(dt), 1, nShocks);
        hist(t,:) = max(min(hist(t-1,:) + driftsAll(hist(t-1,:)) * dt + (volsAll(hist(t-1,:)) * shock')', upperBounds), lowerBounds)';
    end
    
    hists{i} = hist;
end



