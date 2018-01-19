function [] = checkInputs(domain,model,bc,x0)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Function that checks inputs are correct%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%Making sure that sigmaX is a cell
if ~isa(model.sigmaX, 'cell')
    error('model.sigmaX should be a cell');
else 
    nShocks = size(model.sigmaX{1},2);
    nDims = size(model.sigmaX,2);
    totalGrid = size(model.sigmaX{1}, 1);
    for i = 1:nDims
        if ~(size(model.sigmaX{i},1) == totalGrid)
            error('Number of rows inconsistent in model.sigmaX.')
        end
    end
end

%%%%Making sure dimensions are correct
if ~( size(model.muX,2) == nDims) 
    error('model.muX does not match up with the dimensions provided in model.sigmaX.')
elseif ~( size(model.muX,1) == totalGrid )
    error('model.muX has an unequal number of rows versus model.sigmaX.')
end

%%%%Making sure the number of shocks are correct
for i = 1:nDims
    if ~( size(model.sigmaX{i}, 2) == nShocks) 
        error('Number of shocks inconsistent in model.sigmaX')
    end
end

if ~( size(model.sigmaS, 2) == nShocks)
    error('Number of shocks in model.sigmaS does not match up with model.sigmaX')
elseif ~( size(model.sigmaS, 1) == totalGrid)
    error('model.sigmaS has an unequal number of rows versus model.sigmaX.')
end

if ~( size(model.sigmaC, 2) == nShocks)
    error('Number of shocks in model.sigmaC does not match up with model.sigmaX')
elseif ~( size(model.sigmaC, 1) == totalGrid)
    error('model.sigmaC has an unequal number of rows versus model.sigmaX.')
end

%%%Checking domain
if ~( size(domain,2) == nDims) 
    error('Number of dimensions in domain inconsistent with model.sigmaX or model.muX');
elseif ~( size(domain, 1) == totalGrid)
    error('domain has an unequal number of rows versus model.sigmaX.')
end

%%%Checking x0
if ~( size(x0,2) == nDims) 
    error('Number of dimensions in x0 inconsistent with model.sigmaX or model.muX');
end

%%%Checking boundary conditions

if ~isfield(bc, 'a0')
    error('bc missing a0');
elseif ~(numel(bc.a0) == 1)
    error('bc.a0 wrong size');
end

if ~isfield(bc, 'level')
    error('bc missing level');
elseif ~(numel(bc.level) == nDims)
    error('bc.level wrong size');
end

if ~isfield(bc, 'first')
    error('bc missing first');
elseif ~(numel(bc.first) == nDims)
    error('bc.first wrong size');
end

if ~isfield(bc, 'second')
    error('bc missing second');
elseif ~(numel(bc.second) == nDims)
    error('bc.second wrong size');
end

if ~isfield(bc, 'natural')
    error('bc missing natural');
elseif ~isa(bc.natural, 'logical')
    error('bc.natural must be a boolean'); 
elseif ~(numel(bc.a0) == 1)
    error('bc.natural wrong size');
end

end
