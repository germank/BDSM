require 'utils'
require 'config'
function initializeParameters(hiddenSize, visibleSize)
    ---- Initialize parameters randomly based on layer sizes.
    local r  = math.sqrt(6) / math.sqrt(hiddenSize+visibleSize+1)   -- we'll choose weights uniformly from the interval [-r, r]
    local W1 = torch.rand(hiddenSize, visibleSize) * 2 * r - r

    local b1 = torch.Tensor(hiddenSize, 1):zero()

    -- concatenate parameters
    local theta = ncat(W1:reshape(hiddenSize*visibleSize), b1:t()[1])


    if layers ==2 then
        local W2 = torch.rand(hiddenSize, hiddenSize+1) * 2 * r - r
        local b2 = torch.Tensor(hiddenSize+1, 1):zero()
        theta = ncat(theta, W2:reshape((hiddenSize+1)*hiddenSize), b2:t()[1])

    end
    return theta
end

