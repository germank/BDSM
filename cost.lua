require 'torch'
require 'utils'
require 'math'
require 'table'
tablex = require 'pl/tablex'

TensorType = torch.Tensor

function copy(x)
    return TensorType(x:size()):copy(x)
end

hist_theta = nil

reported_t = nil
function new_report()
   reported_t = {}
end
function get_report()
    return reported_t
end
function report(measure, val, _reported_t)
    _reported_t = _reported_t or reported_t
    if not _reported_t[measure] then
        _reported_t[measure] = {}
    end
    _reported_t[measure][#reported_t[measure]+1] = val
--    print (measure..': '..val)
end

function show_report(_reported_t)
    _reported_t = _reported_t or reported_t
    --require 'os'.execute('clear')
    for measure, vals in tablex.sort(_reported_t) do
       -- print (measure..': '.. vals[#vals])
    end

    --print '------------------------'
    io.stdout:write('.')
    io.stdout:flush()
end

function unpack_theta(theta)
    local h=hiddenSize
    local v=visibleSize
    if layers == 2 then
        local W = theta[{{1,h*v}}]:reshape(h, v)
        local b = theta[{{h*v+1,h*v+h}}]:reshape(
                           h,1)
        local W2 = theta[{{h*v+h+1,h*v+h+h*(h+1)}}]:reshape(h+1, h)
        local b2 = theta[{{h*v+h+h*(h+1)+1,h*v+h+h*(h+1)+h+1}}]:reshape(
                           h+1,1)
        return W,b,W2,b2

    else
        local W = theta[{{1,h*v}}]:reshape(h, v)
        local b = theta[{{h*v+1,h*v+h}}]:reshape(
                       h,1)
        return W,b
    end
end

function papernic_cost(theta, visibleSize, hiddenSize,
                            lambda, sparsityParam, beta, hs, cs, y, compute_grad)
    -- visibleSize: the number of input units 
    -- hiddenSize: the number of hidden units
    -- lambda: weight decay parameter
    -- sparsityParam: The desired average activation for the hidden units (denoted in the lecture
    --                           notes by the greek alphabet rho, which looks like a lower-case "p").
    -- beta: weight of sparsity penalty term
      

    compute_grad = compute_grad ==nil and true or compute_grad
    hs = hs:t()
    cs = cs:t()
    y = y:t()
    
    local l1_W,l1_b,l2_W,l2_b
    if layers then
        local W,b,W2,b2 = unpack_theta(theta)
        l1_W = W
        l1_b = b
        l2_W = W2
        l2_b = b2
    else
        local W,b = unpack_theta(theta)
        l1_W = W
        l1_b = b
        l2_W = W
        l2_b = b
    end

    local pos_ind = y
    local neg_ind = y*-1 + 1

    local n1 = pos_ind:sum(2)[{1,1}]
    local n0 = neg_ind:sum(2)[{1,1}]

    local w1 = (n1+n0)/(2*n1)
    local w0 = (n1+n0)/(2*n0)
    
    -- Cost and gradient variables
    -- Here, we initialize them to zeros. 
    local cost = 0


    --[[ FORWARD PASS: Layer 1 ]]--
    local l1_hs_in = hs
    local l1_cs_in = cs
    local l1_hs_out = TensorType(l1_W:size()[1], l1_hs_in:size()[2]):zero()
    l1_hs_out:addmm(l1_W, l1_hs_in)
    l1_hs_out:add(l1_b:expandAs(l1_hs_out))
    l1_hs_out = sigmoid(l1_hs_out, t) --inplace??
    local l1_cs_out = l1_W * l1_cs_in
    l1_cs_out:add(l1_b:expandAs(l1_cs_out))
    l1_cs_out = sigmoid(l1_cs_out, t)



    local l2_hs_in, l2_cs_in, l2_hs_out, l2_cs_out

    if layers == 2 then
        l2_hs_in = l1_hs_out
        l2_cs_in = l1_cs_out
        l2_hs_out = TensorType(l2_W:size()[1], l2_hs_in:size()[2]):zero()
        l2_hs_out:addmm(l2_W, l2_hs_in)
        l2_hs_out:add(l2_b:expandAs(l2_hs_out))
        l2_hs_out = sigmoid(l2_hs_out, t) --inplace??
        l2_cs_out = l2_W * l2_cs_in
        l2_cs_out:add(l2_b:expandAs(l2_cs_out))
        l2_cs_out = sigmoid(l2_cs_out, t)
    else
        l2_hs_in = l1_hs_in
        l2_cs_in = l1_cs_in
        l2_hs_out = l1_hs_out
        l2_cs_out = l1_cs_out
    end


    --[[ FORWARD PASS: Layer 2 ]]--
    --softmax(1-hs2,cs2)
    local e_h, e_c, e
    if l2type == 'softmax' then
        e_h = -l2_hs_out + 1
        e_h:mul(M)
        e_h:exp()
        e_c = copy(l2_cs_out)
        e_c:mul(M)
        e_c:exp()
        e = e_h + e_c
        e:log()
        e:div(M)
    elseif l2type == 'boolean' then
        e = torch.cmul(l2_hs_out, l2_cs_out)
        e:mul(-1)
        e:add(l2_cs_out)
        e:add(l2_hs_out)
    else
        error("Invalid layer 2 type: " .. l2type)
    end



    --[[ FORWARD PASS: Layer 3 ]]--
    local h
    if hyptype == 'min'then
        h = -e
        h:mul(M2)
        h:exp()
        h = -((h:sum(1):log())/M2)
    elseif hyptype == 'avg' then
        h = e:mean(1)
        h = sigmoid(h, t, d)
    elseif hyptype == 'avg0' then
        h = e:mean(1)
    elseif hyptype == 'prod' then
        h = e:cumprod(1)[{-1,{}}]:reshape(1,e:size()[2])
    else
        error ("Invalid hypothesis type: "..hyptype)
    end

    --[[
    gnuplot.figure()
    gnuplot.pngfigure('histogram.png')
    gnuplot.hist(h)
    gnuplot.plotflush()
    gnuplot.closeall()]]--
    

    report('Mean h (y=1)',torch.cmul(h, y):sum()/torch.sum(y))
    report('Mean h (y=0)',torch.cmul(h, y*-1+1):sum()/torch.sum(y*-1+1))

    if losstype == 'mse' then
        cost = torch.mean(torch.cmul(torch.mul(y,w1),torch.pow(torch.norm(h - y, 2, 1), 2)))/2 --norm2 over dimension 1
        cost = cost + torch.mean(torch.cmul(torch.mul(-y+1,w0),torch.pow(torch.norm(h - y, 2, 1), 2)))/2 --norm2 over dimension 1
        report('SMSE', cost*2)
    elseif losstype == 'logl' then
        cost = torch.mean(torch.mul(torch.cmul(torch.mul(y,w1),torch.log(h+logl_marg)) + torch.cmul(torch.mul(-y+1,w0),torch.log(-h + 1 + logl_marg)), -1))
    end


    local all_mse, neg_mse, pos_mse =  mse(h, y,w0,w1)
    report('MSE', all_mse)
    report('MSE_0', neg_mse)
    report('MSE_1', pos_mse)
    if regularization then
        cost = cost + lambda / 2 * (torch.pow(l1_W,2)):sum()--regularization
        if layers == 2 then
            cost = cost + lambda / 2 * (torch.pow(l2_W,2)):sum()--regularization
        end
    end


    local d4, d3, d2a, d2b
    if compute_grad then
    --[[ BACKWARD PASS: Layer 3 ]]--
        if losstype == 'mse' then
            d4 = torch.cmul(torch.mul(y,w1),h - y)
            d4 = d4 + torch.cmul(torch.mul(-y+1,w0),h - y)
        elseif losstype == 'logl' then
            d4 = -torch.cmul(torch.mul(y,w1), torch.pow(h+logl_marg, -1)) + torch.cmul(torch.mul(-y+1,w0), torch.pow(-h + 1+logl_marg, -1))
        end
        if hyptype == 'min' then
            d3 = -e
            d3:mul(M2)
            d3:exp()
            d3:cdiv(d3:sum(1):expandAs(d3))
        elseif hyptype == 'avg' then
            d3 = -h
            d3:add(1)
            d3:cmul(h)
            d3:div(t)
            d3:div(e:size()[1])
            d3 = copy(d3:expandAs(e))
        elseif hyptype == 'avg0' then
            d3 = torch.Tensor(h:size()):fill(1)
            d3:div(e:size()[1])
            d3 = copy(d3:expandAs(e))
        elseif hyptype == 'prod' then
            local H = e:size()[1]
            local M = e:size()[2]
            d3 = torch.Tensor(e:size()):fill(1)
            for i=1,M do
                for j=1,H do
                    for k=1,H do
                        if k ~= j then
                            d3[{j,i}] = d3[{j,i}] * e[{k,i}]
                        end
                    end
                end
            end
            --[[
            local tmp = e:t():reshape(M,H,1):expand(M,H,H):clone()
            for i=1,H do for j=1,M do tmp[{j,i,i}] = 1 end end
            tmp = tmp:cumprod(2)
            d3 = tmp[{{},-1, {}}]:t():reshape(H,M)]]--
        end
        d3:cmul(d4:expandAs(d3))
        

    --[[ BACKWARD PASS: Layer 2 ]]--

        if l2type == 'softmax' then
            d2a = e_h + e_c
            d2a:pow(-1)
            d2a:cmul(d3)
            d2b = copy(d2a)
            d2a:cmul(e_h)
            d2a:mul(-1)
            d2b:cmul(e_c)
        elseif l2type == 'boolean' then
            d2a = l2_cs_out:clone()
            d2a:mul(-1)
            d2a:add(1)
            d2a:cmul(d3)
            d2b = l2_hs_out:clone()
            d2b:mul(-1)
            d2b:add(1)
            d2b:cmul(d3)
        end
    end

    --[[ SPARSENESS ]]--
    --
        --[[
        local sp_h = copy(hs2)
        sp_h:mul(M)
        sp_h:exp()
        local sp_c = e_c --already calculated
        local s = sp_c + sp_h
        s:log()
        s:div(M)
        ]]--
        --sparseness measurements
        local rho_hat = torch.mean(l2_hs_out, 2) 
        local rho_hat_t = torch.sum(l2_hs_out, 1) 
        report ('min act',torch.min(rho_hat))
        report ('max act',torch.max(rho_hat))
        report ('mean act',torch.mean(rho_hat))
        report ('std dev num active',torch.std(rho_hat_t))
        report ('mean num active',torch.mean(rho_hat_t))
    --activated units
    local KLgrad
    local d2kla,d2klb
    if sparsity then
        local rho = sparsityParam
        local KL = torch.log(torch.pow(rho_hat+kl_marg, -1) * (rho+kl_marg)) * (rho+kl_marg) +  torch.log(torch.pow(-rho_hat+1+kl_marg, -1)*(-rho+1+kl_marg)) * (-rho+1+kl_marg) --how can it be implemented w/out pow?
        --print (KL:t())
        cost = cost + beta * torch.sum(KL)
        if compute_grad then
            KLgrad = -torch.pow(rho_hat+kl_marg, -1) * (rho+kl_marg) +  torch.pow(-rho_hat+1+kl_marg, -1)*(-rho+kl_marg+1) --how can it be implemented w/out pow?

            d2kla = TensorType(l2_hs_out:size()):fill(1)
            d2klb = TensorType(l2_hs_out:size()):fill(0)

            --[[
            d2kla = sp_h + sp_c
            d2kla:pow(-1)
            d2klb = copy(d2kla)
            d2kla:cmul(sp_h)
            d2klb:cmul(sp_c)]]--
        end
    end

    --[[ SPARSENESS END ]]--


    --[[ BACKWARD PASS: Layer 1-2 ]]--
    local d1a, d1b
    local d1kla, d1klb
    if compute_grad then
        d1a = copy(l2_hs_out)
        d1a:mul(-1)
        d1a:add(1)
        d1a:cmul(l2_hs_out)
        d1a:div(t)
        
        d1b = copy(l2_cs_out)
        d1b:mul(-1)
        d1b:add(1)
        d1b:cmul(l2_cs_out)
        d1b:div(t)

        --sparse
        if sparsity then
            d1kla = torch.cmul((KLgrad * beta):expandAs(d1a),torch.cmul(d2kla, d1a))
            d1klb = torch.cmul((KLgrad * beta):expandAs(d1b),torch.cmul(d2klb, d1b))
        end
    end


    local grad
    if compute_grad then
        d1a:cmul(d2a)
        d1b:cmul(d2b)

    
        local Wgrad, bgrad
        Wgrad  = d1a * l2_hs_in:t() + d1b  * l2_cs_in:t()
        --sparsity
        if sparsity then
            Wgrad:add( d1kla * l2_hs_in:t() + d1klb * l2_cs_in:t())
        end
        Wgrad:div(l2_hs_in:size()[2])
        if regularization then
            Wgrad:add(l2_W * lambda) --regularization
        end
        bgrad = (d1a + d1b):mean(2)
        if sparsity then 
            bgrad:add((d1kla + d1klb):mean(2))
        end

        grad = torch.cat(flatten(Wgrad), flatten(bgrad))
    end

    --[[ BACKWARD PASS: Layer 1-1 ]]--
    if layers ==2 then
        local d11a, d11b
        local d11kla, d11klb
        if compute_grad then
            d11a = copy(l1_hs_out)
            d11a:mul(-1)
            d11a:add(1)
            d11a:cmul(l1_hs_out)
            d11a:div(t)
            
            d11b = copy(l1_cs_out)
            d11b:mul(-1)
            d11b:add(1)
            d11b:cmul(l1_cs_out)
            d11b:div(t)

            if sparsity then
                d11kla = torch.cmul(l2_W:t() * d1kla,d11a)
                d11klb = torch.cmul(l2_W:t() * d1klb,d11b)
            end
        end


        if compute_grad then
            d11a:cmul(l2_W:t()*d1a)
            d11b:cmul(l2_W:t()*d1b)

        
            local Wgrad, bgrad
            Wgrad  = d11a * l1_hs_in:t() + d11b  * l1_cs_in:t()
            if sparsity then
                Wgrad:add( d11kla * l1_hs_in:t() +  d11klb * l1_cs_in:t())
            end
            Wgrad:div(l1_hs_in:size()[2])
            if regularization then
                Wgrad:add(l1_W * lambda) --regularization
            end
            bgrad = (d11a + d11b):mean(2)
            if sparsity then 
                bgrad:add(( d11kla + d11klb):mean(2))
            end

            grad = ncat(flatten(Wgrad), flatten(bgrad),grad)
        end
    end
    report('Norm theta', torch.norm(theta))
    --collect the garbage before we leave
    report ('cost',cost)

    if save_hist and compute_grad then
        print ('saving history ('..(#hist_theta+1)..')')
        --[[
        if grad_hist then
            grad_hist[#grad_hist+1] = copy(grad)
        else
            grad_hist = {copy(grad)}
        end
        ]]--
        if hist_size and #hist_theta >= hist_size then
            table.remove(hist_theta, 1)
        end
        hist_theta[#hist_theta+1] = copy(theta)
    end
    collectgarbage()
    show_report()
    return cost, grad
end
---------------------------------------------------------------------
-- Here's an implementation of the sigmoid function, which you may find useful
-- in your computation of the costs and the gradients.  This inputs a (row or
-- column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigmoid(x, t, d)
   --GermanK: I think there should be a more efficient way to write it
   --but I'm not sure how to write it.
   --return x:apply(function(x_i) return 1 / (1 + math.exp(-x_i)) end);
   d = d or 0
   return torch.pow(torch.exp(-(x-d)/t ) + 1, -1)
end

function binarize(x)
    if x< 0.5 then return 0 else return 1 end
end

function mse(x, y,w0, w1)
    w0 = w0 or 1
    w1 = w1 or 1
    x = copy(x)
    x:apply(binarize)

    return torch.mean(torch.cmul(y*w1,torch.pow(x - y, 2))+torch.cmul((y*-1+1)*w0,torch.pow(x - y, 2))),
    --FIXME: unweighted approximations
        torch.pow(torch.cmul(x-y, y), 2):sum()/torch.sum(y),
        torch.pow(torch.cmul(x-y, y*-1+1), 2):sum()/torch.sum(y*-1+1)
end


