#!/usr/bin/env lua
require 'torch'
require 'config'
require 'datasets'
require 'cost'
require 'optim'
require 'utils'
require 'gnuplot'
require 'lfs'
require 'table_save'
require 'initialize_parameters'

function optimize(minFunc, func, theta, options, maxiter)
    print ('Optimizing...')
    for i =1,maxiter do
        local opttheta, cost = minFunc( func, 
                               theta, options);
        --opttheta is actually the same vector as theta, but just for clarity:
        theta = opttheta
    end
    return theta
end


require 'memdebug'


command = nil
init_theta_file = nil

function get_theta_filename()
    local out_filename = 'theta_'..get_desc()..'.th7'
    return 'saved_params/'..out_filename
end


function get_theta(pick_best_test, save)
    local theta_filename = get_theta_filename()
    local theta, bestmse
    if not pcall(function()
        theta = torch.load(theta_filename)
        print ("Theta loaded from ".. theta_filename)
        end) then
        print ("Cache not found: ".. theta_filename)
        theta, bestmse = train_cv_theta(pick_best_test, save)
    end
    return theta, bestmse
end


function get_best_mse(tr_reports, ts_reports, pick_best_test)
    local best= nil
    local reports
    if pick_best_test and test_positive_ds and test_negative_ds then
        reports = ts_reports
    else
        reports = tr_reports
    end
    for i=1,#reports do
        local mse = reports[i]['MSE'][#reports[i]['MSE']]
        if best then
            local best_mse = reports[best]['MSE'][#reports[best]['MSE']]
            if mse < best_mse then
                best = i
            end
        else
            best = i
        end
    end
    local nbest = #reports[best]['MSE']
    return best, reports[best]['MSE'][nbest]
end

function train_cv_theta(pick_best_test, save)
    pick_best_test = false
    train_positive_ds, train_negative_ds, test_positive_ds, test_negative_ds = unpack(datasets)
    local tr_reports, ts_reports = {}, {}
    --local thetas = {}
    local best_theta = nil
    local bestval = nil
    for i=1,Nretrain do
        hist_theta = {}
        new_report()
        local theta = train_theta()
        --thetas[#thetas+1] = opttheta
        local tr_report = get_report()
        tr_reports[#tr_reports+1] = tr_report
        if test_positive_ds and test_negative_ds then
            new_report()
            test_theta(hist_theta)
            local ts_report = get_report()
            ts_reports[#ts_reports+1] = ts_report
            --save_reports(tr_report, ts_report)
        end
            
        --Don't keep retraining if we already found a good fit of the data
        local best_i, bestval_i = get_best_mse(tr_reports, ts_reports, false)
        if bestval_i  < max_train_err/100.0 then
            bestval = bestval_i
            best_theta = theta
            print('Found training MSE='..bestval_i)
            break
        end
    end
    --[[
    local best, bestval = get_best_mse(tr_reports, ts_reports, pick_best_test)
    local nbest = #tr_reports[best]['MSE']
    print('Best training MSE for best theta: '..tr_reports[best]['MSE'][nbest])
    if ts_reports[best] then
        for x in pairs(ts_reports[best]) do print (x) end
        local nbest_ts = #ts_reports[best]['MSE']
        print('Best testing MSE for best theta: '..ts_reports[best]['MSE'][nbest_ts])
        save_reports(tr_reports[best], ts_reports[best])
    end
    ]]--
    if save then
        local out_filename = get_theta_filename()
        print ('saving theta to '..out_filename)
        torch.save(out_filename, best_theta)
    end
    return best_theta, bestval
end
function train_theta()
    local options = {}
    options.maxIter = maxIter     -- Maximum number of iterations of L-BFGS to run 
    options.verbose = true
    train_positive_ds, train_negative_ds, test_positive_ds, test_negative_ds = unpack(datasets)
    local hs, cs, y = load_dataset_vectors(train_positive_ds, train_negative_ds, maxDSLen)
    local papernic_cost_theta = function(x) return papernic_cost(x, 
                                   visibleSize, hiddenSize, 
                                   lambda, sparsityParam, 
                                   beta, hs, cs, y) end
    local theta
    if (init_theta_file) then
        theta = torch.load(init_theta_file)
    else 
        theta = initializeParameters(hiddenSize, visibleSize);
    end
    local opttheta = optimize(optim.lbfgs, papernic_cost_theta, theta, options, repetitions)
    return opttheta
end

function check_grad()
    train_positive_ds, train_negative_ds, test_positive_ds, test_negative_ds = unpack(datasets)
    local hs, cs, y = load_dataset_vectors(train_positive_ds, train_negative_ds)
    local theta = initializeParameters(hiddenSize, visibleSize);
    local cost, grad = papernic_cost(theta, visibleSize, hiddenSize, lambda, 
                                sparsityParam, beta, hs, cs, y)
    require 'compute_numerical_gradient'
    local numgrad = computeNumericalGradient( function(x) return 
                                                papernic_cost(x, visibleSize, 
                                                  hiddenSize, lambda, 
                                                  sparsityParam, beta, 
                                                  hs, cs, y, false) end, theta);

    -- Use this to visually compare the gradients side by side
    print(torch.cat(torch.cat(grad, numgrad, 2), grad-numgrad, 2))
    print  (torch.cdiv(numgrad,grad))
    -- Compare numerically computed gradients with the ones obtained from backpropagation
    local diff = torch.norm(numgrad-grad)/torch.norm(numgrad+grad);
    print(diff); 
end

function plot_cost(maxN)
    local theta = initializeParameters(hiddenSize, visibleSize);
    train_positive_ds, train_negative_ds, test_positive_ds, test_negative_ds = unpack(datasets)
    local hs, cs, y = load_dataset_vectors(train_positive_ds, train_negative_ds)
    local papernic_cost_2d = function(x1, x2) 
        local x = copy(theta)
        local dim1 = plot_dim1
        local dim2 = plot_dim2 
        print(hiddenSize)
        print(visibleSize)
        print(x:size())
        x[dim1] = x1
        x[dim2] = x2
        return papernic_cost(x, 
                           visibleSize, hiddenSize, 
                           lambda, sparsityParam, 
                           beta, hs, cs, y, false) 
    end
    local x=torch.linspace(-plot_limx,plot_limx,plot_points)
    local costy = torch.Tensor(x:size()[1], x:size()[1]):zero()
    local tcosty = {}
    for i = 1,x:size()[1] do
        for j = 1,x:size()[1] do
            print (x:size())
            print (i.."---"..j)
            print (x:size())
            costy[i][j] = papernic_cost_2d(x[i], x[j])
            tcosty[#tcosty+1] = {x[i], x[j], costy[i][j]}
        end
    end
    local costx1 = torch.Tensor(costy:size())
    local costx2 = torch.Tensor(costy:size())
    for i=1,costx1:size(1) do costx1:select(1,i):fill(x[i]) end
    for i=1,costx2:size(2) do costx2:select(2,i):fill(x[i]) end
    local m1 = torch.min(costy)
    local M1 = torch.max(costy)
    costy:add(-m1)
    costy:div((M1-m1))
    gnuplot.splot({costx1,costx2,costy})
    gnuplot.pngfigure('cost_func.png')
    gnuplot.splot({costx1,costx2,costy})
    gnuplot.plotflush()
    gnuplot.closeall()
end

function run_cost_function(theta, hs, cs, y)
    papernic_cost(theta, visibleSize, hiddenSize, lambda, 
        sparsityParam, beta, hs, cs, y, false)
end

function predict(theta, pos_ds, neg_ds, maxN)
    local W, b,W2,b2 = unpack_theta(theta)

    local revindex, vectors = load_vectors()
    local pos_ws = load_dataset(pos_ds, revindex, maxN)
    local neg_ws = load_dataset(neg_ds, revindex, maxN)
    local hs, cs, y = load_dataset_vectors(pos_ds, neg_ds)
    hs = hs:t()
    cs = cs:t()
    y = y:t()

    hs = W * hs  + b:expand(b:size()[1], hs:size()[2])
    cs = W * cs  + b:expand(b:size()[1], cs:size()[2])

    if layers == 2 then
        hs = sigmoid(hs, t)
        cs = sigmoid(cs, t)
        hs = W2 * hs  + b2:expand(b2:size()[1], hs:size()[2])
        cs = W2 * cs  + b2:expand(b2:size()[1], cs:size()[2])

    end


    hs:apply(function(x) if x <= 0 then return 0 else return 1 end end)
    cs:apply(function(x) if x <= 0 then return 0 else return 1 end end)

    --active units in the intersection
    local inter = torch.cmul(hs, cs)
    local sum_inter = inter:t():sum(2)
    --active units in the lhs
    local sum_hs = hs:t():sum(2)
    --active units in the rhs
    local sum_cs = cs:t():sum(2)
    local ret = {}
    for i=1,y:size()[2] do
        local p
        if y[{1,i}] == 1 then
            p = pos_ws[i]
        else
            p = neg_ws[i-#pos_ws]
        end
        local id = table.concat(p, "&")
        ret[#ret+1] = {id, y[{1,i}], sum_inter[{i,1}], sum_hs[{i,1}], sum_cs[{i,1}]}
    end
    return ret
end

function test_theta(hist_theta, test_ds)
    if test_ds then
        test_positive_ds, test_negative_ds = unpack(test_ds)
    else
        train_positive_ds, train_negative_ds, test_positive_ds, test_negative_ds = unpack(datasets)
    end

    reported_t = {}
    if hist_theta and #hist_theta > 0 then 
        local cv_hs, cv_cs, cv_y = load_dataset_vectors(test_positive_ds, test_negative_ds, maxDSLen)
        
        for i=1,#hist_theta do
            run_cost_function(hist_theta[i], cv_hs, cv_cs, cv_y, false);
        end
    end
end

function save_reports(tr_reported_t, ts_reported_t)
    for k in pairs(tr_reported_t) do
        local l = math.min(#tr_reported_t[k], #ts_reported_t[k])
        lfs.mkdir('out/png/'..k)
        lfs.mkdir('out/txt/'..k)
        gnuplot.pngfigure('out/png/'..k..'/'..k..'_'..desc..'.png')
        gnuplot.plot({{'training data', torch.Tensor(tr_reported_t[k])[{{#tr_reported_t[k]-l+1, #tr_reported_t[k]}}]},
            {'cv data', torch.Tensor(ts_reported_t[k])[{{#ts_reported_t[k]-l+1, #ts_reported_t[k]}}]}})
        gnuplot.plotflush()
        table.save( {train=tr_reported_t[k], test=ts_reported_t[k]}, 
        'out/txt/'..k..'/'..k..'_'..desc..'.txt')
        collectgarbage()
    end
    gnuplot.closeall()
end

function plot_theta()
    gnuplot.figure()
    gnuplot.title('gradient')
    gnuplot.imagesc(grad_hist, 'color')
    gnuplot.figure()
    gnuplot.title('theta')
    gnuplot.imagesc(hist_theta, 'color')
end
