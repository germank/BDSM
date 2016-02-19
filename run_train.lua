#!/usr/bin/env luajit
--require 'pl/strict'
require 'train'
require 'io'
require 'lfs'
require 'utils'
require 'config'

command = arg[1] or 'predict' --'plot', 'plot-theta', 'train'
if command == 'grid-search' then
    grid_hiddenSize = tonumber(arg[2])
    if grid_hiddenSize then
        print ('Only searching for parameters with hiddenSize='..grid_hiddenSize)
    end
    grid_sparsity = tonumber(arg[3])
    if grid_sparsity then
        print ('Only searching for parameters with sparsity='..grid_sparsity)
    end
else 
    init_theta_file = arg[2]
end

commands = {['plot']= true, ['plot-theta']= true, ['train']=true, ['check-grad']=true,
    ['grid-search'] = true, ['cross-validate'] = true, ['plot-develop'] = true, ['test'] = true,
    ['predict'] = true}

if not commands[command] then
    require 'os'
    print ('invalid command: '..command)
    os.exit(1)
end

mse_hist = {}
params_hist = {}
function run_command(...)
    load_parameters(...)
    print ('Using parameters: ')
    require 'pl/pretty'.dump({...})

    reported_t = {}
    desc = get_desc()
    if command == 'check-grad' then
        --print ('WARNING: using manual seed')
        --torch.manualSeed(42)
        t=1
        save_hist = false
        visibleSize=toyVisibleSize
        hiddenSize=toyHiddenSize
        check_grad()
    elseif command == 'grid-search' then
        show_report = function(t) end
        if (not grid_hiddenSize or grid_hiddenSize == hiddenSize) and
            (not grid_sparsity or grid_sparsity == sparsityParam)
        then
            local avg_mse = 0
            local mse_samples = 0
            for i=1,Ngrid_samples do
                local theta, mse = get_theta(true, false)
                --new_report()
                test_theta {theta}
                --local ts_report = get_report()
                if mse then
                    print('Testing MSE='..mse)--ts_report['MSE'][#ts_report['MSE']])
                    avg_mse = avg_mse + mse --ts_report['MSE'][#ts_report['MSE']]
                    mse_samples = mse_samples + 1
                else
                    print ("Couldn't find a solution for the parameters")
                end
            end
            avg_mse = avg_mse / mse_samples
            params_hist[#params_hist + 1] = ...
            mse_hist[#mse_hist + 1 ] = {avg_mse, mse_samples}
        else 
            print('Print ignoring hiddenSize='..hiddenSize..', sparsityParam='..sparsityParam)
        end
    elseif command == 'cross-validate' then
        show_report = function(t) end
        local theta = get_theta(true, true)
        new_report()
        test_theta {theta}
        local ts_report = get_report()
        mse_hist[#mse_hist + 1 ] = ts_report['MSE'][#ts_report['MSE']]
    elseif command == 'plot-develop' then
        train_cv_theta(true, true)
    elseif command == 'predict' then
        show_report = function(t) end
        local tsoutdir = 'out/predict/test_ds/'..get_desc()
        local troutdir = 'out/predict/train_ds/'..get_desc()
        os.execute('mkdir -p '..tsoutdir)
        os.execute('mkdir -p '..troutdir)
        save_hist = false
        local all_datasets = datasets
        for ds in pairs(all_datasets) do
            datasets = all_datasets[ds]
            train_positive_ds, train_negative_ds, test_positive_ds, test_negative_ds = unpack(datasets)
            print('Running tests for datasets...', unpack(datasets))
            if resume and file_exists(tsoutdir..'/'..ds..'.txt') and
                file_exists(troutdir..'/'..ds..'.txt') then
                print 'Skipping'
            else
                local theta,mse = get_theta(false, false)
                if theta then
                    print('Training MSE: '..mse)
                    local ret = predict (theta, test_positive_ds, test_negative_ds)
                    local tsfout = io.open(tsoutdir..'/'..ds..'.txt', 'w')
                    for i=1,#ret do
                        tsfout:write(table.concat(ret[i], '\t')..'\n')
                    end
                    tsfout:close()
                    local trfout = io.open(troutdir..'/'..ds..'.txt', 'w')
                    local ret = predict (theta, train_positive_ds, train_negative_ds)
                    for i=1,#ret do
                        trfout:write(table.concat(ret[i], '\t')..'\n')
                    end
                    trfout:close()
                else
                    print ('Could not fit the model to get under '..max_train_err..'% error on training data')
                    --We don't want incomplete data: abort!
                    return
                end
            end
        end
    elseif command == 'test' then
        show_report = function(t) end
        local fout = io.open('out/test_output_'..get_desc()..'.txt', 'w')
        local ftrout = io.open('out/train_output_'..get_desc()..'.txt', 'w')
        save_hist = false
        local all_datasets = datasets
        for ds in pairs(all_datasets) do
            datasets = all_datasets[ds]
            print('Running tests for datasets...', unpack(datasets))
            local theta, trmse = get_theta(false, false)
            ftrout:write((1-trmse)..'\n')
            ftrout:flush()
            new_report()
            local ret = test_theta {theta}
            local ts_report = get_report()
            assert(#ts_report['MSE'] == 1)
            local test_acc = 1-ts_report['MSE'][#ts_report['MSE']]
            fout:write(test_acc..'\n')
            fout:flush()
            print ('TEST ACCURACY: '..test_acc) 
        end
        ftrout:close()
        fout:close()
    elseif command == 'plot' then
    elseif command == 'plot' then
        save_hist = false
        lambda = 1e-4
        sparsityParam = 1
        beta = 1
        visibleSize=toyVisibleSize
        hiddenSize=toyHiddenSize
        plot_cost()
    end
end

entry = run_command


dofile('parameters.lua')


if command == 'cross-validate' then
    print ("Mean test MSE: "..mean(mse_hist))
elseif command == 'grid-search' then
    Nretrain = 20
    local bests = {}
    for i, v in pairs(mse_hist) do
        local mse_avg, mse_samples = unpack(v)
        if mse_samples == Ngrid_samples then
            local H = params_hist[i]['hiddenSize']
            local sp = params_hist[i]['sparsityParam']
            if not bests[H] then
                bests[H] = {}
            end
            if not bests[H][sp] or mse_hist[bests[H][sp]][1] > mse_avg then
                bests[H][sp] = i
            end
        end
    end
    for H, best_H in pairs(bests) do
        for sp, best_H_sp in pairs(best_H) do
            print ('Hidden Size: '..H)
            print ('Sparsity Param: '..sp)
            print ('Best MSE: '.. mse_hist[best_H_sp][1])
            print ('Best params: ')
            require 'pl.pretty'.dump(params_hist[best_H_sp])
        end
    end
end
