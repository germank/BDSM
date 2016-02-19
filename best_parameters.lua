require 'dataset_files'
--[[
develop_datasets ={{ '../datasets/eacl/positive-examples-train-cv.txt', '../datasets/eacl/negative-examples-train-cv.txt', '../datasets/eacl/positive-examples-train-train.txt', '../datasets/eacl/negative-examples-train-train.txt'}}
develop_datasets ={{ '../datasets/eacl/positive-examples-train-train.txt', '../datasets/eacl/negative-examples-train-train.txt'}}
develop_datasets = {{'../datasets/entailment/10-fold/positive-examples.train-1.txt', '../datasets/entailment/10-fold/negative-examples.train-1.txt'}}
]]--
local ds_type
if string.startswith(_ds_name, 'sick') then
    ds_type = 'sick'
else
    ds_type = 'words'
end

chosen={}
chosen[10] = true
chosen[100] = true
chosen[500] = true
chosen[1000] = true
chosen[1500] = true
chosen['100-extra'] = true
chosen['500-extra'] = true
chosen['1000-extra'] = true
chosen['1500-extra'] = true

if layers == 2 then
entry
{
    ds_name = _ds_name,
    datasets = _datasets,
    beta = 0.1,
    lambda = 1e-05,
    sparsityParam = 0.5,
    t = 0.1,
    hiddenSize = 100
}

else
    if ds_type == 'sick' then
      entry {
        ds_name = _ds_name,
        datasets = _datasets,
        beta = 0.1,
        lambda = 1e-05,
        sparsityParam = 0.5,
        t = 1,
        hiddenSize = 100
      }
    elseif ds_type == 'words' then 
    --[[HS=10 ]]--
    if chosen[10] then
      entry {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 0.1,
          lambda = 0.0001,
          sparsityParam = 0.5,
          t = 0.1,
          hiddenSize = 10
      }
        entry
        {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 0.1,
          lambda = 0.0001,
          sparsityParam = 0.1,
          t = 0.1,
          hiddenSize = 10
        }

        entry
        {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 0.1,
          lambda = 0.001,
          sparsityParam = 0.75,
          t = 0.1,
          hiddenSize = 10
        }
        entry
        {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 1,
          lambda = 0.0001,
          sparsityParam = 0.25,
          t = 0.1,
          hiddenSize = 10
        }
        entry
        {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 0.1,
          lambda = 0.0001,
          sparsityParam = 0.05,
          t = 0.1,
          hiddenSize = 10
        }
        entry
        {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 0.1,
          lambda = 1e-05,
          sparsityParam = 0.01,
          t = 0.1,
          hiddenSize = 10
        }
    end

    --[[HS=500 ]]--
    if chosen[500] then
        entry
        {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 0.1,
          lambda = 0.0001,
          sparsityParam = 0.01,
          t = 0.1,
          hiddenSize = 500
        }
        entry
        {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 0.1,
          lambda = 0.001,
          sparsityParam = 0.05,
          t = 0.001,
          hiddenSize = 500
        }
        entry
        {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 10,
          lambda = 0.001,
          sparsityParam = 0.1,
          t = 0.1,
          hiddenSize = 500
        }
    end

    --[[Extra sparsity parameters]]--
    if chosen['500-extra'] then
    entry
    {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 1,
      lambda = 0.001,
      sparsityParam = 0.25,
      t = 0.1,
      hiddenSize = 500
    }
    entry
    {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 1,
      lambda = 1e-05,
      sparsityParam = 0.5,
      t = 1,
      hiddenSize = 500
    }
        entry
        {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 0.1,
          lambda = 0.001,
          sparsityParam = 0.75,
          t = 0.1,
          hiddenSize = 500
        }
    end

    --[[
    --HS=100
    ]]--

    if chosen[100] then
    entry {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 0.1,
      lambda = 0.001,
      sparsityParam = 0.01,
      t = 0.1,
      hiddenSize = 100
    }
    entry {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 0.1,
      lambda = 1e-05,
      sparsityParam = 0.05,
      t = 0.0001,
      hiddenSize = 100
    }
    entry {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 0.1,
      lambda = 1e-05,
      sparsityParam = 0.1,
      t = 0.1,
      hiddenSize = 100
    }
    end


    if chosen['100-extra'] then
    entry
    {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 0.1,
      lambda = 1e-05,
      sparsityParam = 0.5,
      t = 1,
      hiddenSize = 100
    }
    --[[
        entry
        {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 10,
          lambda = 1e-05,
          sparsityParam = 0.25,
          t = 1,
          hiddenSize = 100
        }
        entry
        {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 0.1,
          lambda = 0.001,
          sparsityParam = 0.75,
          t = 0.1,
          hiddenSize = 100
        }
    --]]
    end

    --[[
    --HS=1000
    ]]--

    if chosen['1000-extra'] then 
    entry
    {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 1,
      lambda = 1e-05,
      sparsityParam = 0.25,
      t = 1,
      hiddenSize = 1000
    }
    entry
    {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 0.1,
      lambda = 0.0001,
      sparsityParam = 0.5,
      t = 0.1,
      hiddenSize = 1000
    }
        entry
        {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 0.1,
          lambda = 0.001,
          sparsityParam = 0.75,
          t = 0.1,
          hiddenSize = 1000
        }
    end

    if chosen[1000] then
    entry {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 0.1,
      lambda = 0.001,
      sparsityParam = 0.01,
      t = 0.1,
      hiddenSize = 1000
    }

    entry {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 0.1,
      lambda = 1e-05,
      sparsityParam = 0.05,
      t = 0.1,
      hiddenSize = 1000
    }
    entry {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 1,
      lambda = 0.001,
      sparsityParam = 0.1,
      t = 0.1,
      hiddenSize = 1000
    }
    end

    --[[
    --HS=1500
    ]]--

    if chosen[1500] then
    entry
    {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 0.1,
      lambda = 0.001,
      sparsityParam = 0.01,
      t = 0.1,
      hiddenSize = 1500
    }

    entry
    {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 1,
      lambda = 1e-05,
      sparsityParam = 0.05,
      t = 0.1,
      hiddenSize = 1500
    }

    entry
    {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 0.1,
      lambda = 0.0001,
      sparsityParam = 0.1,
      t = 0.1,
      hiddenSize = 1500
    }
    end

    if chosen['1500-extra'] then
    entry
    {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 1,
      lambda = 0.001,
      sparsityParam = 0.25,
      t = 0.1,
      hiddenSize = 1500
    }
    entry
    {
      ds_name = _ds_name,
      datasets = _datasets,
      beta = 0.1,
      lambda = 1e-05,
      sparsityParam = 0.5,
      t = 0.1,
      hiddenSize = 1500
    }
        entry
        {
          ds_name = _ds_name,
          datasets = _datasets,
          beta = 0.1,
          lambda = 0.001,
          sparsityParam = 0.75,
          t = 0.1,
          hiddenSize = 1500
        }
    end
    end
end
