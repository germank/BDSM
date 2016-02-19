require 'table'
require 'dataset_files'
require 'pl.stringx'.import()

resume = false

maxDSLen = nil
visibleSize =300;   -- number of input units 
--declare global variables
hiddenSize = nil;     -- number of hidden units 
sparsityParam = nil;   -- desired average activation of the hidden units.
                     -- (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
             --  in the lecture notes). 
lambda = nil    -- weight decay parameter       
beta = nil-- 0.5;            -- weight of sparsity penalty term       
desc = nil
--[[ COST PARAMETERS ]]--
sparsity = true
regularization = true


M = 100
M2 = 100
t = nil
d = nil
plot_limx=10
plot_points=50

toyHiddenSize = 10
toyVisibleSize = 5

plot_dim1 = 1
plot_dim2 = 11

--train_positive_ds, train_negative_ds = '../datasets/eacl/positive-examples-train-train.txt', '../datasets/eacl/negative-examples-train-train.txt'
--test_positive_ds, test_negative_ds = '../datasets/eacl/positive-examples-train-cv.txt', '../datasets/eacl/negative-examples-train-cv.txt'
datasets=nil
ds_name = nil
train_positive_ds, train_negative_ds = nil, nil
test_positive_ds, test_negative_ds = nil, nil


Nretrain = 1 --number of different starting points to try the optimization
max_train_err = 45
Ngrid_samples = 1 --Average this number of samples to get an estimate of cv accuracy
maxIter = 1000 --max number of iterations of lbfgs
repetitions=1 --number of times that we restart lbfgs after a break

save_hist = false
hist_size = 50

--sparate a bit the limits of the KL func so it doesn't go to inf
kl_marg = 1e-100
logl_marg = 0.05 --should be a function of t
--type of hypothesis function
hyptype = 'min'
l2type = 'softmax'
losstype = 'mse'
layers=nil


_ds_name = 'develop'
_datasets = _G[_ds_name..'_datasets']
--space_type = 'sick_pmi_svd300_normed_adv_weighted_additive'
--space_type = 'sick_pmi_svd300_normed_plf'
--space_type = 'sick_mikolov_normed_adv_weighted_additive'
space_type='count'
pos_space=false
if space_type == 'mikolov' then
    vectors_file = 'eacl_mikolov.th7'
    words_file = 'eacl_mikolov_words.txt'
elseif space_type == 'count' then
    vectors_file = 'spaces/count/CORE_SS.EN-wform.w.2.ppmi.svd_300.row.th7'
    words_file = 'spaces/count/rows'
elseif space_type == 'dm' then
    vectors_file = '../spaces/dm/ri.w-lw.row.th7'
    words_file = '../spaces/dm/rows'
    pos_space=true
elseif string.startswith(space_type,'sick') then
    space_name = string.sub(space_type, string.len('sick_')+1)
    vectors_file = '../spaces/sick/merged/vectors_'..space_name..'.th7'
    words_file = '../spaces/sick/merged/sentences_'..space_name..'.txt'
else
    error('unknown space type: '..space_type)
end


function get_desc()
    local spdesc, regdesc,layersdesc
    if sparsity then spdesc = "_sparse" else spdesc="" end
    if regularization then regdesc = "_reg" else regdesc="" end
    if layers then layersdesc = "_layers_"..layers else layersdesc="" end
    local desc=ds_name.."_"..space_type.."_"..hyptype.."_"..l2type.."_"..losstype.."_v_"..visibleSize.."_h_"..hiddenSize.."_sp_"..sparsityParam.."_t_"..t.."_l_"..lambda.."_b_"..beta..spdesc..regdesc..layersdesc
    return desc
end

function load_parameters(...) 
    if ... and table.length(...) > 0 then
        for k,v in pairs(...) do
            _G[k] = v
        end
    end
end
