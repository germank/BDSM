cv_datasets = {}
for i=1,1 do
    for j=1,10 do
        cv_datasets[#cv_datasets + 1] = {'../datasets/entailment/10-fold/case_'..i..'/positive-examples-train.train-'..j..'.txt',
                                        '../datasets/entailment/10-fold/case_'..i..'/negative-examples-train.train-'..j..'.txt', 
                                        '../datasets/entailment/10-fold/case_'..i..'/positive-examples-train.test-'..j..'.txt', 
                                        '../datasets/entailment/10-fold/case_'..i..'/negative-examples-train.test-'..j..'.txt'} 
    end
end

cv_bootstrap_datasets = {}
for i=1,1 do
    for j=1,10 do
        cv_bootstrap_datasets[#cv_bootstrap_datasets + 1] = {'../datasets/entailment/10-fold.bootstrap.1/case_'..i..'/positive-examples-train.train-'..j..'.txt',
                                        '../datasets/entailment/10-fold.bootstrap.1/case_'..i..'/negative-examples-train.train-'..j..'.txt', 
                                        '../datasets/entailment/10-fold.bootstrap.1/case_'..i..'/positive-examples-train.test-'..j..'.txt', 
                                        '../datasets/entailment/10-fold.bootstrap.1/case_'..i..'/negative-examples-train.test-'..j..'.txt'} 
    end
end

decreasing_datasets = {}
for i=1,9 do
    for j=1,10 do
        decreasing_datasets[#decreasing_datasets + 1] = {'../datasets/entailment/decreasing/case_'..i..'/positive-examples-train.train-'..j..'.txt',
                                        '../datasets/entailment/decreasing/case_'..i..'/negative-examples-train.train-'..j..'.txt', 
                                        '../datasets/entailment/decreasing/case_'..i..'/positive-examples-train.test-'..j..'.txt', 
                                        '../datasets/entailment/decreasing/case_'..i..'/negative-examples-train.test-'..j..'.txt'} 
    end
end

reversed_test_datasets = {}
for i=1,1 do
    for j=1,10 do
        reversed_test_datasets[#reversed_test_datasets + 1] = {
                                                   '../datasets/entailment/reversed-test/case_'..i..'/positive-examples-train.train-'..j..'.txt',
                                                   '../datasets/entailment/reversed-test/case_'..i..'/negative-examples-train.train-'..j..'.txt', 
                                                   '../datasets/entailment/reversed-test/case_'..i..'/positive-examples-train.test-'..j..'.txt', 
                                                   '../datasets/entailment/reversed-test/case_'..i..'/negative-examples-train.test-'..j..'.txt'} 
    end
end

reversed_datasets = {}
for i=1,1 do
    for j=1,10 do
        reversed_datasets[#reversed_datasets + 1] = {'../datasets/entailment/reversed/cv/case_'..i..'/positive-examples-core.train-'..j..'.txt',
                                                   '../datasets/entailment/reversed/cv/case_'..i..'/negative-examples-core.train-'..j..'.txt', 
                                                   '../datasets/entailment/reversed/cv/case_'..i..'/positive-examples-core.test-'..j..'.txt', 
                                                   '../datasets/entailment/reversed/cv/case_'..i..'/negative-examples-core.test-'..j..'.txt'} 
    end
end

minoverlap_datasets = {}
for i=1,100 do
        minoverlap_datasets[#minoverlap_datasets + 1] = {
            '../datasets/entailment/minoverlap/positive-examples-train-train-'..i..'.txt',
            '../datasets/entailment/minoverlap/negative-examples-train-train-'..i..'.txt',
            '../datasets/entailment/minoverlap/positive-examples-train-test-'..i..'.txt',
            '../datasets/entailment/minoverlap/negative-examples-train-test-'..i..'.txt'}
end
develop_datasets = {{ '../datasets/entailment/positive-examples-train.txt', '../datasets/entailment/negative-examples-train.txt', '../datasets/entailment/positive-examples-dev.txt', '../datasets/entailment/negative-examples-dev.txt'}}

full_datasets = {{'../datasets/entailment/positive-examples-balanced.txt',
        '../datasets/entailment/negative-examples-balanced.txt'}}

eacl_bless_coord_datasets = {{'../datasets/entailment/positive-examples-balanced.txt',
        '../datasets/entailment/negative-examples-balanced.txt', 
        '../datasets/bless/hyper_filtered.txt',
        '../datasets/bless/coord_filtered.txt'}}

eacl_bless_mero_datasets = {{'../datasets/entailment/positive-examples-balanced.txt',
        '../datasets/entailment/negative-examples-balanced.txt', 
        '../datasets/bless/hyper_filtered.txt',
        '../datasets/bless/mero_filtered.txt'}}

taxonomy_datasets = {{'../datasets/taxonomy/eacl-filtered-positive.txt',
        '../datasets/taxonomy/eacl-filtered-negative.txt', 
        '../datasets/taxonomy/positive.txt',
        '../datasets/taxonomy/negative.txt'}}

taxonomy_full_datasets = {{'../datasets/taxonomy-full/eacl-filtered-positive.txt',
        '../datasets/taxonomy-full/eacl-filtered-negative.txt', 
        '../datasets/taxonomy-full/positive.txt',
        '../datasets/taxonomy-full/negative.txt'}}

taxonomy_full_relaxed_datasets = {{'../datasets/taxonomy-full/eacl-positive.txt',
        '../datasets/taxonomy-full/eacl-negative.txt', 
        '../datasets/taxonomy-full/positive.txt',
        '../datasets/taxonomy-full/negative.txt'}}

sick_develop_datasets = {{
        '../datasets/sick/SICK_train-positive.txt',
        '../datasets/sick/SICK_train-negative.txt',
        '../datasets/sick/SICK_trial-positive.txt',
        '../datasets/sick/SICK_trial-negative.txt'}}
sick_balanced_develop_datasets = {{
        '../datasets/sick/SICK_train_balanced-positive.txt',
        '../datasets/sick/SICK_train_balanced-negative.txt',
        '../datasets/sick/SICK_trial_balanced-positive.txt',
        '../datasets/sick/SICK_trial_balanced-negative.txt'}}
sick_test_datasets = {{
        '../datasets/sick/SICK_train-positive.txt',
        '../datasets/sick/SICK_train-negative.txt',
        '../datasets/sick/SICK_test-positive.txt',
        '../datasets/sick/SICK_test-negative.txt'}}
sick_decreasing_datasets = {}
for i=1,9 do
    for j=1,10 do
        sick_decreasing_datasets[#sick_decreasing_datasets+1] = {
            '../datasets/sick/decreasing/case_'..i..'/SICK-positive.train-'..j..'.txt',
            '../datasets/sick/decreasing/case_'..i..'/SICK-negative.train-'..j..'.txt',
            '../datasets/sick/decreasing/case_'..i..'/SICK-positive.test-'..j..'.txt',
            '../datasets/sick/decreasing/case_'..i..'/SICK-negative.test-'..j..'.txt'}
    end
end

sick_decreasing2_datasets = {}
for i=1,19 do
    for j=1,10 do
        sick_decreasing2_datasets[#sick_decreasing2_datasets+1] = {
            '../datasets/sick/decreasing2/case_'..i..'/SICK-positive.train-'..j..'.txt',
            '../datasets/sick/decreasing2/case_'..i..'/SICK-negative.train-'..j..'.txt',
            '../datasets/sick/decreasing2/case_'..i..'/SICK-positive.test-'..j..'.txt',
            '../datasets/sick/decreasing2/case_'..i..'/SICK-negative.test-'..j..'.txt'}
    end
end

sick_decreasing_balanced2_datasets = {}
for i=1,19 do
    for j=1,10 do
        sick_decreasing_balanced2_datasets[#sick_decreasing_balanced2_datasets+1] = {
            '../datasets/sick/decreasing-balanced2/case_'..i..'/SICK-balanced-positive.train-'..j..'.txt',
            '../datasets/sick/decreasing-balanced2/case_'..i..'/SICK-balanced-negative.train-'..j..'.txt',
            '../datasets/sick/decreasing-balanced2/case_'..i..'/SICK-balanced-positive.test-'..j..'.txt',
            '../datasets/sick/decreasing-balanced2/case_'..i..'/SICK-balanced-negative.test-'..j..'.txt'}
    end
end

sick_decreasing_orig_test_datasets = {}
for i=1,9 do
    for j=1,10 do
        sick_decreasing_orig_test_datasets[#sick_decreasing_orig_test_datasets+1] = {
            '../datasets/sick/decreasing-orig-test/case_'..i..'/SICK-positive.train-'..j..'.txt',
            '../datasets/sick/decreasing-orig-test/case_'..i..'/SICK-negative.train-'..j..'.txt',
            '../datasets/sick/decreasing-orig-test/case_'..i..'/SICK-positive.test-'..j..'.txt',
            '../datasets/sick/decreasing-orig-test/case_'..i..'/SICK-negative.test-'..j..'.txt'}
    end
end

sick_decreasing_balanced_datasets = {}
for i=1,9 do
    for j=1,10 do
        sick_decreasing_balanced_datasets[#sick_decreasing_balanced_datasets+1] = {
            '../datasets/sick/decreasing-balanced/case_'..i..'/SICK-positive.train-'..j..'.txt',
            '../datasets/sick/decreasing-balanced/case_'..i..'/SICK-negative.train-'..j..'.txt',
            '../datasets/sick/decreasing-balanced/case_'..i..'/SICK-positive.test-'..j..'.txt',
            '../datasets/sick/decreasing-balanced/case_'..i..'/SICK-negative.test-'..j..'.txt'}
    end
end
