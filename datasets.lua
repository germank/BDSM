require 'torch'
require 'io'
require 'string'
require 'config'
function load_dataset(filename, revindex, max)
    local ds = {}
    local file = io.open(filename)
    if not file then
        error('Couldn\'t find file: '..filename)
    end
    for line in file:lines() do
        local h,c = string.match(line, '(%w+)-n\t(%w+)-n')
        --try to get the line from sick
        if not h or not c then
            h,c = string.match(line, '([%w-.\'",_]+)\t([%w.\',-_]+)')
        end
        if pos_space then
            h = h.."-n"
            c = c.."-n"
        end
        if not revindex or ( revindex[h] ~= nil and revindex[c] ~= nil ) then
            ds[#ds+1] = {h,c}
            if max ~= nil then
                max = max -1
                if max <= 0 then break end
            end
        elseif revindex then
            if not h or not c then
                io.stderr:write('Failed to parse line: "'.. line .. '"\n')
                print(h,c)
            else 
                if revindex[h] == nil then
                    io.stderr:write('Missing word: '..h..'\n')
                end
                if revindex[c] == nil then
                    io.stderr:write('Missing word: '..c..'\n')
                end
            end
        end
    end
    return ds
end
function load_wordlist(filename, revindex)
    local ds = {}
    local file = io.open(filename)
    local max = nil
    for line in file:lines() do
        local w = string.match(line, '(%w+)-n')
        if revindex[w] ~= nil then
            ds[#ds+1] = w
            if max ~= nil then
                max = max -1
                if max <= 0 then break end
            end
        else
            if revindex[w] == nil then
                io.stderr:write('Missing word: '..w..'\n')
            end
        end
    end
    return ds
end
function load_vectors() 
    local revindex = {}
    local file = io.open(words_file)
    local i=1
    for line in file:lines() do
        revindex[line] = i
        i = i + 1
    end
    local vectors = torch.load(vectors_file)
    return revindex,vectors
end
function wordpairs_to_vectors(wordpairs, revindex, vectors)
    local hs = torch.Tensor(#wordpairs, vectors:size()[2])
    local cs = torch.Tensor(#wordpairs, vectors:size()[2])
    for i,p in ipairs(wordpairs) do
        hs[i] = vectors[revindex[p[1]]]
        cs[i] = vectors[revindex[p[2]]]
    end
    return hs, cs
end

function words_to_vectors(words, revindex, vectors)
    local vs = torch.Tensor(#words, vectors:size()[2])
    for i,w in ipairs(words) do
        if revindex[w] == nil then
            error('Word '..w..' is not in the vocabulary')
        end
        vs[i] = vectors[revindex[w]]
    end
    return vs
end

function load_dataset_vectors(positive_file, negative_file, maxN)
    local revindex, vectors = load_vectors()
    if vectors:size()[2] > visibleSize then
        vectors = vectors[{{},{1,visibleSize}}]
    end
    local positive = load_dataset(positive_file, revindex, maxN)
    local positive_hs, positive_cs = wordpairs_to_vectors(positive, revindex, vectors)
    local positive_y = torch.Tensor(positive_hs:size()[1], 1):fill(1)
    
    local negative = load_dataset(negative_file, revindex, maxN)
    local negative_hs, negative_cs = wordpairs_to_vectors(negative, revindex, vectors)
    local negative_y = torch.Tensor(negative_hs:size()[1], 1):fill(0)

    local hs = torch.cat(positive_hs, negative_hs, 1)
    local cs = torch.cat(positive_cs, negative_cs, 1)
    local y = torch.cat(positive_y, negative_y, 1)
    return hs, cs, y
end


function ds_to_libsvm(hs, cs, y) 
    for i=1,y:size()[1] do
        io.write(y[{i,1}])
        io.write(" ")
        for j=1,hs:size()[2] do
            io.write(j)
            io.write(':')
            io.write(hs[{i,j}])
            io.write(' ')
        end

        for j=1,cs:size()[2] do
            io.write(hs:size()[2] + j)
            io.write(':')
            io.write(cs[{i,j}])
            io.write(' ')
        end
        io.write('\n')
    end
end
function ds_to_arff(hs, cs, y) 
    m = y:size()[1]
    n = hs:size()[2]
    io.write('@RELATION entailment\n')

    for j=1,2*n do
        io.write('@ATTRIBUTE att_'..j..' NUMERIC\n')
    end
    io.write('@ATTRIBUTE class {-1, 1}\n')

    io.write('@DATA\n')
    for i=1,m do
        for j=1,n do
            io.write(hs[{i,j}])
            io.write(',')
        end

        for j=1,n do
            io.write(cs[{i,j}])
            io.write(',')
        end
        io.write(y[{i,1}])
        io.write('\n')
    end
end
