
function count_all(f)
        local seen = {}
        local count_table
        count_table = function(t)
                if seen[t] then return end
                f(t)
                seen[t] = true
                for k,v in pairs(t) do
                        if type(v) == "table" then
                                count_table(v)
                        elseif type(v) == "userdata" then
                                f(v)
                        end
                end
        end
        count_table(_G)
end

function type_count()
        local counts = {}
        local enumerate = function (o)
                local t = type_name(o)
                counts[t] = (counts[t] or 0) + 1
        end
        count_all(enumerate)
        return counts
end

global_type_table = nil
function type_name(o)
        if global_type_table == nil then
                global_type_table = {}
                for k,v in pairs(_G) do
                        global_type_table[v] = k
                end
                global_type_table[0] = "table"
        end
        return global_type_table[getmetatable(o) or 0] or "Unknown"
end

function dump(o)
    if type(o) == 'table' then
        local s = '{ '
        for k,v in pairs(o) do
            if type(k) ~= 'number' then k = '"'..k..'"' end
                    s = s .. '['..k..'] = ' .. dump(v) .. ','
        end
        return s .. '} '
    else
        return tostring(o)
    end
end
