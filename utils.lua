--Concatenation of multiple tensors 
--(not sure if there is a more efficient way to do it)
function ncat(...) 
   local args = {...}
   local dim = args[#args]
   if tonumber(dim) ~= nil then
      table.remove(args)
   end
   local r = table.remove(args, 1)
   for k,x in ipairs(args) do
      if tonumber(dim) ~=nil then
         r = torch.cat(r,x,dim)
      else
         r = torch.cat(r,x)
      end
   end
   return r
end

function flatten(v)
    return v:reshape(#v:storage())
end

--local nanString = (tostring((-1) ^ 0.5)); --sqrt(-1) is also NaN. 
--Unfortunately, 
--  tostring((-1)^0.5))       = "-1.#IND"
--  x = tostring((-1)^0.5))   = "0"
--With this bug in LUA we can't use this optimization
local function isnan(x) 
    if (x ~= x) then
        --print(string.format("NaN: %s ~= %s", x, x));
        return true; --only NaNs will have the property of not being equal to themselves
    end;

    --but not all NaN's will have the property of not being equal to themselves

    --only a number can not be a number
    if type(x) ~= "number" then
       return false; 
    end;

    --fails in cultures other than en-US, and sometimes fails in enUS depending on the compiler
--  if tostring(x) == "-1.#IND" then

    --Slower, but works around the three above bugs in LUA
    if tostring(x) == tostring((-1)^0.5) then
        --print("NaN: x = sqrt(-1)");
        return true; 
    end;

    --i really can't help you anymore. 
    --You're just going to have to live with the exception

    return false;
end
function mean(values)
    local sum,n = 0,#values
    for i = 1,n do
        sum = sum + values[i]
    end
    local mean = sum/n
    return mean
end


function table.length(t)
    local n = 0
    for k in pairs(t) do
        n = n+1
    end
    return n
end
function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then io.close(f) return true else return false end
end
