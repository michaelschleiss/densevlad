function out = im2single(in)
%IM2SINGLE Fallback im2single implementation without Image Processing Toolbox.
%   Converts input to single, scaling integer inputs to [0,1].

if isfloat(in)
    out = single(in);
    return;
end

out = single(in);
out = out ./ single(intmax(class(in)));
end
