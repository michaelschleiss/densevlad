function R = rodrigue(t,v)

M = asym(v);
R = eye(3)+sin(t)*M + (1-cos(t))*M*M;


function M = asym(d)
M = [0 -d(3) d(2);d(3) 0 -d(1);-d(2) d(1) 0];
