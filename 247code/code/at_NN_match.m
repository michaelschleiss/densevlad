function [match,idx12,idx21] = at_NN_match(f1,desc1,f2,desc2,sifttype)

if strcmp(sifttype,'DSIFT')
  level = max(f1(4,:));
  id1 = find(f1(4,:)==level);
  id2 = find(f2(4,:)==level);
  desc1 = desc1(:,id1);
  desc2 = desc2(:,id2);
end

Ndesc = size(desc1,2);

[idx12, dis12] = yael_nn(single(desc2), single(desc1), 2);
[idx21, dis21] = yael_nn(single(desc1), single(desc2), 2);

match = NaN(3,Ndesc);
if strcmp(sifttype,'DSIFT')
  for ii=1:Ndesc
    if ~isnan(idx12(1,ii))
      if idx21(1,idx12(1,ii)) == ii
        match(:,ii) = [id1(ii); id2(idx12(1,ii)); dis12(1,ii)];
      end
    end
  end
else
  for ii=1:Ndesc
    if ~isnan(idx12(1,ii))
      if idx21(1,idx12(1,ii)) == ii
        match(:,ii) = [ii; idx12(1,ii); dis12(1,ii)];
      end
    end
  end
end
match = match(:,~isnan(match(1,:)));