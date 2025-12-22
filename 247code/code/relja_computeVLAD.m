% Author: Relja Arandjelovic (relja@relja.info)

function vlad= computeVLAD(descs, cents, kdtree)
    if nargin<3
        kdtree= vl_kdtreebuild(cents);
    end
    
    nn= vl_kdtreequery(kdtree, cents, descs) ;
    vlad= vl_vlad(descs, cents, nn, 'NormalizeComponents');
    
end
