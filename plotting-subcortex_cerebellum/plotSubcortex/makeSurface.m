close all;
clear;

shrinkFactor = 0.5;

M = niftiread('Tian_Subcortex_S2_3T.nii.gz');

nROI = max(M, [], 'all')/2;

sz = size(M);

for p = 1 : nROI
    VL{p}.x = [];
    VL{p}.y = [];
    VL{p}.z = [];
end

for p = 1 : nROI
    VR{p}.x = [];
    VR{p}.y = [];
    VR{p}.z = [];
end

for i = 1 : sz(1)
    for j = 1 : sz(2)
        for k = 1 : sz(3)
            p = M(i,j,k);
            if p > 0
                if p <= 16 % RH
                    VR{p}.x = [VR{p}.x; -i];
                    VR{p}.y = [VR{p}.y; j];
                    VR{p}.z = [VR{p}.z; k];
                else % LH
                    p = p - 16;
                    VL{p}.x = [VL{p}.x; -i];
                    VL{p}.y = [VL{p}.y; j];
                    VL{p}.z = [VL{p}.z; k];
                end
            end
        end
    end
end

for p = 1 : nROI
    nVL(p) = length(VL{p}.x);
    if nVL(p)
        VL{p}.S = boundary(VL{p}.x, VL{p}.y, VL{p}.z, shrinkFactor);
    end
end

for p = 1 : nROI
    nVR(p) = length(VR{p}.x);
    if nVR(p)
        VR{p}.S = boundary(VR{p}.x, VR{p}.y, VR{p}.z, shrinkFactor);
    end
end

save subcorticalSurf VL VR nVL nVR