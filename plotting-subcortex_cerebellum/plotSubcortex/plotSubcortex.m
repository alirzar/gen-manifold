
function f = plotSubcortex(data, color_range)

% 32 ROIs subcortex plots

% example usage: plotSubcortex((1:32)', [1, 32])

load subcorticalSurf;

nROI = 16;

h = 0.2;    % subplot height
w = 0.2;    % subplot width
m = 0.02;   % subplot horizontal margin

f(1)=axes('position',[0.5-w*2-m*1.5 0.5-h/2 w h]);
hold;

for p = 1 : nROI
    if nVL(p)
        trisurf(VL{p}.S, VL{p}.x, VL{p}.y, VL{p}.z, ones(nVL(p),1)*data(p), 'EdgeColor', 'none');
    end
end

view(-90, 0)
%view(0,0)
daspect([1 1 1]); axis tight; camlight; axis vis3d off;
lighting phong; material dull; shading flat;

f(2)=axes('position',[0.5-w-m*0.5 0.5-h/2 w h]);
hold;

for p = 1 : nROI
    if nVL(p)
        trisurf(VL{p}.S, VL{p}.x, VL{p}.y, VL{p}.z, ones(nVL(p),1)*data(p), 'EdgeColor', 'none');
    end
end

view(90, 0)
%view(180,0)
daspect([1 1 1]); axis tight; camlight; axis vis3d off;
lighting phong; material dull; shading flat;

f(3)=axes('position',[0.5+m*0.5 0.5-h/2 w h]);
hold;

for p = 1 : nROI
    if nVR(p)
        trisurf(VR{p}.S, VR{p}.x, VR{p}.y, VR{p}.z, ones(nVR(p),1)*data(p+nROI), 'EdgeColor', 'none');
    end
end

view(-90,0)
%view(0, 0)
daspect([1 1 1]); axis tight; camlight; axis vis3d off;
lighting phong; material dull; shading flat;

f(4)=axes('position',[0.5+w+m*1.5 0.5-h/2 w h]);
hold;

for p = 1 : nROI
    if nVR(p)
        trisurf(VR{p}.S, VR{p}.x, VR{p}.y, VR{p}.z, ones(nVR(p),1)*data(p+nROI), 'EdgeColor', 'none');
    end
end

view(90,0)
%view(180, 0)
daspect([1 1 1]); axis tight; camlight; axis vis3d off;
lighting phong; material dull; shading flat;

for i=1:length(f)
    set(f(i),'CLim', color_range);
    set(f(i),'Tag',['SurfStatView ' num2str(i)]);
end

whitebg(gcf, 'white');
set(gcf,'Color', 'white', 'InvertHardcopy', 'off');

dcm_obj=datacursormode(gcf);
set(dcm_obj,'UpdateFcn',@SurfStatDataCursor,'DisplayStyle','window');
end




