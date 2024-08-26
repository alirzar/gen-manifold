function f = plotCerebellum(data, color_range)

% 32 ROIs subcortex plots

% example usage: plotCerebellum((1:32)', [1, 32])

load cerebellarAtlas;

idx = G.cdata(:,1);
map = data(idx);

w = 1;
h = 1;

f = axes('position',[0.5-w/2, 0.5-h/2, w, h]);
%colormap('parula')
plotflatmap(map, 'type', 'func', 'cscale', color_range, 'border', [], 'alpha', 0.75);
%colorbar
axis off
end