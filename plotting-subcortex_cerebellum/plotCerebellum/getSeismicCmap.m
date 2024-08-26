% function cmap = getSeismicCmap(m, start_frac, end_frac)
%     if nargin < 1
%         m = 256; % Default number of colors
%     end
%     if nargin < 3
%         start_frac = 1;
%         end_frac = 0;
%     end
% 
%     % Full seismic colormap (blue-white-red)
%     mid = floor(m/2);
%     r = [(0:1:mid-1)/mid, ones(1, mid)];
%     g = [(0:1:mid-1)/mid, (mid-1:-1:0)/mid];
%     b = [ones(1, mid), (mid-1:-1:0)/mid];
% 
%     seismic = [r', g', b'];
% 
%     % Calculate start and end indices for the subset
%     start_idx = max(1, floor(start_frac * m));
%     end_idx = min(m, ceil(end_frac * m));
% 
%     % Extract the subset
%     cmap = seismic(start_idx:end_idx, :);
% end
% 
% % Usage
% %cmap_subset = create_seismic_subset(256, 0.15, 0.86);

function cmap = getSeismicCmap(m, start_frac, end_frac)
    if nargin < 1
        m = 256; % Default number of colors
    end
    if nargin < 3
        start_frac = 1;
        end_frac = 0;
    end

    % Full seismic colormap (blue-white-red)
    mid = floor(m/2);
    r = [(0:1:mid-1)/mid, ones(1, mid)];
    g = [(0:1:mid-1)/mid, (mid-1:-1:0)/mid];
    b = [ones(1, mid), (mid-1:-1:0)/mid];

    seismic = [r', g', b'];

    % Change the last color to yellow
    seismic(end, :) = [1, 1, 0];  % RGB for yellow

    % Calculate start and end indices for the subset
    start_idx = max(1, floor(start_frac * m));
    end_idx = min(m, ceil(end_frac * m));

    % Extract the subset
    cmap = seismic(start_idx:end_idx, :);
end

