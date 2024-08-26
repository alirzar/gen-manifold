% function cmapFinal = getCmap()
%     % Define the cyan and orange RGB values
%     cyan = [0, 1, 1];
%     orange = [1, 0.5, 0];
% 
%     % Generate a colormap that transitions from cyan to orange
%     m = 256;  % Number of colors in the colormap
%     cmap = [linspace(cyan(1), orange(1), m)', linspace(cyan(2), orange(2), m)', linspace(cyan(3), orange(3), m)'];
% 
%     % Determine the indices for the start and end of the middle 90% of the colormap
%     startIdx = round(m * 0.05) + 1;
%     endIdx = round(m * 0.95);
% 
%     % Trim the colormap
%     cmapTrimmed = cmap(startIdx:endIdx, :);
% 
%     % Define the gray RGB value
%     gray = [0.5, 0.5, 0.5];
% 
%     % Add the gray color to the beginning of the colormap
%     cmapFinal = [gray; cmapTrimmed];
% end
function cmapFinal = getCmap()
    % Define the yellow and orange RGB values
    yellow = [1, 1, 0];  % Yellow color
    orange = [1, 0.5, 0];  % Orange color

    % Generate a colormap that transitions from yellow to orange
    m = 256;  % Number of colors in the colormap
    cmap = [linspace(orange(1), yellow(1), m)', ...
            linspace(orange(2), yellow(2), m)', ...
            linspace(orange(3), yellow(3), m)'];

    % Define the black RGB value for zero
    gray = [0.8, 0.8, 0.8];  % Black color

    % Insert the black color at the beginning of the colormap
    cmapFinal = [gray; cmap(2:end, :)];  % Assume the zero value maps to black
end