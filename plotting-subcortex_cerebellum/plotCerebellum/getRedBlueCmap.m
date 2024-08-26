function cmapFinal = getRedBlueCmap()
    % Define the red, blue, and lighter gray RGB values
    red = [1, 0, 0];
    blue = [0, 0, 1];
    gray = [0.9, 0.9, 0.9];  % Lighter gray

    % Generate a colormap that transitions from blue to gray and then from gray to red
    m = 256;  % Number of colors in each half of the colormap
    gray_zone = 1;  % Number of colors around zero to be gray

    % Create the first half of the colormap, from blue to gray
    cmap1 = [linspace(blue(1), gray(1), m - gray_zone / 2)', ...
             linspace(blue(2), gray(2), m - gray_zone / 2)', ...
             linspace(blue(3), gray(3), m - gray_zone / 2)'];

    % Create the second half of the colormap, from gray to red
    cmap2 = [linspace(gray(1), red(1), m - gray_zone / 2 + 1)', ...
             linspace(gray(2), red(2), m - gray_zone / 2 + 1)', ...
             linspace(gray(3), red(3), m - gray_zone / 2 + 1)'];

    % Insert the gray_zone in the middle of the colormap
    cmapGray = repmat(gray, gray_zone, 1);  % A small segment of gray colors

    % Combine the colormaps to create the final colormap
    cmapFinal = [cmap1; cmapGray; cmap2(2:end, :)];  % Avoid repeating the gray value
end