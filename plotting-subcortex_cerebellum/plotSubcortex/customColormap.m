function cmap = customColormap()

    % Define the 32 colors in RGB format
    colors = [
        60, 180, 75;   % Green
        255, 225, 25;  % Yellow
        0, 130, 200;   % Blue
        245, 130, 48;  % Orange
        145, 30, 180;  % Purple
        70, 240, 240;  % Cyan
        240, 50, 230;  % Magenta
        210, 245, 60;  % Lime
        250, 190, 190; % Pink
        % 0, 128, 128;   % Teal
        230, 190, 255; % Lavender
        230, 25, 75;   % Red
        170, 110, 40;  % Brown
        128, 0, 0;     % Maroon
        170, 255, 195; % Mint
        128, 128, 0;   % Olive
        % 255, 215, 180; % Coral
        0, 0, 128;     % Navy
    ];
    % Normalize the RGB values to the range [0, 1]
    cmap = colors / 255;

end