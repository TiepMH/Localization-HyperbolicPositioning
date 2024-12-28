clear all; clc; 

% In the (x_tilde, y_tilde)-coordinate system, the standar hyperbolic
% ... equation is (x_tilde^2 / a^2) - (y_tilde^2 / a^2) = 1. Hence, given
% ... y_tilde, we can calculate x_tilde. For illustration, we pick up some
% ... values for y_tilde in a range.
y_tilde_Range = [-30 30]; % y_tilde-axis range

%% Parameters of the 1-st hyperbola 
Focus1_Hyperbola1 = [0, 0]; % Position of the first focus
Focus2_Hyperbola1 = [7.5, 2]; % Position of the second focus
a_Hyperbola1 = 3; % Half-length of transverse axis

%% Preparation before plotting the 1-st hyperbola
[x_LeftBranch_Hyperbola1, y_LeftBranch_Hyperbola1, ...
 x_RightBranch_Hyperbola1, y_RightBranch_Hyperbola1, ...
 theta_Hyperbola1, ...
 b_Hyperbola1, ...
 h_Hyperbola1, k_Hyperbola1] = collect_points_on_Hyperbola_given_y_values(Focus1_Hyperbola1, ...
                                                                          Focus2_Hyperbola1, ...
                                                                          a_Hyperbola1, ...
                                                                          y_tilde_Range);

% Check if (x, y) lies on Hyperbola 1 
some_x = x_LeftBranch_Hyperbola1(1);
some_y = y_LeftBranch_Hyperbola1(1);
equal_to_1 = checking_if_a_point_is_on_the_hyperbola(some_x, some_y, ...
                                                     theta_Hyperbola1, ...
                                                     a_Hyperbola1, b_Hyperbola1, ...
                                                     h_Hyperbola1, k_Hyperbola1)

%% Parameters of the 2nd hyperbola 
Focus1_Hyperbola2 = [0, 0]; % The first focus is the reference point 
Focus2_Hyperbola2 = [12, 18]; % Position of the second focus
a_Hyperbola2 = 3.5; % Half-length of transverse axis

%% Preparation before plotting the 2-nd hyperbola
[x_LeftBranch_Hyperbola2, y_LeftBranch_Hyperbola2, ...
 x_RightBranch_Hyperbola2, y_RightBranch_Hyperbola2, ...
 theta_Hyperbola2, ...
 b_Hyperbola2, ...
 h_Hyperbola2, k_Hyperbola2] = collect_points_on_Hyperbola_given_y_values(Focus1_Hyperbola2, ...
                                                                          Focus2_Hyperbola2, ...
                                                                          a_Hyperbola2, ...
                                                                          y_tilde_Range);

% Check if (x, y) lies on Hyperbola 2
some_x = x_LeftBranch_Hyperbola2(1);
some_y = y_LeftBranch_Hyperbola2(1);
equal_to_1 = checking_if_a_point_is_on_the_hyperbola(some_x, some_y, ...
                                                     theta_Hyperbola2, ...
                                                     a_Hyperbola2, b_Hyperbola2, ...
                                                     h_Hyperbola2, k_Hyperbola2)

%% Plot the hyperbolas
figure;
hold on;
% --- Plot the reference point ---
fig(1) = plot(Focus1_Hyperbola1(1), Focus1_Hyperbola1(2), 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'LineWidth', 2, ...
              'DisplayName', 'Focal point'); % Focus 1
% --- Plot the 1st hyperbola ---
fig(2) = plot(x_LeftBranch_Hyperbola1, y_LeftBranch_Hyperbola1, 'b', 'LineStyle', ':', 'LineWidth', 2);  % Left branch
fig(3) = plot(x_RightBranch_Hyperbola1, y_RightBranch_Hyperbola1, 'b', 'LineStyle', ':', 'LineWidth', 2);  % Right branch
fig(4) = plot(Focus2_Hyperbola1(1), Focus2_Hyperbola1(2), 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'LineWidth', 2); % Focus 2 of Hyperbola 1
fig(5) = plot(h_Hyperbola1, k_Hyperbola1, 'b+', 'MarkerSize', 8, 'LineWidth', 2, ...
              'DisplayName', 'Center of the 1-st hyperbola'); % Center
% --- Plot the 2nd hyperbola ---
fig(6) = plot(x_LeftBranch_Hyperbola2, y_LeftBranch_Hyperbola2, 'r', 'LineWidth', 2);  % Left branch
fig(7) = plot(x_RightBranch_Hyperbola2, y_RightBranch_Hyperbola2, 'r', 'LineWidth', 2);  % Right branch
fig(8) = plot(Focus2_Hyperbola2(1), Focus2_Hyperbola2(2), 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'LineWidth', 2); % Focus 2 of Hyperbola 2
fig(9) = plot(h_Hyperbola2, k_Hyperbola2, 'rx', 'MarkerSize', 8, 'LineWidth', 2, ...
              'DisplayName', 'Center of the 2-nd hyperbola'); % Center
%
xlabel('x');
ylabel('y');
xlim([-40, 40]);
ylim([-40, 40]);
pbaspect([1 1 1])
% grid on
% hold off;
legend([fig(1), fig(5), fig(9)], 'Location', 'southwest'); % Only display the legends of these curves

%% Preparation before finding the INTERSECTION POINTS of the 2 hyperbolas
% Pack the positions of the three receivers (i.e., the three focal points)
x1 = Focus1_Hyperbola1(1); y1 = Focus1_Hyperbola1(2);
x2 = Focus2_Hyperbola1(1); y2 = Focus2_Hyperbola1(2);
x3 = Focus2_Hyperbola2(1); y3 = Focus2_Hyperbola2(2);
positions_of_focal_points = [x1, y1, x2, y2, x3, y3];

% Define the range for initial guesses (grid and random)
xRange = [-10, 20]; % Range for x-coordinate of the initial guess
yRange = [-10, 20]; % Range for y-coordinate of the initial guess
numGridPoints = 20; % Number of grid points along each axis
numRandomGuesses = 50; % Number of random initial guesses

% Generate grid-based initial guesses
[xGrid, yGrid] = meshgrid(linspace(xRange(1), xRange(2), numGridPoints), ...
                          linspace(yRange(1), yRange(2), numGridPoints));
grid_guesses = [xGrid(:), yGrid(:)];

% Generate random initial guesses
random_guesses = [rand(numRandomGuesses, 1) * (xRange(2) - xRange(1)) + xRange(1), ...
                  rand(numRandomGuesses, 1) * (yRange(2) - yRange(1)) + yRange(1)];

% Combine grid and random guesses
initial_guesses = [grid_guesses; random_guesses];
initial_guesses = unique(initial_guesses, 'rows'); % Remove duplicates

%% Solve the system of 2 hyperbolic equations ... 
% ... by converting it to nonlinear least squares (NLS) problems 
% ... and using the Levenberg-Marquardt algorithm to solve the NLS problems
solutions = solve_hyperbolas_lsqnonlin(positions_of_focal_points, a_Hyperbola1, a_Hyperbola2, initial_guesses);

% Fine-tune solutions to remove duplicates based on tolerance
solutions = remove_duplicate_solutions(solutions, 1e-3);

%% Display solutions with clear message
disp('Solving the system of two hyperbolic equations:');
disp('Each row corresponds to an intersection point (x, y):');
disp('Column 1 is the x-coordinate. Column 2 is the y-coordinate. ');
disp(solutions);
num_solutions = size(solutions,1);

% Plot the hyperbolas along with the intersection points
figure;
hold on;
% --- Plot the reference point ---
fig(1) = plot(Focus1_Hyperbola1(1), Focus1_Hyperbola1(2), 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'LineWidth', 2); % Focus 1
% --- Plot the 1st hyperbola ---
fig(2) = plot(x_LeftBranch_Hyperbola1, y_LeftBranch_Hyperbola1, 'b', 'LineStyle', ':', 'LineWidth', 2);  % Left branch
fig(3) = plot(x_RightBranch_Hyperbola1, y_RightBranch_Hyperbola1, 'b', 'LineStyle', ':', 'LineWidth', 2);  % Right branch
fig(4) = plot(Focus2_Hyperbola1(1), Focus2_Hyperbola1(2), 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'LineWidth', 2); % Focus 2 of Hyperbola 1
fig(5) = plot(h_Hyperbola1, k_Hyperbola1, 'b+', 'MarkerSize', 8, 'LineWidth', 2); % Center
% --- Plot the 2nd hyperbola ---
fig(6) = plot(x_LeftBranch_Hyperbola2, y_LeftBranch_Hyperbola2, 'r', 'LineWidth', 2);  % Left branch
fig(7) = plot(x_RightBranch_Hyperbola2, y_RightBranch_Hyperbola2, 'r', 'LineWidth', 2);  % Right branch
fig(8) = plot(Focus2_Hyperbola2(1), Focus2_Hyperbola2(2), 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'LineWidth', 2); % Focus 2 of Hyperbola 2
fig(9) = plot(h_Hyperbola2, k_Hyperbola2, 'rx', 'MarkerSize', 8, 'LineWidth', 2); % Center
%
xlabel('x');
ylabel('y');
xlim([-40, 40]);
ylim([-40, 40]);
pbaspect([1 1 1])
% grid on
% hold off;
makers = ['s', '>', 'p', 'h'];
if num_solutions ~= 0 
    for sol_idx = 1:num_solutions
        sol = solutions(sol_idx,:);
        sol = round(sol, 1); % We round the vector [111.456, 222.789] up to [111.5, 222.8]
        fig_idx = 9 + sol_idx;
        string = sprintf(' (%0.1f, %0.1f)', sol(1), sol(2));
        string = strcat('Solution ', ' $\approx$', string); 
        fig(fig_idx) = plot(sol(1), sol(2), 'Marker',makers(sol_idx), 'MarkerSize', 8, 'LineStyle', 'none', 'LineWidth', 2, ...
                            'DisplayName', string ...
                            ); % Intersection point
    end
    % Only display the legends of these curves
    legend(fig(10:10+num_solutions-1), 'Location', 'southwest', 'Interpreter', 'latex');
end
title('Finding the intersection points between 2 hyperbolas')

%% Local functions used for calculating basic hyperbolic params and illustrating hyperbolas
% This function is used for 2 purposes:
% ... 1) It returns the (x_tilde, y_tilde) values for illustration
% ... 2) It returns the parameters theta, b, h, k
function [x_LeftBranch, y_LeftBranch, ...
          x_RightBranch, y_RightBranch, ...
          theta,b,h,k ] = collect_points_on_Hyperbola_given_y_values(Focus1, Focus2, a, y_tilde_Range)
    % Find the center 
    x_Focus1 = Focus1(1); 
    y_Focus1 = Focus1(2);
    x_Focus2 = Focus2(1); 
    y_Focus2 = Focus2(2);
    h = (x_Focus1 + x_Focus2) / 2;
    k = (y_Focus1 + y_Focus2) / 2;
    % Calculate the distance between the foci
    d = sqrt((x_Focus2 - x_Focus1)^2 + (y_Focus2 - y_Focus1)^2); 
    % The distance from the center to each focus
    c = d / 2; 
    
    % Validate parameters
    if c < a
        error("Invalid parameters: c must be greater than or equal to a.");
    end
    
    % Calculate b from a and c
    b = sqrt(c^2 - a^2);
    
    % Generate y_tilde values within the given range
    step_size = 0.01;
    y_tilde = y_tilde_Range(1):step_size:y_tilde_Range(2);
    
    % Calculate x values for the valid y values using the hyperbola equation
    x_tilde_LeftBranch = - a * sqrt(1 + y_tilde.^2 / (b^2));  % Left branch
    x_tilde_RightBranch = + a * sqrt(1 + y_tilde.^2 / (b^2));  % Right branch
    
    % Calculate the rotation angle
    % theta = atan2(y_Focus2 - y_Focus1, x_Focus2 - x_Focus1);
    theta = atan( (y_Focus2 - y_Focus1) / (x_Focus2 - x_Focus1) );
    
    % Rotate the hyperbola
    [x_LeftBranch, y_LeftBranch] = Hyperbola_from_xyTILDE_system_to_xy_system(x_tilde_LeftBranch, y_tilde, h, k, theta);
    [x_RightBranch, y_RightBranch] = Hyperbola_from_xyTILDE_system_to_xy_system(x_tilde_RightBranch, y_tilde, h, k, theta);
end

% Given (x_tilde, y_tilde), we can calculate (x, y) values
function [x, y] = Hyperbola_from_xyTILDE_system_to_xy_system(x_tilde, y_tilde, h, k, theta)
    % Rotate the (x_tilde, y_tilde) system to the (x_hat, y_hat) system 
    x_hat = cos(theta) * x_tilde - sin(theta) * y_tilde;
    y_hat = sin(theta) * x_tilde + cos(theta) * y_tilde;

    % Shift the (x_hat, y_hat) system to the (x, y) system 
    x = x_hat + h;
    y = y_hat + k;
end

% This function checks if a point (x, y) is on a specific hyperbola
function equal_to_1 = checking_if_a_point_is_on_the_hyperbola(x, y, theta, a, b, h, k)
    x_hat = x - h; % Shift x to x_hat
    y_hat = y - k; % Shift y to y_hat
    a_column_vector = [x_hat;
                       y_hat];
    RotationMatrix_from_xTilde_to_xHat = [cos(theta), sin(theta);
                                          -sin(theta), cos(theta)];
    Ab = mtimes(RotationMatrix_from_xTilde_to_xHat, ...
                a_column_vector); % c = A x b is a column vector
    x_tilde = Ab(1);
    y_tilde = Ab(2);
    % x_tilde = x_hat * cos(theta) + y_hat * sin(theta);
    % y_tilde = x_hat * (-sin(theta)) + y_hat * cos(theta);

    equal_to_1 = x_tilde^2 / (a^2) - y_tilde^2 / (b^2);
end

%% Local functions used for finding the intersection points between 2 hyperbolas
% This function returns a column vector containing the errors for two hyperbolas
% ... What are the errors of hyperbolas? 
% ... The errors are defined for the purpose of solving a system of 2 equations.
% ... Firstly, we convert it into a Nonlinear Least Squares (NLS) problems 
% ... Secondly, we use a method for solving the NLS problems.
% ... The method we will use is the Levenberg-Marquardt algorithm.
% ... The algorithm is executed through the Matlab function: lsqnonlin
% ... Look at the next local function for more information.
function F = sum_of_errors(vector_containing_x_and_y, positions_of_focal_points, a_Hyperbola1, a_Hyperbola2)
    x = vector_containing_x_and_y(1); 
    y = vector_containing_x_and_y(2);
    
    % Extract parameters
    x1 = positions_of_focal_points(1);
    y1 = positions_of_focal_points(2); 
    x2 = positions_of_focal_points(3); 
    y2 = positions_of_focal_points(4); 
    x3 = positions_of_focal_points(5); 
    y3 = positions_of_focal_points(6);

    % Calculate the parameters of Hyperbola 1
    h1 = 0.5 * (x1 + x2);
    k1 = 0.5 * (y1 + y2);
    a1 = a_Hyperbola1;
    c1 = 0.5 * sqrt((x2 - x1)^2 + (y2 - y1)^2);
    b1 = sqrt(c1^2 - a_Hyperbola1^2);
    theta1 = atan2(y2 - y1, x2 - x1);

    % Calculate the parameters of Hyperbola 2
    h2 = 0.5 * (x1 + x3);
    k2 = 0.5 * (y1 + y3);
    a2 = a_Hyperbola2; 
    c2 = 0.5 * sqrt((x3 - x1)^2 + (y3 - y1)^2);
    b2 = sqrt(c2^2 - a_Hyperbola2^2);
    theta2 = atan2(y3 - y1, x3 - x1);

    % Error for Hyperbola 1
    term1 = ((x - h1) * cos(theta1) + (y - k1) * sin(theta1))^2 / a1^2;
    term2 = ((x - h1) * (-sin(theta1)) + (y - k1) * cos(theta1))^2 / b1^2;
    error1 = term1 - term2 - 1;
    
    % Error for Hyperbola 2
    term1 = ((x - h2) * cos(theta2) + (y - k2) * sin(theta2))^2 / a2^2;
    term2 = ((x - h2) * (-sin(theta2)) + (y - k2) * cos(theta2))^2 / b2^2;
    error2 = term1 - term2 - 1;
    
    % Return errors for both hyperbolas (sum of squared errors)
    F = [error1; error2];  % Return errors as a vector
end

% This function executes the Levenberg-Marquardt algorithm for solving the
% ... NLS problems. Note that the errors for the NLS problems have already 
% ... been defined by the previous local function. 
function solutions = solve_hyperbolas_lsqnonlin(positions_of_focal_points, a_Hyperbola1, a_Hyperbola2, initial_guesses)
    options = optimoptions('lsqnonlin', 'Display', 'off', 'TolFun', 1e-5, 'TolX', 1e-5);
    solutions = [];
    for i = 1:size(initial_guesses, 1)
        x0 = initial_guesses(i, :);
        sol = lsqnonlin(@(xy) sum_of_errors(xy, positions_of_focal_points, a_Hyperbola1, a_Hyperbola2), x0, [], [], options);
        solutions = [solutions; sol]; %#ok
    end
    solutions = unique(round(solutions, 5), 'rows'); % Deduplicate solutions
end

% This function removes some duplicate solutions based on a tolerance
% ... For example, we have solutions = [x=1, y=1;
% ...                                   x=1, y=1;
% ...                                   x=1, y=1;
% ...                                   x=2, y=3];
% ... There are 2 distinct solutions (x=1, y=1) and (x=2, y=3). 
% ... But, (x=1, y=1) is repeated several times. 
% ... This function simplifies the solutions to the following: [x=1, y=1;
% ...                                                           x=2, y=3];
function solutions = remove_duplicate_solutions(solutions, tolerance)
    % Initialize the filtered solutions
    filtered_solutions = solutions(1, :);
    
    for i = 2:size(solutions, 1)
        % Compare the distance between the current solution and the last valid solution
        dist = norm(solutions(i, :) - filtered_solutions(end, :));
        
        % If the distance is greater than the tolerance, keep the point
        if dist > tolerance
            filtered_solutions = [filtered_solutions; solutions(i, :)]; %#ok
        end
    end
    
    % Return the filtered solutions
    solutions = filtered_solutions;
end
