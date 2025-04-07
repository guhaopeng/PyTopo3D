%% Run MATLAB Benchmarks for Large-Scale Problems
% This script runs benchmarks on the MATLAB 3D topology optimization code
% for large-scale problems to address reviewer comments about scaling performance

% Create output directory
output_dir = 'matlab_benchmarks_large';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Define benchmark problem sizes - 8x the elements for each doubling
% - 32x16x16 = 8,192 elements (original paper)
% - 64x32x32 = 65,536 elements (8x larger)
% - 128x64x64 = 524,288 elements (64x larger)
problem_sizes = [
    struct('nelx', 32, 'nely', 16, 'nelz', 16, 'elements', 32*16*16), ...
    struct('nelx', 64, 'nely', 32, 'nelz', 32, 'elements', 64*32*32), ...
    struct('nelx', 128, 'nely', 64, 'nelz', 64, 'elements', 128*64*64)
];

% Select which problems to run (default is all, but can be modified)
problem_indices = 1:length(problem_sizes);
run_problem_indices = [1, 2, 3]; % Change this to select specific problems

% Common parameters
volfrac = 0.3;
penal = 3.0;
rmin = 4.0;  % Scale this for larger problems if needed

% Adjust parameters for large problems
max_iterations_by_size = [200, 100, 50]; % Reduce iterations for larger problems
tolerance_by_size = [0.01, 0.02, 0.05]; % Increase tolerance for larger problems

% Loop through selected problem sizes
for i = run_problem_indices
    if i > length(problem_sizes)
        fprintf('Invalid problem index: %d\n', i);
        continue;
    end
    
    problem = problem_sizes(i);
    nelx = problem.nelx;
    nely = problem.nely;
    nelz = problem.nelz;
    elements = problem.elements;
    
    % Adjust parameters based on problem size
    maxloop = max_iterations_by_size(min(i, length(max_iterations_by_size)));
    tolx = tolerance_by_size(min(i, length(tolerance_by_size)));
    
    fprintf('\n========================================================\n');
    fprintf('Running benchmark for size %d x %d x %d (%d elements)\n', nelx, nely, nelz, elements);
    fprintf('Parameters: volfrac=%.2f, penal=%.2f, rmin=%.2f\n', volfrac, penal, rmin);
    fprintf('Iterations: %d, Tolerance: %.4f\n', maxloop, tolx);
    fprintf('========================================================\n\n');
    
    % Create output file path
    output_file = fullfile(output_dir, sprintf('benchmark_size_%d.csv', elements));
    
    % Run benchmark with modified maxloop and tolx parameters
    benchmark_top3d_modified(nelx, nely, nelz, volfrac, penal, rmin, output_file, maxloop, tolx);
    
    % Force garbage collection after large problems
    clear functions;
    pack; % Consolidate memory
    pause(2); % Give system time to recover
end

% Create combined results file
try
    combine_large_benchmark_results(output_dir, problem_sizes);
catch e
    fprintf('Error combining results: %s\n', e.message);
end

fprintf('\nAll benchmarks complete.\n');
fprintf('To compare with Python implementation run:\n');
fprintf('python scripts/run_benchmark.py --matlab-data=%s/matlab_benchmarks.json --run-python\n', output_dir);

% ----- Functions -----

function benchmark_top3d_modified(nelx, nely, nelz, volfrac, penal, rmin, output_file, maxloop, tolx)
    % This is a wrapper function that calls benchmark_top3d with additional parameters
    % to control max iterations and tolerance
    
    % Call the benchmark function with maxloop and tolx
    benchmark_top3d_with_params(nelx, nely, nelz, volfrac, penal, rmin, output_file, maxloop, tolx);
end

function combine_large_benchmark_results(output_dir, problem_sizes)
    % Create combined CSV file
    combined_file = fullfile(output_dir, 'matlab_benchmarks_combined.csv');
    
    % Write header
    fid = fopen(combined_file, 'w');
    fprintf(fid, 'Size_X,Size_Y,Size_Z,Elements,Time,Memory,Iterations\n');
    
    % Process each benchmark file
    for i = 1:length(problem_sizes)
        problem = problem_sizes(i);
        nelx = problem.nelx;
        nely = problem.nely;
        nelz = problem.nelz;
        elements = problem.elements;
        
        % Read benchmark file
        file_path = fullfile(output_dir, sprintf('benchmark_size_%d.csv', elements));
        
        if exist(file_path, 'file')
            try
                % Read first line (header)
                f = fopen(file_path, 'r');
                header = fgetl(f);
                % Read second line (values)
                values_line = fgetl(f);
                fclose(f);
                
                % Parse values
                values = strsplit(values_line, ',');
                
                % Extract time, memory and iterations
                time_idx = find(strcmpi(strsplit(header, ','), 'Time'));
                memory_idx = find(strcmpi(strsplit(header, ','), 'Memory'));
                iterations_idx = find(strcmpi(strsplit(header, ','), 'Iterations'));
                
                if ~isempty(time_idx) && ~isempty(memory_idx) && ~isempty(iterations_idx)
                    time = str2double(values{time_idx});
                    memory = str2double(values{memory_idx});
                    iterations = str2double(values{iterations_idx});
                    
                    % Write to combined file
                    fprintf(fid, '%d,%d,%d,%d,%f,%f,%d\n', nelx, nely, nelz, elements, time, memory, iterations);
                end
            catch e
                fprintf('Error processing file %s: %s\n', file_path, e.message);
            end
        end
    end
    
    fclose(fid);
    fprintf('Combined benchmark results saved to %s\n', combined_file);
    
    % Create JSON format for Python comparison
    try
        result_data = struct();
        
        % Read combined CSV
        csv_data = readtable(combined_file);
        
        % Convert to JSON structure
        for i = 1:height(csv_data)
            nelx = csv_data.Size_X(i);
            nely = csv_data.Size_Y(i);
            nelz = csv_data.Size_Z(i);
            elements = csv_data.Elements(i);
            key = num2str(elements);
            
            result_data.(['n' key]) = struct(...
                'total_time_seconds', csv_data.Time(i), ...
                'peak_memory_mb', csv_data.Memory(i), ...
                'iterations', csv_data.Iterations(i), ...
                'problem_size', struct('nelx', nelx, 'nely', nely, 'nelz', nelz) ...
            );
        end
        
        % Save JSON file
        json_file = fullfile(output_dir, 'matlab_benchmarks.json');
        json_str = jsonencode(result_data);
        fid = fopen(json_file, 'w');
        fprintf(fid, '%s', json_str);
        fclose(fid);
        fprintf('Benchmark results saved in Python-compatible JSON format to %s\n', json_file);
    catch e
        fprintf('Error creating JSON file: %s\n', e.message);
    end
end 