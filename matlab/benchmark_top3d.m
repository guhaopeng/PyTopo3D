% MODIFIED 3D TOPOLOGY OPTIMIZATION CODE BY LIU AND TOVAR WITH BENCHMARKING
function benchmark_top3d(nelx, nely, nelz, volfrac, penal, rmin, output_file)
% This function is a modified version of the original top3d.m for benchmarking
% 
% Additional parameters:
%   output_file: Path to save benchmark results in CSV format

% Get system info
system_info = get_system_info();

% Start timing
fprintf('Starting benchmark with size: %d x %d x %d (total elements: %d)\n', ...
    nelx, nely, nelz, nelx*nely*nelz);
tic;

% USER-DEFINED LOOP PARAMETERS
maxloop = 200;    % Maximum number of iterations
tolx = 0.001;      % Terminarion criterion
displayflag = 0;  % Display structure flag

% Memory tracking
memory_before = get_memory_usage();

% USER-DEFINED MATERIAL PROPERTIES
E0 = 1;           % Young's modulus of solid material
Emin = 1e-9;      % Young's modulus of void-like material
nu = 0.3;         % Poisson's ratio

% USER-DEFINED LOAD DOFs
[il,jl,kl] = meshgrid(nelx, 0, 0:nelz);                 % Coordinates
loadnid = kl*(nelx+1)*(nely+1)+il*(nely+1)+(nely+1-jl); % Node IDs
loaddof = 3*loadnid(:) - 1;                             % DOFs

% USER-DEFINED SUPPORT FIXED DOFs
[iif,jf,kf] = meshgrid(0,0:nely,0:nelz);                  % Coordinates
fixednid = kf*(nelx+1)*(nely+1)+iif*(nely+1)+(nely+1-jf); % Node IDs
fixeddof = [3*fixednid(:); 3*fixednid(:)-1; 3*fixednid(:)-2]; % DOFs

% PREPARE FINITE ELEMENT ANALYSIS
nele = nelx*nely*nelz;
ndof = 3*(nelx+1)*(nely+1)*(nelz+1);
F = sparse(loaddof,1,-1,ndof,1);
U = zeros(ndof,1);
freedofs = setdiff(1:ndof,fixeddof);
KE = lk_H8(nu);
nodegrd = reshape(1:(nely+1)*(nelx+1),nely+1,nelx+1);
nodeids = reshape(nodegrd(1:end-1,1:end-1),nely*nelx,1);
nodeidz = 0:(nely+1)*(nelx+1):(nelz-1)*(nely+1)*(nelx+1);
nodeids = repmat(nodeids,size(nodeidz))+repmat(nodeidz,size(nodeids));
edofVec = 3*nodeids(:)+1;
edofMat = repmat(edofVec,1,24)+ ...
    repmat([0 1 2 3*nely + [3 4 5 0 1 2] -3 -2 -1 ...
    3*(nely+1)*(nelx+1)+[0 1 2 3*nely + [3 4 5 0 1 2] -3 -2 -1]],nele,1);
iK = reshape(kron(edofMat,ones(24,1))',24*24*nele,1);
jK = reshape(kron(edofMat,ones(1,24))',24*24*nele,1);

% Time the filter construction
t_filter_start = toc;
% PREPARE FILTER
iH = ones(nele*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for k1 = 1:nelz
    for i1 = 1:nelx
        for j1 = 1:nely
            e1 = (k1-1)*nelx*nely + (i1-1)*nely+j1;
            for k2 = max(k1-(ceil(rmin)-1),1):min(k1+(ceil(rmin)-1),nelz)
                for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
                    for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                        e2 = (k2-1)*nelx*nely + (i2-1)*nely+j2;
                        k = k+1;
                        iH(k) = e1;
                        jH(k) = e2;
                        sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2+(k1-k2)^2));
                    end
                end
            end
        end
    end
end
H = sparse(iH,jH,sH);
Hs = sum(H,2);
t_filter = toc - t_filter_start;
fprintf('Filter construction time: %.4f seconds\n', t_filter);

% INITIALIZE ITERATION
x = repmat(volfrac,[nely,nelx,nelz]);
xPhys = x; 
loop = 0; 
change = 1;

% Track timing for optimization phases
t_assembly = 0;
t_solve = 0;
t_sensitivity = 0;
t_filter_apply = 0;
t_update = 0;

% START ITERATION
while change > tolx && loop < maxloop
    loop = loop+1;
    
    % FE-ANALYSIS (Assembly)
    t_start = toc;
    sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),24*24*nele,1);
    K = sparse(iK,jK,sK); K = (K+K')/2;
    t_assembly = t_assembly + (toc - t_start);
    
    % Linear system solve
    t_start = toc;
    U(freedofs,:) = K(freedofs,freedofs)\F(freedofs,:);
    t_solve = t_solve + (toc - t_start);
    
    % OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    t_start = toc;
    ce = reshape(sum((U(edofMat)*KE).*U(edofMat),2),[nely,nelx,nelz]);
    c = sum(sum(sum((Emin+xPhys.^penal*(E0-Emin)).*ce)));
    dc = -penal*(E0-Emin)*xPhys.^(penal-1).*ce;
    dv = ones(nely,nelx,nelz);
    t_sensitivity = t_sensitivity + (toc - t_start);
    
    % FILTERING AND MODIFICATION OF SENSITIVITIES
    t_start = toc;
    dc(:) = H*(dc(:)./Hs);  
    dv(:) = H*(dv(:)./Hs);
    t_filter_apply = t_filter_apply + (toc - t_start);
    
    % OPTIMALITY CRITERIA UPDATE
    t_start = toc;
    l1 = 0; l2 = 1e9; move = 0.2;
    while (l2-l1)/(l1+l2) > 1e-3
        lmid = 0.5*(l2+l1);
        xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
        xPhys(:) = (H*xnew(:))./Hs;
        if sum(xPhys(:)) > volfrac*nele, l1 = lmid; else l2 = lmid; end
    end
    change = max(abs(xnew(:)-x(:)));
    x = xnew;
    t_update = t_update + (toc - t_start);
    
    % PRINT RESULTS
    fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n',loop,c,mean(xPhys(:)),change);
    
    % PLOT DENSITIES (disabled for benchmarking)
    % if displayflag, clf; display_3D(xPhys); end
end

% Calculate total time and peak memory
total_time = toc;
memory_after = get_memory_usage();
memory_used = memory_after - memory_before;

% Print benchmark summary
fprintf('\nBenchmark Results:\n');
fprintf('Total elements: %d\n', nele);
fprintf('Total time: %.4f seconds\n', total_time);
fprintf('Assembly time: %.4f seconds (%.1f%%)\n', t_assembly, 100*t_assembly/total_time);
fprintf('Solve time: %.4f seconds (%.1f%%)\n', t_solve, 100*t_solve/total_time);
fprintf('Sensitivity time: %.4f seconds (%.1f%%)\n', t_sensitivity, 100*t_sensitivity/total_time);
fprintf('Filter apply time: %.4f seconds (%.1f%%)\n', t_filter_apply, 100*t_filter_apply/total_time);
fprintf('Update time: %.4f seconds (%.1f%%)\n', t_update, 100*t_update/total_time);
fprintf('Filter construction time: %.4f seconds\n', t_filter);
fprintf('Memory used: %.1f MB\n', memory_used);
fprintf('Iterations: %d\n', loop);

% Save benchmark results to CSV if output_file is provided
if nargin >= 7 && ~isempty(output_file)
    % Create benchmark results
    benchmark_data = {
        'Size', nelx, ...
        'Elements', nele, ...
        'Time', total_time, ...
        'Memory', memory_used, ...
        'Iterations', loop, ...
        'Assembly', t_assembly, ...
        'Solve', t_solve, ...
        'Sensitivity', t_sensitivity, ...
        'FilterApply', t_filter_apply, ...
        'Update', t_update, ...
        'FilterConstruction', t_filter, ...
        'Platform', system_info.platform, ...
        'CPUs', system_info.cpus, ...
        'Memory_GB', system_info.memory_gb ...
    };
    
    % Reshape data for CSV
    benchmark_data = reshape(benchmark_data, 2, [])';
    
    % Write to CSV
    try
        fid = fopen(output_file, 'w');
        % Write header
        for i = 1:size(benchmark_data, 1)
            if i > 1
                fprintf(fid, ',');
            end
            fprintf(fid, '%s', benchmark_data{i, 1});
        end
        fprintf(fid, '\n');
        
        % Write values
        for i = 1:size(benchmark_data, 1)
            if i > 1
                fprintf(fid, ',');
            end
            if ischar(benchmark_data{i, 2})
                fprintf(fid, '%s', benchmark_data{i, 2});
            else
                fprintf(fid, '%g', benchmark_data{i, 2});
            end
        end
        fprintf(fid, '\n');
        fclose(fid);
        fprintf('Benchmark results saved to %s\n', output_file);
    catch e
        fprintf('Error saving benchmark results: %s\n', e.message);
    end
end

% Export results in format compatible with Python benchmarking
if nargin >= 7 && ~isempty(output_file)
    % Create JSON file path by replacing extension with .json
    [filepath, name, ~] = fileparts(output_file);
    json_file = fullfile(filepath, [name '.json']);
    
    % Create benchmark data structure
    json_data = struct();
    json_data.(['n' num2str(nele)]) = struct(...
        'total_time_seconds', total_time, ...
        'peak_memory_mb', memory_used, ...
        'iterations', loop, ...
        'system_info', struct(...
            'platform', system_info.platform, ...
            'processor', 'MATLAB', ...
            'cpu_count', system_info.cpus, ...
            'total_memory_gb', system_info.memory_gb ...
        ), ...
        'phases', struct(...
            'assembly', struct('total_seconds', t_assembly, 'percentage', 100*t_assembly/total_time), ...
            'solve', struct('total_seconds', t_solve, 'percentage', 100*t_solve/total_time), ...
            'sensitivity', struct('total_seconds', t_sensitivity, 'percentage', 100*t_sensitivity/total_time), ...
            'filter', struct('total_seconds', t_filter_apply, 'percentage', 100*t_filter_apply/total_time), ...
            'update', struct('total_seconds', t_update, 'percentage', 100*t_update/total_time) ...
        ) ...
    );
    
    % Export to JSON
    try
        json_str = jsonencode(json_data);
        fid = fopen(json_file, 'w');
        fprintf(fid, '%s', json_str);
        fclose(fid);
        fprintf('Benchmark results saved in Python-compatible format to %s\n', json_file);
    catch e
        fprintf('Error exporting JSON: %s\n', e.message);
        fprintf('To convert CSV to JSON format, you can use the Python converter script later.\n');
    end
end

% Display final result
if displayflag
    clf; 
    display_3D(xPhys);
end

end

% === GENERATE ELEMENT STIFFNESS MATRIX ===
function [KE] = lk_H8(nu)
A = [32 6 -8 6 -6 4 3 -6 -10 3 -3 -3 -4 -8;
    -48 0 0 -24 24 0 0 0 12 -12 0 12 12 12];
k = 1/144*A'*[1; nu];

K1 = [k(1) k(2) k(2) k(3) k(5) k(5);
    k(2) k(1) k(2) k(4) k(6) k(7);
    k(2) k(2) k(1) k(4) k(7) k(6);
    k(3) k(4) k(4) k(1) k(8) k(8);
    k(5) k(6) k(7) k(8) k(1) k(2);
    k(5) k(7) k(6) k(8) k(2) k(1)];
K2 = [k(9)  k(8)  k(12) k(6)  k(4)  k(7);
    k(8)  k(9)  k(12) k(5)  k(3)  k(5);
    k(10) k(10) k(13) k(7)  k(4)  k(6);
    k(6)  k(5)  k(11) k(9)  k(2)  k(10);
    k(4)  k(3)  k(5)  k(2)  k(9)  k(12)
    k(11) k(4)  k(6)  k(12) k(10) k(13)];
K3 = [k(6)  k(7)  k(4)  k(9)  k(12) k(8);
    k(7)  k(6)  k(4)  k(10) k(13) k(10);
    k(5)  k(5)  k(3)  k(8)  k(12) k(9);
    k(9)  k(10) k(2)  k(6)  k(11) k(5);
    k(12) k(13) k(10) k(11) k(6)  k(4);
    k(2)  k(12) k(9)  k(4)  k(5)  k(3)];
K4 = [k(14) k(11) k(11) k(13) k(10) k(10);
    k(11) k(14) k(11) k(12) k(9)  k(8);
    k(11) k(11) k(14) k(12) k(8)  k(9);
    k(13) k(12) k(12) k(14) k(7)  k(7);
    k(10) k(9)  k(8)  k(7)  k(14) k(11);
    k(10) k(8)  k(9)  k(7)  k(11) k(14)];
K5 = [k(1) k(2)  k(8)  k(3) k(5)  k(4);
    k(2) k(1)  k(8)  k(4) k(6)  k(11);
    k(8) k(8)  k(1)  k(5) k(11) k(6);
    k(3) k(4)  k(5)  k(1) k(8)  k(2);
    k(5) k(6)  k(11) k(8) k(1)  k(8);
    k(4) k(11) k(6)  k(2) k(8)  k(1)];
K6 = [k(14) k(11) k(7)  k(13) k(10) k(12);
    k(11) k(14) k(7)  k(12) k(9)  k(2);
    k(7)  k(7)  k(14) k(10) k(2)  k(9);
    k(13) k(12) k(10) k(14) k(7)  k(11);
    k(10) k(9)  k(2)  k(7)  k(14) k(7);
    k(12) k(2)  k(9)  k(11) k(7)  k(14)];
KE = 1/((nu+1)*(1-2*nu))*...
    [ K1  K2  K3  K4;
    K2'  K5  K6  K3';
    K3' K6  K5' K2';
    K4  K3  K2  K1'];
end

% === DISPLAY 3D TOPOLOGY (ISO-VIEW) ===
function display_3D(rho)
[nely,nelx,nelz] = size(rho);
hx = 1; hy = 1; hz = 1;            % User-defined unit element size
face = [1 2 3 4; 2 6 7 3; 4 3 7 8; 1 5 8 4; 1 2 6 5; 5 6 7 8];
set(gcf,'Name','ISO display','NumberTitle','off');
for k = 1:nelz
    z = (k-1)*hz;
    for i = 1:nelx
        x = (i-1)*hx;
        for j = 1:nely
            y = nely*hy - (j-1)*hy;
            if (rho(j,i,k) > 0.5)  % User-defined display density threshold
                vert = [x y z; x y-hx z; x+hx y-hx z; x+hx y z; x y z+hx;x y-hx z+hx; x+hx y-hx z+hx;x+hx y z+hx];
                vert(:,[2 3]) = vert(:,[3 2]); vert(:,2,:) = -vert(:,2,:);
                patch('Faces',face,'Vertices',vert,'FaceColor',[0.2+0.8*(1-rho(j,i,k)),0.2+0.8*(1-rho(j,i,k)),0.2+0.8*(1-rho(j,i,k))]);
                hold on;
            end
        end
    end
end
axis equal; axis tight; axis off; box on; view([30,30]); pause(1e-6);
end

% Helper function to get system information
function info = get_system_info()
    info = struct();
    
    % Get platform information
    if ispc
        info.platform = 'Windows';
    elseif ismac
        info.platform = 'MacOS';
    elseif isunix
        info.platform = 'Linux/Unix';
    else
        info.platform = 'Unknown';
    end
    
    % Get CPU count
    try
        info.cpus = feature('numcores');
    catch
        info.cpus = 1; % Default to 1 if detection fails
    end
    
    % Get memory information
    try
        mem = memory;
        info.memory_gb = mem.MemAvailableAllArrays / (1024^3);
    catch
        info.memory_gb = 0; % Default to 0 if detection fails
    end
end

% Helper function to estimate memory usage
function mem_mb = get_memory_usage()
    try
        mem = memory;
        mem_mb = (mem.MemUsedMATLAB) / (1024*1024); % Convert to MB
    catch
        mem_mb = 0; % Default if detection fails
    end
end

% Function to run batch benchmark with multiple sizes
function run_benchmark_batch(output_dir)
    if nargin < 1
        output_dir = 'matlab_benchmarks';
    end
    
    % Create output directory if it doesn't exist
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Define benchmark sizes
    sizes = [8, 16, 32, 64];
    
    % Run benchmarks for each size
    for i = 1:length(sizes)
        size = sizes(i);
        nelx = size;
        nely = max(1, round(size/2));
        nelz = max(1, round(size/3));
        
        fprintf('\n=== Running benchmark for size %d x %d x %d ===\n', nelx, nely, nelz);
        
        % Create output file path
        output_file = fullfile(output_dir, sprintf('benchmark_size_%d.csv', nelx*nely*nelz));
        
        % Run benchmark
        benchmark_top3d(nelx, nely, nelz, 0.3, 3.0, 4.0, output_file);
    end
    
    % Combine results into a single file
    combine_benchmark_results(output_dir, sizes);
end

% Function to combine benchmark results
function combine_benchmark_results(output_dir, sizes)
    % Create combined CSV file
    combined_file = fullfile(output_dir, 'matlab_benchmarks_combined.csv');
    
    % Write header
    fid = fopen(combined_file, 'w');
    fprintf(fid, 'Size,Elements,Time,Memory\n');
    
    % Process each benchmark file
    for i = 1:length(sizes)
        size = sizes(i);
        nelx = size;
        nely = max(1, round(size/2));
        nelz = max(1, round(size/3));
        elements = nelx * nely * nelz;
        
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
                
                % Extract time and memory
                time_idx = find(strcmpi(strsplit(header, ','), 'Time'));
                memory_idx = find(strcmpi(strsplit(header, ','), 'Memory'));
                
                if ~isempty(time_idx) && ~isempty(memory_idx)
                    time = str2double(values{time_idx});
                    memory = str2double(values{memory_idx});
                    
                    % Write to combined file
                    fprintf(fid, '%d,%d,%f,%f\n', size, elements, time, memory);
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
            elements = csv_data.Elements(i);
            key = num2str(elements);
            
            result_data.(['n' key]) = struct(...
                'total_time_seconds', csv_data.Time(i), ...
                'peak_memory_mb', csv_data.Memory(i), ...
                'matlab_size_param', csv_data.Size(i) ...
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