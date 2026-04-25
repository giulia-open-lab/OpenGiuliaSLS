%==========================================================================
% David Lopez Perez
% UPV
%==========================================================================

%--------------------------------------------------------------------------
% Function generates a map of spatially correlated shadow fading values in 
% decibels in form of a 2D matrix
%--------------------------------------------------------------------------

function G_shadow_map_dB = shadow_map(varargin)

if nargin == 5
    nrows = varargin{1};
    ncols = varargin{2};
    resolution_m = varargin{3};
    alpha_vec = 1./varargin{4};  %alpha = 1/20;
    sigma_shadow_dB = varargin{5};    % generates multiple maps if sigma_shadow_dB is vector

elseif nargin == 6
    nrows = varargin{1};
    ncols = varargin{2};
    resolution_m = varargin{3};
    alpha_vec = 1./varargin{4};  %alpha = 1/20;
    sigma_shadow_dB = varargin{5};    % generates multiple maps if sigma_shadow_dB is vector 
    seed = varargin{6};     
    if size(sigma_shadow_dB) ~= size(seed)
        error('sigma_shadow_dB and seed need to have the same dimensions')
    end
else
    error('Wrong number of arguments')
end

nmaps = max(size(sigma_shadow_dB));

G_shadow_map_dB = zeros(nrows, ncols, nmaps); 
for m = 1:nmaps

    %alpha = 1/20;
    if (m == 1 || alpha_vec(m) ~= alpha_vec(m-1))

        alpha = alpha_vec(m);
        correlation = exp(-(alpha * resolution_m));
        
        r_1   = exp(-(alpha * resolution_m)); 
        r_s2  = exp(-(alpha * resolution_m * sqrt(2)));
        r_2   = exp(-(alpha * resolution_m * 2));
        r_s5  = exp(-(alpha * resolution_m * sqrt(5)));
        r_s8  = exp(-(alpha * resolution_m * sqrt(8)));
        r_3   = exp(-(alpha * resolution_m * 3));
        r_s10 = exp(-(alpha * resolution_m * sqrt(10)));
        r_4   = exp(-(alpha * resolution_m * 4));
        
        R2 = [ 1     r_1;...
               r_1   1    ];
        R2_12 = chol(R2)';
        
        R5  = [ 1      r_1    r_2    r_1    r_s2;...
                r_1    1      r_1    r_s2   r_1;...
                r_2    r_1    1      r_s5   r_s2;...
                r_1    r_s2   r_s5   1      r_1;...
                r_s2   r_1    r_s2   r_1    1];
        R5_12 = chol(R5)';
        R5x = R5(1:4, 1:4);
        R5x_12 = chol(R5x)';
        inv_R5x_12 = inv(R5x_12);
        
        R9  = [ 1      r_1    r_2    r_1    r_1    r_s5   r_1    r_3    r_s2;...
                r_1    1      r_1    r_s2   r_s2   r_s2   r_2    r_2    r_1;...
                r_2    r_1    1      r_s5   r_s5   r_1    r_3    r_1    r_s2;...
                r_1    r_s2   r_s5   1      r_2    r_s8   r_s2   r_s10  r_1;...
                r_1    r_s2   r_s5   r_2    1      r_2    r_s2   r_s10  r_s5;...
                r_s5   r_s2   r_1    r_s8   r_2    1      r_s10  r_s2   r_s5;...
                r_1    r_2    r_3    r_s2   r_s2   r_s10  1      r_4    r_s5;... 
                r_3    r_2    r_1    r_s10  r_s10  r_s2   r_4    1      r_s5;...
                r_s2   r_1    r_s2   r_1    r_s5   r_s5   r_s5   r_s5   1];
        R9_12 = chol(R9)';
        R9x = R9(1:8,1:8);
        R9x_12 = chol(R9x)';
        inv_R9x_12 = inv(R9x_12);
    end 
        

    if nargin == 5
        stream0 = RandStream('mt19937ar','Seed',seed(m));
        %RandStream.setDefaultStream(stream0);
        RandStream.setGlobalStream(stream0);
    end
    
    G_shadow_map_dB(1,1,m) = randn(1,1);
    for r = 2:size(G_shadow_map_dB,1)
        L_new_dB = randn(1,1);
        y = [G_shadow_map_dB(r-1,1,m); L_new_dB];
        G_shadow_map_dB(r,1,m) = R2_12(2,:) * y;
    end
    for c = 2:size(G_shadow_map_dB,2)
        L_new_dB = randn(1,1);
        y = [G_shadow_map_dB(1,c-1,m); L_new_dB];
        G_shadow_map_dB(1,c,m) = R2_12(2,:) * y;
    end
    for r = 2:size(G_shadow_map_dB,1)
        for c = 2:size(G_shadow_map_dB,2)
            L_new_dB = randn(1,1);
            if (c == size(G_shadow_map_dB,2))
                G_shadow_map_dB(r,c,m) = ((1-correlation)* L_new_dB) + ((correlation*(G_shadow_map_dB(r-1,c,m)) + correlation*(G_shadow_map_dB(r,c-1,m))) / 2 );
            elseif ((r==2) | (c==2) | (c == size(G_shadow_map_dB,2)-1))
                L5x = [ G_shadow_map_dB(r-1,c-1,m); G_shadow_map_dB(r-1,c,m); G_shadow_map_dB(r-1,c+1,m); G_shadow_map_dB(r,c-1,m)];
                y = [inv_R5x_12 * L5x; L_new_dB];
                G_shadow_map_dB(r,c,m) = R5_12(5,:) * y;
            else
                L9x = [ G_shadow_map_dB(r-1,c-1,m); G_shadow_map_dB(r-1,c,m);   G_shadow_map_dB(r-1,c+1,m); G_shadow_map_dB(r,c-1,m);...
                    G_shadow_map_dB(r-2,c-1,m); G_shadow_map_dB(r-2,c+1,m); G_shadow_map_dB(r-1,c-2,m); G_shadow_map_dB(r-1,c+2,m)];
                y = [inv_R9x_12 * L9x; L_new_dB];
                G_shadow_map_dB(r,c,m) = R9_12(9,:) * y;
            end
        end
    end
    
    % scale to required standard deviation
    %G_shadow_map_dB(:,:,m) = G_shadow_map_dB(:,:,m) * ((10^(sigma_shadow_dB(m)/10)) / mean(std(G_shadow_map_dB(:,:,m))));
    G_shadow_map_dB(:,:,m) = G_shadow_map_dB(:,:,m) * sigma_shadow_dB(m) / mean(std(G_shadow_map_dB(:,:,m)));
    
end



