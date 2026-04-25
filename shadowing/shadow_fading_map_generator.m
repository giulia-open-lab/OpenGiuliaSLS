%==========================================================================
% David Lopez Perez
% UPV
%==========================================================================

%--------------------------------------------------------------------------
% Script creates shadowing maps for our python simulator
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Inputs:
% - model = "G_shadow_map_3GPPTR38_901_UMa" or "G_shadow_map_3GPPTR38_901_UMi"
% - sigma_shadow_dB: Shadow fading standard deviation in dB. It can be a vector. The first position is for the LoS and the second for the NLoS 
% - autocorrelation_m: Shadow fading autocorrelation in dB. It can be a vector. The first position is for the LoS and the second for the NLoS 
% - number_of_maps_per_sigma: Number of shadow fading maps created for each standard deviation specified
% - scenario_size_m: Scenario dimensions, e.g. [3000,3000]
% - scenario_resolution_m: Size of each grid point of the square grid in which the scenario is divided
%--------------------------------------------------------------------------
    
    %% Inputs
    
    % Model
    model = "G_shadow_map_3GPPTR38_811_Dense_Urban_NTN"
    number_of_maps_per_sigma = 19*2; % Number of maps to generate per shadow fading standard deviation

    % Scenario
    scenario_size_m = [3000,3000];  % scenario dimensions [2.4*r_macro_m, 2.4*r_macro_m];%*1.7;  % size of the scenario [width, height] [m]
    scenario_resolution_m = 2;      % distance between two shadwoing grid points [m]

    % Plots
    fontsize = 8;
    saveFig = 1;
    
    %% Pre processing 
    
    % Selecting model parameters according to the input 
    if model == "G_shadow_map_ITU_R_M2135_UMa"
        sigma_shadow_dB = [4, 6]; % Shadow fading standard deviations UMa
        autocorrelation_m = [37 50];
        crosscorelation = 0.5;  
    elseif model == "G_shadow_map_ITU_R_M2135_UMi"
        sigma_shadow_dB = [3, 4, 7]; % Shadow fading standard deviations UMa
        autocorrelation_m = [10 13 7];
        crosscorelation = 0.5;          
    elseif model == "G_shadow_map_3GPPTR38_901_UMa"
        sigma_shadow_dB = [4, 6]; % Shadow fading standard deviations UMa
        autocorrelation_m = [37 50];
        crosscorelation = 0.5;
    elseif model == "G_shadow_map_3GPPTR38_901_UMi"
        sigma_shadow_dB = [4, 7.82]; % Shadow fading standard deviations UMi
        autocorrelation_m = [10 13];
        crosscorelation = 0.5;
    elseif model == "G_shadow_map_3GPPTR36_814_Case_1"
        sigma_shadow_dB = [8, 8]; % Shadow fading standard deviations UMi
        autocorrelation_m = [50 50];
        crosscorelation = 0.5;
    elseif model == "G_shadow_map_3GPPTR36_777_UMa_AV_50m"
        sigma_shadow_dB = [4, 6, 4.64*exp(-0.0066*50)]; % Shadow fading standard deviations UMi
        autocorrelation_m = [37 50 37];
        crosscorelation = 0.5;          
    elseif model == "G_shadow_map_3GPPTR36_777_UMa_AV_100m"
        sigma_shadow_dB = [4, 6, 4.64*exp(-0.0066*100)]; % Shadow fading standard deviations UMi
        autocorrelation_m = [37 50 37];
        crosscorelation = 0.5;  
    elseif model == "G_shadow_map_3GPPTR36_777_UMa_AV_200m"
        sigma_shadow_dB = [4, 6, 4.64*exp(-0.0066*200)]; % Shadow fading standard deviations UMi
        autocorrelation_m = [37 50 37];
        crosscorelation = 0.5;
    elseif model == "G_shadow_map_3GPPTR36_777_UMa_AV_300m"
        sigma_shadow_dB = [4, 6, 4.64*exp(-0.0066*300)]; % Shadow fading standard deviations UMi
        autocorrelation_m = [37 50 37];
        crosscorelation = 0.5;

    elseif model == "G_shadow_map_3GPPTR38_811_Dense_Urban_NTN"
        sigma_shadow_dB = [3.5, 2.9, 3.0, 3.1, 2.7, 2.5, 2.3, 1.2, 15.5, 13.9, 12.4, 11.7, 10.6, 10.5, 10.1, 9.2]; % Shadow fading standard deviations UMi
        autocorrelation_m = [37 37 37 37 37 37 37 37 50 50 50 50 50 50 50 50];
        crosscorelation = 0.5;            
        

    end 

    % Calculate scenario limits
    scenario_elements = floor(scenario_size_m/scenario_resolution_m)+1;
    scenario_offset_m = -[scenario_size_m(1)/2, scenario_size_m(2)/2];
    coord_map_x = repmat(scenario_offset_m(1):scenario_resolution_m:scenario_elements(1)*scenario_resolution_m+scenario_offset_m(1)-scenario_resolution_m,scenario_elements(2),1)';
    coord_map_y = repmat(scenario_offset_m(2):scenario_resolution_m:scenario_elements(2)*scenario_resolution_m+scenario_offset_m(2)-scenario_resolution_m,scenario_elements(1),1);

    % Adapt shadow fading inputs 
    % Create a vector with as many positions as shadowing maps, with their
    %respective standard deviations
    number_of_maps_per_sigma_plus = number_of_maps_per_sigma + 1; % we add the common map
    total_number_of_maps = number_of_maps_per_sigma_plus * size(sigma_shadow_dB,2);

    sigma_shadow_vector_dB = ones(number_of_maps_per_sigma_plus,1) * sigma_shadow_dB(1);
    autocorrelation_vector_m = ones(number_of_maps_per_sigma_plus,1) * autocorrelation_m(1);
    for m=2:size(sigma_shadow_dB,2)
        sigma_shadow_vector_dB = [sigma_shadow_vector_dB; ones(number_of_maps_per_sigma_plus,1) * sigma_shadow_dB(m)];
        autocorrelation_vector_m = [autocorrelation_vector_m; ones(number_of_maps_per_sigma_plus,1) * autocorrelation_m(m)];
    end 
    
    % Create seeds
    scenario_seed = 1; %scenario seed
    seed_vector = (scenario_seed * 1000 + [1:1:total_number_of_maps] + 1)';
      
    %% Generate independent shadow fading map
    G_shadow_temp = shadow_map(scenario_elements(1), scenario_elements(2), scenario_resolution_m, autocorrelation_vector_m, sigma_shadow_vector_dB, seed_vector);
    
    %% Generate final correlated maps
    G_shadow_map_dB = zeros(size(G_shadow_temp,1), size(G_shadow_temp,2), size(sigma_shadow_dB,2)*number_of_maps_per_sigma);
    for sigma_batch = 1:size(sigma_shadow_dB,2)
        
        % We select the common map
        index_first_of_batch = sigma_batch + (sigma_batch-1)*(number_of_maps_per_sigma);
        G_shadow_map_macro_common_dB = G_shadow_temp(:, :, index_first_of_batch); % We set the first map as common map for cross-correlation
        
        % We pass the common map through all the maps with the same
        % standard deviation
        for map_within_sigma = 1:number_of_maps_per_sigma
            
            %Shadow fading values with 0.5 correlation for macrocells
            index_within_batch =  map_within_sigma + index_first_of_batch;
            G_shadow_map_dB(:,:, map_within_sigma + (sigma_batch-1) * (number_of_maps_per_sigma) ) = crosscorelation*sqrt(2)*G_shadow_map_macro_common_dB + (1-crosscorelation)*sqrt(2)*G_shadow_temp(:, :, index_within_batch);
        end 

        figure(140 + sigma_batch)
        clf
        %colormap('jet') 
           cmap = colormap(gray)+0.15;%+0.25;
            %cmap(1,:)=1;
            cmap2 = zeros(size(cmap));
            for i=1:size(cmap,1)
                cmap2(size(cmap,1)-i+1,:) = cmap(i,:);
            end 
            cmap2 = min(cmap2,1);
            cmap2 = max(cmap2,0);
            colormap(cmap2)
        set(gca,'FontSize',fontsize);
        surf(coord_map_x, coord_map_y, G_shadow_map_macro_common_dB,'EdgeColor','none'); 
        lighting phong
        axis equal;
        colorbar
        view(0,90)
        box on;
        title('Shadow fading map for urban macrocell [dB]')
        xlabel('x[m]');
        ylabel('y[m]');
        
        if(saveFig == 1)
             str_saveas = sprintf('f_shadowing_%s_%d', model, round(sigma_batch));
             saveas(gcf,str_saveas, 'fig');
        end
        
        % plot pdf of shadow fading values
        res_pdf = 0.2;
        PDF_values_dB = [-30:res_pdf:30];
        PDF = pdf(G_shadow_map_dB(:,:,index_first_of_batch+1), PDF_values_dB);

        %std_dev_dB = 10*log10(mean(std(G_shadow_map_dB(:,:,index_first_of_batch+1))));
        std_dev_dB = mean(std(G_shadow_map_dB(:,:,index_first_of_batch+1)));
        
        figure(150 + sigma_batch)
        subplot(2,1,1)
        plot(PDF_values_dB, PDF);
        text(4.5,max(PDF)*0.8,['Standard deviation = ',num2str(std_dev_dB),' dB' ])
        title(['PDF of shadow fading gains (resolution = ', num2str(res_pdf),' dB)'])
        xlabel('Shadow fading gain [dB]');
        ylabel('PDF');
        
        % plot correlation factor - from function G_shadow_map_dB.m
        d_m = [0:1:100]; 
        alpha = 1/ autocorrelation_m(sigma_batch);
        correlation = exp(-(alpha * d_m));
        
        %figure(11)
        subplot(2,1,2)
        plot(d_m, correlation);
        title('Shadow fading correlation as a function of distance')
        xlabel('x[m]');
        ylabel('correlation factor');

    end

    %% Add small random values to randomize UE cell selection
    G_shadow_map_dB = round(G_shadow_map_dB + randn(scenario_elements(1),scenario_elements(2), size(sigma_shadow_dB,2)*number_of_maps_per_sigma)*0.0001,2);
    
    %% Save results 
    str_saveas = sprintf('%s.mat', model); 
    save(str_saveas, 'autocorrelation_m', 'crosscorelation', 'scenario_offset_m', 'scenario_resolution_m', 'sigma_shadow_dB','number_of_maps_per_sigma', 'G_shadow_map_dB');
