clc
clear
%% evaluation on Test Set
addpath('metrics')
%video_name = {'[01] KITTI - City LR_x2','[02] KITTI - Person LR_x2','[03] KITTI - Campus LR_x2','[04] VIRAT Court LR_x2','[05] VIRAT Student Campus LR_x2','[06] VIRAT Full Parking Lot LR_x2','[07] Wide Area LR_x2','[08] Human Interaction LR_x2','[09] Edinburgh Office LR_x2','[10] MMDA Day LR_x2','[11] Archer_s Eye LR_x2','[12] Pasay Bike Incident LR_x2','[13] Bus LR_x2','[14] Convenience Store LR_x2','[15] Retail Store LR_x2','[16] Grocery Theft LR_x2','[17] Abbey Road LR_x2','[18] Wolves Highway LR_x2','[19] Restaurant LR_x2','[20] Halloween LR_x2'};
video_name = {'[01] KITTI - City LR_x2','[02] KITTI - Person LR_x2', '[03] KITTI - Campus LR_x2'}
psnr_set = [];
ssim_set = [];
rmse_set = [];
for idx_video = 1:length(video_name)
    psnr_video = [];
    ssim_video = [];
    rmse_video = [];
    name = char(video_name(idx_video))
    video_path = fullfile('Super Resolution', name)
    a=dir([video_path '/*.png'])
    disp(video_path)
    n=numel(a)-1
    disp(n)
    for idx_frame = 9:n 				% exclude the first and last 2 frames
        img_hr = imread(['E:/Projects/Thesis/Baseline B/SOF-VSR/TIP/data/test/Set/',video_name{idx_video},'/hr/hr_', num2str(idx_frame-9,'%d'),'.png']);
        img_sr = imread(['Super Resolution/',video_name{idx_video},'/sr_', num2str(idx_frame-9,'%d'),'.png']);
        
        h = min(size(img_hr, 1), size(img_sr, 1));
        w = min(size(img_hr, 2), size(img_sr, 2));
        
        border = 6 + 2;
        
        img_hr_ycbcr = rgb2ycbcr(img_hr);
        img_hr_y = img_hr_ycbcr(1+border:h-border, 1+border:w-border, 1);
        img_sr_ycbcr = rgb2ycbcr(img_sr);
        img_sr_y = img_sr_ycbcr(1+border:h-border, 1+border:w-border, 1);

        rmse_video(idx_frame-8) = sqrt(mean((img_hr_y(:)-img_sr_y(:)).^2));
        psnr_video(idx_frame-8) = cal_psnr(img_sr_y, img_hr_y);
        ssim_video(idx_frame-8) = cal_ssim(img_sr_y, img_hr_y);
    end
    psnr_set(idx_video) = mean(psnr_video);
    ssim_set(idx_video) = mean(ssim_video);
    rmse_set(idx_video) = mean(rmse_video);
    disp([video_name{idx_video},'---Mean PSNR: ', num2str(mean(psnr_video),'%0.4f'),', Mean SSIM: ', num2str(mean(ssim_video),'%0.4f'),', Mean RMSE: ', num2str(mean(rmse_video),'%0.4f')])
end
disp(['---------------------------------------------'])
disp(['Set ',' SR---Mean PSNR: ', num2str(mean(psnr_set),'%0.4f'),', Mean SSIM: ', num2str(mean(ssim_set),'%0.4f'),', Mean RMSE: ', num2str(mean(rmse_set),'%0.4f')])