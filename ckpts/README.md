1. VITONHD.ckpt : 원 저자의 최종 weight
2. VITONHD_PBE_pose.ckpt : 원 저자의 초기 weight
3. VITONHD_VAE_finetuning.ckpt : 원 저자의 autoencoder의 decoder를 재 학습한 weight (AE only)
4. initial_checkpoint.ckpt : CrossAttention, ControlNet의 Decoder of UNet 을 추가하여 Decoder만 초기화 한 weight
5. paintnet_365_20240517.ckpt : paintnet으로 cloth warping만 학습한 weight
6. updated_initial_checkpoint.ckpt : cond_stage를 dinov2로 변경한 weight (추측임 확실치 않음 확실하려면 weight 뜯어서 검증해야함)