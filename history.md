### 2024.07.26
 1. network_train.sh에 --resume_path 의 경로를 변경함 다시 원복해야함 (original : --resume_path ./ckpts/updated_initial_checkpoint.ckpt \)
 2. controlnet은 paintnet으로 weight 얼리고 attention만 학습한 버전

### 2024.08.08
 1. inference.sh : inference가 가능하도록 base 코드에서 변경함
 2. inference.py : inference시에 파일 경로명을 SSIM, FID에서 이름이 동일해야하므로 이 이름을 똑같이 정렬해주었음
  122번줄 to_path = opj(save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg") -> to_path = opj(save_dir, f"{fn.split('.')[0]}.jpg")
 3. inference 폴더에 20240707 (stableviton의 weight로 학습한 decoder확장버전), 20240726 (controlnet은 paintnet으로 weight 얼리고 attention만 학습한 버전)을 inference하고, ./evaluation폴더에 fid, ssim 테스트 해서 작성함

### 2024.08.10
 1. attention만 학습 하는게 아니라 controlnet을 학습하도록 설정함(DGIST.yaml 파일에 use_control_net = true). lr : 1e-6
     1) cldm.py에서 41번 줄에 use_control_net=True, 을 추가
     2) 72번 줄에 self.use_control_net = use_control_net 추가
     3) 294번 줄에 
        params = list(self.control_model.parameters())
        print("control model is added")
        을 지우고 아래 내용을 추가함
        if self.use_control_net:
            params = list(self.control_model.parameters())
            print("control model is added")
        else:
            print("control model don't update")
     4) cldm/warping_cldm_network.py도 
 2. 이렇게 바꾸면 network_train.py 에 223줄에 trainer 매개변수로 resume_from_checkpoint=args.resume_path 이걸 지워줘야함. resume 할때만 넣어줘야함
 3. 즉 이게 UNet 추가한 최종 버전인데 1e-5는 어디있지?
 4. cond_stage_config : CLIP
이 다음은 --repaint를 추가해서 해주어야함.


### 2024.08.30
 1. Controlnet에서 cross attention 하는 부분을 spatial attention으로 변경할것

### 2024.09.28
 1. configs/DGIST.yaml 파일안에 아래 부분 수정함

 cond_stage_config:
      target: ldm.modules.image_encoders.dino.FrozenDinoV2Encoder
      weight: weight/dinov2_vitg14_pretrain.pth
      #target: ldm.modules.image_encoders.modules.FrozenCLIPImageEmbedder

 2. 그리고 ./dinov2 폴더 추가함
 3. ldm/modules/image_encoders/dino.py 추가함
 4. weights/dinov2_vitg14_pretrain.pth 추가함
 위 세가지 버전 돌린것이 logs/20240928_Base에 저장함

 5. lr = 1e-05로 해서 위 과정을 추가한것이 logs/20240928_Base 폴더에 저장된것임

 ### 2024.10.09

 6. lr = 1e-04로 해서 위 과정을 추가한것이 logs/20241009_Base 폴더에 저장된것임, 그러나 nan이 떠서 그냥 490epoch에서 꺼버렸음.

 ### 2024.10.14
 7. lr = 1e-06로 해서 위 과정을 추가한것이 logs/20241014_Base 폴더에 저장된것임, 20241025에 종료됨 약 11일 걸림

 ### 2024.10.25
 8. lr = 3e-05로 해서 위 과정을 추가한것이 logs/20241025_Base 폴더에 저장된것임

 ### 2024.11.07
 - inference를 모두 수행하고, SSIM, FID 성능을 구글 스프레드 시트에 정리함
 - 그러나 20240810_Base는 뭐가 문젠지 모르겠으나, inference 자체가 안됨
    - 이 문제는 CLIP encoder로 셋팅해야하는데 지금 configs/DGIST.yaml 파일은 dinov2로 되어 있어서 발생한 문제임
    - DGIST_CLIP.yaml 파일로 실행하면 에러가 없음

### 2024.11.14
 9. lr = 5e-05로 해서 위 과정을 추가한것이 logs/20241114_Base 폴더에 저장된것임

### 2024.11.15
 10. lr = 8e-06로 해서 위 과정을 추가한것이 logs/20241115_Base 폴더에 저장된것임. 이거 근데 하다가 꺼짐