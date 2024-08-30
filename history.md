### 2024.07.25
 1. network_train.sh에 --resume_path 의 경로를 변경함 다시 원복해야함 (original : --resume_path ./ckpts/updated_initial_checkpoint.ckpt \)

### 2024.08.08
 1. inference.sh : inference가 가능하도록 base 코드에서 변경함
 2. inference.py : inference시에 파일 경로명을 SSIM, FID에서 이름이 동일해야하므로 이 이름을 똑같이 정렬해주었음
  122번줄 to_path = opj(save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg") -> to_path = opj(save_dir, f"{fn.split('.')[0]}.jpg")
 3. inference 폴더에 20240707 (stableviton의 weight로 학습한 decoder확장버전), 20240726 (controlnet은 paintnet으로 weight 얼리고 attention만 학습한 버전)을 inference하고, ./evaluation폴더에 fid, ssim 테스트 해서 작성함

### 2024.08.10
 1. attention만 학습 하는게 아니라 controlnet을 학습하도록 설정함(DGIST.yaml 파일에 use_control_net = true). lr : 1e-6
 2. 이렇게 바꾸면 network_train.py 에 223줄에 trainer 매개변수로 resume_from_checkpoint=args.resume_path 이걸 지워줘야함. resume 할때만 넣어줘야함

이 다음은 --repaint를 추가해서 해주어야함.