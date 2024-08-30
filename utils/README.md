1. compare_layer.py : 두 모델의 layer를 비교함
2. compare_weight.py : 두 모델의 layer 뿐만 아니라 weight를 비교함
3. copy_weight.py : weight를 복사하는 코드 - model.diffusion_model의 weight를 control_model로 복사할수 있음
4. copy_compare_control_model.py : 3번을 거친후 제대로 weight가 복사 되었는지 확인하는 코드
5. remove_layer.py : 내가 새롭게 warping_attention을 만들어주어서, 이전에 있던 weight들은 지워줘야 함.