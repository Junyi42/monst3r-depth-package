
CUDA_VISIBLE_DEVICES=1 python infer_training_set.py --a=0 --b=-1 --dataset_name=Tartanair & CUDA_VISIBLE_DEVICES=1 python infer_training_set.py --a=0 --b=-1 --dataset_name=spring & CUDA_VISIBLE_DEVICES=1 python infer_training_set.py --a=0 --b=-1 --dataset_name=SceneFlow & CUDA_VISIBLE_DEVICES=1 python infer_training_set.py --a=0 --b=-1 --dataset_name=Vkitti & CUDA_VISIBLE_DEVICES=1 python infer_training_set.py --a=0 --b=-1 --dataset_name=PointOdyssey


CUDA_VISIBLE_DEVICES=1 python infer_test_set.py --a=0 --b=10000 --dataset_name=bonn & CUDA_VISIBLE_DEVICES=1 python infer_test_set.py --a=0 --b=10000 --dataset_name=davis & CUDA_VISIBLE_DEVICES=1 python infer_test_set.py --a=0 --b=10000 --dataset_name=sintel & CUDA_VISIBLE_DEVICES=1 python infer_test_set.py --a=0 --b=10000 --dataset_name=tum
