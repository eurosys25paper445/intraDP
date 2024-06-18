models=(MobileNet_V3_Small ResNet101 VGG19_BN)
modes=(fix flex)
envs=(indoors outdoors)
dur=120

for model in ${models[*]}; do
    bash start_work.sh tp "python3 /project/ParallelCollaborativeInference/ros_ws/src/torchvision/scripts/run_torchvision.py -a $model -d ImageNet --no-offload" indoors robot2_torch13 $dur torchvision_${model}_local
done
