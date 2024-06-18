models=(VGG19_BN DenseNet121 ConvNeXt_Large ConvNeXt_Base RegNet_X_16GF)
modes=(mixed2 fix flex)
envs=(indoors outdoors)
dur=${1-1800}
dur_local=300

for model in ${models[*]}; do
    for env in ${envs[*]}; do
        for mode in ${modes[*]}; do
            bash start_work.sh $mode "python3 /project/ParallelCollaborativeInference/ros_ws/src/torchvision/scripts/run_torchvision.py -a $model -d ImageNet" $env robot2_torch13 $dur torchvision_$model
        done
    done
    echo "Running local cases..."
    bash start_work.sh flex "python3 /project/ParallelCollaborativeInference/ros_ws/src/torchvision/scripts/run_torchvision.py -a $model -d ImageNet --no-offload" indoors robot2_torch13 $dur_local torchvision_${model}_local False
done

