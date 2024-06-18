modes=(mixed2 fix flex)
envs=(indoors outdoors)
dur=${1-1800}

for env in ${envs[*]}; do
    for mode in ${modes[*]}; do
        bash start_work.sh $mode "rosrun agrnav inference_ros.py" $env robot2_torch13 $dur agrnav
    done
done
echo "Running local cases..."
    bash start_work.sh flex "rosrun agrnav inference_ros.py" indoors robot2_torch13 300 agrnav_local False

