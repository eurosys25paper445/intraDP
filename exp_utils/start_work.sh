mode=${1-"flex"}    # flex means no latency constraint; fix means 1 hz latency constraint
cmd=${2-"rosrun kapao pose_follower.py"}
env=${3-"indoors"}
container=${4-"robot2"}
dur=${5-1800}
workload=${6-"kapao"}
offload=${7-"True"}
log_dir="${mode}_${env}_${workload}"
bag_name=${8-pose_follower_offload.bag}
echo "offload_mode: $mode; env: $env; cmd: $cmd; container: $container; offload: $offload"
echo "Do start replay bandwidth at $env.txt at the server side."

echo "log_dir $work/log/$log_dir"

docker restart $container &> /dev/null
echo "Restarted container for robot control and native torch"

tmux has-session -t offload_exp 2>/dev/null

if [ $? != 0 ]; then
  # Set up your session
    tmux new -t offload_exp -d

    sleep 5
    tmux new-window -t offload_exp -n env
    tmux new-window -t offload_exp -n work
    tmux new-window -t offload_exp -n power
    sleep 3
    tmux send-keys -t offload_exp:power "sudo su" ENTER
fi


if ! pgrep -f roscore &> /dev/null; then
    tmux new-window -t offload_exp -n roscore
    tmux send-keys -t offload_exp:roscore "roscore" ENTER
    sleep 5
    echo "Didn't found roscore. Started roscore on host."
fi

ssh guanxiux@192.168.10.8 "tmux has-session -t replay_bw; if [ \$? != 0 ]; then tmux new -t replay_bw -d; sleep 3; tmux new-window -t replay_bw -n server; tmux new-window -t replay_bw -n bw; fi"
sleep 10
ssh guanxiux@192.168.10.8 "tmux send-keys -t replay_bw:server C-c ENTER; tmux send-keys -t replay_bw:server \"cd \\\$work; mkdir -p \\\$work/log/$log_dir; python3 start_server.py 0.0.0.0 | tee \\\$work/log/$log_dir/server.txt\" ENTER;"

echo "Started server"
# tmux send-keys -t offload_exp:env "docker exec -it $container zsh" ENTER
# tmux send-keys -t offload_exp:env "roslaunch turn_on_wheeltec_robot ros_torch_env.launch" ENTER
# echo "Started wheeltech robot sensors and motion controller"

tmux send-keys -t offload_exp:work "docker exec -it $container zsh" ENTER
tmux send-keys -t offload_exp:work "rm -rf \$work/log/$log_dir; mkdir -p \$work/log/$log_dir; rosparam set /offload_mode $mode; rosparam set /offload_method $mode; rosparam set /offload $offload" ENTER
tmux send-keys -t offload_exp:work "ROS_LOG_DIR=\$work/log/$log_dir $cmd" ENTER
echo "Running $cmd"

echo "Waiting for workload initialization."
while ! rosservice list | grep /Start &> /dev/null
do
    sleep 5
done

while ! rosservice list | grep /Start &> /dev/null
do
    sleep 5
done


while ! rosservice list | grep /Start &> /dev/null
do
    sleep 5
done

echo "Worload initialization finished. Press any key to start."
# read start

tmux send-keys -t offload_exp:power C-c ENTER
tmux send-keys -t offload_exp:power "python3 $work/exp_utils/power_monitor.py _interval:=1 &> $work/log/$log_dir/power_record.txt" ENTER
echo "Started power_monitor"


ssh guanxiux@192.168.10.8 "tmux send-keys -t replay_bw:bw C-c ENTER; tmux send-keys -t replay_bw:bw \"cd \\\$work/exp_utils; sudo python3 replay_bandwidth.py ${env}.txt eno1 1\" ENTER;"

# tmux new-window -t offload_exp -n bw
# tmux send-keys -t offload_exp:bw "python3 $work/exp_utils/replay_bandwidth.py $work/exp_utils/$env.txt &> $work/log/$log_dir/replay_bandwidth.txt" ENTER
# echo "Started replay_bandwidth.py"

rosservice call /Start
echo "Worload started. Killing in ${dur}s."

sleep $dur
#rosnode kill -a
# tmux send-keys -t offload_exp:env C-c ENTER
tmux send-keys -t offload_exp:work C-c ENTER
tmux send-keys -t offload_exp:work C-c ENTER
tmux send-keys -t offload_exp:work C-c ENTER
tmux send-keys -t offload_exp:power C-c ENTER
tmux send-keys -t offload_exp:power C-c ENTER
tmux send-keys -t offload_exp:power C-c ENTER
sudo pkill -f "power_monitor.py"
ssh guanxiux@192.168.10.8 "tmux send-keys -t replay_bw:bw C-c ENTER"
ssh guanxiux@192.168.10.8 "tmux send-keys -t replay_bw:bw C-c ENTER"
ssh guanxiux@192.168.10.8 "tmux send-keys -t replay_bw:bw C-c ENTER"
ssh guanxiux@192.168.10.8 "tmux send-keys -t replay_bw:server C-c ENTER"
ssh guanxiux@192.168.10.8 "tmux send-keys -t replay_bw:server C-c ENTER"
ssh guanxiux@192.168.10.8 "tmux send-keys -t replay_bw:server C-c ENTER"
# tmux send-keys -t offload_exp:bw C-c ENTER
sleep 10
echo "Fin"
echo ""

