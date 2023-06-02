#!bin/bash

embedding=$1
camera=default

# kitchen_env=("micro_close" "micro_open" "rdoor_close" "sdoor_open")

# for env in "${kitchen_env[@]}"
# do  
#     echo "embedding: ${embedding} | task: ${env}"
#     python ./evaluation/trajopt/trajopt/eval_mppi.py \
#         env=kitchen_${env}-v3 embedding=${embedding} camera=${camera} \
#         exp_name=${embedding}-agentago-wois@${env}@${camera} \
#         env_kwargs.agent_ago=True
# done

kitchen_env=("ldoor_open" "micro_close" "micro_open" "rdoor_close" "sdoor_open")
for env in "${kitchen_env[@]}"
do  
    echo "embedding: ${embedding} | task: ${env}"
    python ./evaluation/trajopt/trajopt/eval_mppi.py \
        env=kitchen_${env}-v3 embedding=${embedding} camera=${camera}
done


# embedding=("vip" "r3m" "clip")
# camera=default
# kitchen_env=$1

# for emb in "${embedding[@]}"
# do  
#     echo "embedding: ${emb} | task: ${kitchen_env}"
#     python ./evaluation/trajopt/trajopt/eval_mppi.py \
#         env=kitchen_${kitchen_env}-v3 embedding=${emb} camera=${camera} \
#         exp_name=${emb}@${kitchen_env}@${camera} &
# done
