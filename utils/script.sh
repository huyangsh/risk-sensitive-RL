# Toy-10
python ./NN_train.py --env Toy-10 --beta 0.01 --tau 1.0 --dim_emb 8 --num_train 2000 --batch_size 10000 --disp_V_opt --disp_V_pi --disp_policy

# Toy-100
python ./NN_train.py --env Toy-100 --beta 0.01 --tau 1.0 --dim_emb 100 --num_train 2000 --batch_size 10000 --disp_V_opt --disp_V_pi --disp_policy

# Toy-1000
# Warning: evaluation is very slow.
python ./NN_train.py --env Toy-1000 --beta 0.01 --tau 1.0 --dim_emb 100 --num_train 2000 --freq_eval 200 --thres_eval 0.01 --batch_size 10000 --disp_V_opt --disp_V_pi --disp_policy

# Eval
python ./NN_eval.py --agent_prefix ./log/active/<agent_prefix> --disp_V_opt --disp_V_pi --disp_policy
python ./NN_eval.py --agent_prefix ./log/active/Toy-10_0.15_2000_20_10000_0.5_1.0_20230514_113537 --disp_V_opt --disp_V_pi --disp_policy