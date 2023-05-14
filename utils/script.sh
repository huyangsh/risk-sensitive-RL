# Toy-10
python ./NN_train.py --env Toy-10 --beta 0.01 --tau 1.0 --dim_emb 8 --num_train 2000 --batch_size 10000 --disp_V_opt --disp_V_pi --disp_policy

# Toy-100
python ./NN_train.py --env Toy-100 --beta 0.01 --tau 1.0 --dim_emb 100 --num_train 2000 --batch_size 10000 --disp_V_opt --disp_V_pi --disp_policy

# Toy-1000
# Warning: evaluation is very slow.
python ./NN_train.py --env Toy-1000 --beta 0.01 --tau 1.0 --dim_emb 300 --num_train 10000 --freq_eval 2000 --thres_eval 0.01 --batch_size 10000 --disp_V_opt --disp_V_pi --disp_policy

# CartPole.
python ./NN_train.py --env CartPole --beta 0.01 --tau 1.0 --dim_emb 6 --num_train 2000 --batch_size 10000
python ./NN_online_train.py --env CartPole --beta 0.01 --tau 1.0 --dim_emb 6 --T_train 100000 --batch_size 10000 --buffer_size 1000000 --off_ratio 0.1

# Pendulum.
python ./NN_train.py --env Pendulum --beta 0.01 --tau 1.0 --num_actions 2 --dim_emb 3 --num_train 2000 --batch_size 10000

# Eval
python ./NN_eval.py --agent_prefix ./log/active/<agent_prefix> --disp_V_opt --disp_V_pi --disp_policy
python ./NN_eval.py --agent_prefix ./log/selected/CartPole_0.0_2000_20_10000_0.5_1.0_20230514_131851 --disp_V_opt --disp_V_pi --disp_policy