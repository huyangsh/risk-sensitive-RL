python ./NN_train.py --env Toy-10 --beta 0.01 --num_train 2000 --batch_size 10000 --disp_V_opt --disp_V_pi --disp_policy
python ./NN_eval.py --agent_prefix ./log/Toy-10_0.15_2000_20_10000_0.5_0.1_20230514_110813 --disp_V_opt --disp_V_pi --disp_policy