MODE=$1



# if MODE is not score 
if [ "$MODE" != "score_only" ];
then 
    python leaderboard/data_dir/_create_tables.py pairwise-gpt4t -1 &
    python leaderboard/data_dir/_create_tables.py pairwise-llama -1 &
    python leaderboard/data_dir/_create_tables.py pairwise-haiku -1 &

    python leaderboard/data_dir/_create_tables.py pairwise-gpt4t 500 &
    python leaderboard/data_dir/_create_tables.py pairwise-llama 500 &
    python leaderboard/data_dir/_create_tables.py pairwise-haiku 500 &

    python leaderboard/data_dir/_create_tables.py pairwise-gpt4t 1000 &
    python leaderboard/data_dir/_create_tables.py pairwise-llama 1000 &
    python leaderboard/data_dir/_create_tables.py pairwise-haiku 1000 &

    python leaderboard/data_dir/_create_tables.py pairwise-gpt4t 1500 &
    python leaderboard/data_dir/_create_tables.py pairwise-llama 1500 &
    python leaderboard/data_dir/_create_tables.py pairwise-haiku 1500 &
fi 
wait 

# Score only 
python leaderboard/data_dir/_create_tables.py score

python leaderboard/data_dir/_merge_results.py

# margin=3;tie_margin=2;K=4;dynamic=True;interval=16
margin=2;tie_margin=2;K=4;dynamic=True;interval=100;LM=1500
python -m leaderboard.wb_elo --K $K --margin $margin --tie_margin $tie_margin --num_rounds 128 --dynamic $dynamic --interval $interval --num_processes 4 --length_margin $LM

python leaderboard/data_dir/_merge_results.py

if [ "$MODE" == "score_only" ];
then 
    python leaderboard/show_table.py --mode taskwise_score
else 
    python leaderboard/show_table.py --mode main
fi
