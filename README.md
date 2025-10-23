# pathway-highprep-assignment
FOLDER 1: pathway-docker-demo
This is a program which from a CSV of stock trades:
->Calculates profit/loss for each trade.
->Filters profitable trades.
->Computes daily total profit.
->Computes cumulative profit per stock.

to run this program, in the terminal:

docker build -t pathway-demo .
docker run --rm -v "$(pwd)/data":/app/data -v "$(pwd)/out":/app/out pathway-demo

FOLDER 2: pathway-task2
To run this program, in the terminal:
python simulation_without_pathway.py
python simulation_with_pathway.py

you can see the output in /out folder.