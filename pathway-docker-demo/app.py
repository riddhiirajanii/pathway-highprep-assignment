import pathway as pw

# Define schema for finance trades
class Trade(pw.Schema):
    date: str
    stock: str
    buy_price: float
    sell_price: float
    quantity: int

# Read CSV file
trades = pw.io.csv.read("data/trades.csv", schema=Trade)

# Compute profit for each trade
trades_with_profit = trades.select(
    pw.this.date,
    pw.this.stock,
    pw.this.buy_price,
    pw.this.sell_price,
    pw.this.quantity,
    profit=(pw.this.sell_price - pw.this.buy_price) * pw.this.quantity
)

# Filter profitable trades
profitable_trades = trades_with_profit.filter(pw.this.profit > 0)

# Compute total daily profit across all stocks
daily_profit = (
    profitable_trades.groupby(pw.this.date)
    .reduce(total_profit=pw.reducers.sum(pw.this.profit))
)

# Compute cumulative profit per stock
cumulative_profit = (
    profitable_trades.groupby(pw.this.stock)
    .reduce(total_profit=pw.reducers.sum(pw.this.profit))
)

# Write outputs to JSONL
pw.io.jsonlines.write(profitable_trades, "out/profitable_trades.jsonl")
pw.io.jsonlines.write(daily_profit, "out/daily_profit.jsonl")
pw.io.jsonlines.write(cumulative_profit, "out/stock_profit.jsonl")

print("Starting Pathway finance pipeline with aggregation...")
pw.run()
