from src.pricing.black_scholes import call_price


class HedgingPortfolio:
    def __init__(self, strike, r, sigma, transaction_cost=0.001):
        self.strike = strike
        self.r = r
        self.sigma = sigma
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        self.stock_position = 0.0
        self.cash = 0.0
        self.prev_portfolio_value = 0.0

    def option_liability(self, stock_price, time_to_maturity):
        return call_price(
            S=stock_price,
            K=self.strike,
            T=time_to_maturity,
            r=self.r,
            sigma=self.sigma
        )

    def rebalance_hedge(self, stock_price, target_position):
        trade_size = target_position - self.stock_position
        trade_cost = abs(trade_size) * stock_price * self.transaction_cost
        self.cash -= trade_size * stock_price
        self.cash -= trade_cost
        self.stock_position = target_position
        return trade_cost

    def portfolio_value(self, stock_price, time_to_maturity):
        option_value = self.option_liability(stock_price, time_to_maturity)
        stock_value = self.stock_position * stock_price
        total_value = self.cash + stock_value - option_value
        return total_value