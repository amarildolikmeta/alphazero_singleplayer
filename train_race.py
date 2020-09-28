from envs.race_strategy_model.prediction_model import RaceStrategyModel

model = RaceStrategyModel(year=2019, n_cores=6)
model.train()
model.save()