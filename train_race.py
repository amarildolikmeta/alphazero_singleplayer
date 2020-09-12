from envs.race_strategy_model.prediction_model import RaceStrategyModel

model = RaceStrategyModel(year=2019)
model.train()
model.save()