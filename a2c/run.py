import model

kwargs = {
    'method': 'DOP853'
}

mod = model.LotkaVolterraModel(**kwargs)
model.run_training(mod, model_name='a2c_no_punishment')
