import model

kwargs = {
    'method': 'DOP853'
}

mod = model.LotkaVolterraModel(**kwargs)
model.run_training(mod, model_name='a2c_new_param', path='/Users/yunliqi/DRL_Personalized_AT/a2c/torch_training/')
