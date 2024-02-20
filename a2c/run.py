import model
import torch


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print('Device used:', device)
kwargs = {
    'method': 'DOP853'
}

mod = model.ABMModel(**kwargs)
model.run_training(mod, batch_operation=False, treatment_period=7, device=device, model_name='a2c_test', path='/home/leo/DRL_Personalized_AT/a2c/torch_training/', num_episodes=200000, save_interval=2000)

# model.run_prediction(mod, treatment_period=30, model_store_path='/home/leo/DRL_Personalized_AT/a2c/torch_training/a2c_my_structure/a2c_model_130000.pth')
