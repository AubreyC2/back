from models.test_nobreak_bigonecombined import GroebnerRNN

# ===== LOAD TRAINED MODEL =====
load_path = r"C:\Users\HP\PycharmProjects\PythonProject4\saved_models\groebner_model.pth"

# Recreate the model architecture EXACTLY as before
model = GroebnerRNN(input_size=4, hidden_size=24, output_size=1, num_generators=5)

# Load the saved weights
model.load_state_dict(torch.load(load_path, weights_only=True))
model.eval()  # set to evaluation mode

print("Model loaded successfully from:", load_path)
