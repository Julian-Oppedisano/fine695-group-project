import inspect
import pytorch_tabular
from pytorch_tabular.models.base_model import BaseModel
import pytorch_lightning

print(f"--- Pytorch Tabular ---")
print(f"Version: {pytorch_tabular.__version__}")
try:
    sig_val_batch_end = inspect.signature(BaseModel.on_validation_batch_end)
    print(f"Signature of BaseModel.on_validation_batch_end: {sig_val_batch_end}")
    print(f"Parameters: {list(sig_val_batch_end.parameters.keys())}")
except AttributeError:
    print("BaseModel.on_validation_batch_end not found or not a method.")

print(f"\n--- Pytorch Lightning ---")
print(f"Version: {pytorch_lightning.__version__}")

# You can also check other hooks if needed, for example:
# try:
#     sig_train_epoch_end = inspect.signature(BaseModel.on_train_epoch_end)
#     print(f"\nSignature of BaseModel.on_train_epoch_end: {sig_train_epoch_end}")
#     print(f"Parameters: {list(sig_train_epoch_end.parameters.keys())}")
# except AttributeError:
#     print("\nBaseModel.on_train_epoch_end not found or not a method.") 